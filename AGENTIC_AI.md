# Agentic AML Investigator

The FINDETECT pipeline already produces a decision: a risk score, a tier, graph
exposure and a templated SAR draft for each account/day. What it did not do is
*investigate*. The dashboard chat was a single LLM call over a small, fixed
snippet of the alert, so it could not answer questions like "which seed accounts
did this account pay?" or "show me the largest outflows on the alert date".

This layer adds a tool-using agent on top of the pipeline's saved evidence. The
LLM chooses read-only evidence tools, reads their JSON results, and writes an
answer that is traceable to those results. The pipeline itself is unchanged and
still deterministic.

---

## 1. What was added

| File | Purpose |
|---|---|
| `tools.py` | The three read-only evidence tools, their JSON schemas, a data store, and a dispatcher. |
| `agent.py` | Tool-calling loop over Ollama, with a JSON-protocol fallback and a deterministic offline memo. |
| `agent_cli.py` | Run the tools or the agent without the web server. |
| `main.py` | `/chat` remains for compatibility; the new `/api/cases/{case_id}/chat` binds the selected account/day and audits the AI action. |
| `case_service.py` | Resolves account/day cases and exposes transactions with provenance and currency grouping. |
| `frontend/assistant.html` | Case-aware assistant, evidence chips, live timer, stop, degraded/offline states. |
| `tests/` | 97 backend/real-data/contract tests plus 5 Node frontend logic tests; all pass. |

No new runtime dependencies were added. The tools reuse `pandas` and `duckdb`; the
agent reuses the installed `ollama` client. DOCX/PDF export is implemented with
in-tree writers because the environment could not install extra packages.

---

## 2. The three tools

All tools are **read-only**, validate every argument, bounded in output size, and
always return a JSON envelope instead of raising:

```json
{
  "ok": true,
  "tool": "get_account_scores",
  "account_id": "100428660",
  "data": { "...": "..." },
  "meta": { "run_label": "...", "source": "parquet", "elapsed_ms": 180 }
}
```

A failure is explicit and safe to show the model:

```json
{
  "ok": false,
  "tool": "get_account_scores",
  "error": {"code": "ACCOUNT_NOT_FOUND", "message": "...", "hint": "..."}
}
```

### `get_account_scores(account_id, date=None)`

Risk scores for one account, optionally one day. Returns the final score, risk
label, investigation tier/flag, base risk, graph penalty, the four model
components (XGBoost, anomaly, behavioural, meta), graph features (seed flag, hops,
bidirectional count, fan score, degree, volume), behavioural deviations, the
working thresholds, the account's scored-day history, and the population
percentile of the score on that day.

Use it first for any case. Omit `date` to get every scored day plus the
`highest_risk_day`.

### `get_transactions(account_id, date=None, direction="both", limit=25, min_amount=None)`

Transaction evidence for one account/day:

- totals: count, inbound/outbound count and amount, net flow, largest, average;
- unique counterparties, payment formats;
- top counterparties by volume with inbound/outbound split;
- the most recent transactions (timestamp, direction, counterparty, amount,
  currency, payment format, banks).

`date` defaults to the account's latest scored day. `limit` is clamped to 100 and
the summary is always computed over the full filtered set, not just the returned
rows. This matters for super-nodes: account `100428660` has 28,878 outbound
transactions on 2022-10-23, and the tool still answers in well under a second.

### `get_graph_exposure(account_id, date=None)`

Network exposure for one account/day:

- one-hop seed exposure: seed accounts, direction (inbound/outbound), amounts and
  transaction counts;
- pre-computed two-hop and three-hop exposure counts;
- indicators: bidirectional layering, mule fan-in/fan-out, high-degree hub;
- the graph penalty, its components (rescaled to the capped penalty so their
  shares stay meaningful) and the cap flag;
- the pipeline's interpretation notes.

Only one-hop counterparty identities are available; two/three-hop values are the
aggregates the graph stage already computed.

---

## 3. How the agent works

```text
investigator question
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Prefetch evidence pack (no LLM):                         │
│    scores (for the case date) → transactions → graph        │
│    ~3-6 KB of compact JSON in the system prompt             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Ollama chat with tool schemas                            │
│      • native tool_calls?  → execute, append result, loop   │
│      • JSON tool request?  → execute, append result, loop   │
│      • plain answer?       → return                         │
│    max_steps rounds, then one forced answer without tools   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
answer + tool trace + model + elapsed time + degraded flag
```

The **prefetch** is what makes the common question fast and reliable. The model
starts with a labelled, plain-language evidence brief (scores, flow totals with
explicit `inbound_total`/`outbound_total` labels, and network exposure), so it
does not need a tool round just to find out what happened. Tools remain available
for a different date, more rows, or a filtered view.

The **case date** comes from the selected dashboard alert (`graph_date`) or, in
the CLI, from `--date`. If no date is given, the agent uses the account's
highest-risk scored day. If a requested date has no scores, it falls back to the
highest-risk day and adds a warning to the response.

### Fallbacks

1. **Native tool calls unsupported** – the agent retries without `tools=` and
   appends a strict JSON protocol to the system prompt:
   `{"tool": "...", "arguments": {...}}`. Tool requests are parsed from plain
   text, JSON fences, or a single object.
2. **Model timeout / connection error / empty answer** – the agent returns a
   deterministic memo assembled from the evidence it already collected. It is
   labelled **Offline mode** and keeps the same numbers, counterparties and
   limitations. `/chat` therefore no longer returns HTTP 500 when Ollama is down.

### Guardrails

- Read-only: no tool writes, mutates, or deletes anything.
- Account ids are validated against a strict pattern; dates must be real
  `YYYY-MM-DD` values; directions are enumerated; limits are clamped.
- No string interpolation into SQL: DuckDB queries are parameterised.
- Tool output is bounded: at most 100 transaction rows, 31 scored days, 5 top
  counterparties, 5 seed neighbours per direction.
- Tool errors are returned to the model as structured errors, not exceptions.
- Model answers are instructed to distinguish evidence from inference and to
  never assert guilt. Every answer retains the human-review notice.
- A per-request tool audit is logged (`tool`, `account`, `ok`, `ms`); tool
  payloads are not logged.

---

## 4. Data sources and binding

`AMLDataStore` serves one pipeline run at a time. It can be backed by:

- **Parquet** (normal): `outputs/FINDETECT_Pipeline_Output/<run>/risk_scores.parquet`
  plus the transactions parquet. Queries use DuckDB projection and predicate
  pushdown, so a single account is fetched without loading 1.5M risk rows or 2.7M
  transactions into memory.
- **In-memory DataFrames** (tests, or a run whose outputs failed to save).

`main.py` binds the store:

- at import/startup, to the newest saved run (so chat works after a server
  restart without re-running the pipeline);
- after every `/pipeline/analyze`, to that run's saved `risk_scores.parquet` and
  the exact daily transactions file used for the run.

The run's `summary.json` now records `input_parquet`, and the CLI uses it to find
the right transactions file. Saved SAR drafts are re-associated with their
accounts on startup by parsing `sar_reports.jsonl`, so the agent keeps the
template report as context across restarts.

---

## 5. API

### `POST /chat`

```json
{
  "case_id": "100428660",
  "message": "Why was this account flagged on 2022-10-23?",
  "history": [{"role": "user", "content": "..."}]
}
```

Response (abridged):

```json
{
  "message": "success",
  "case_id": "100428660",
  "response": "## Assessment\n...",
  "answer": "## Assessment\n...",
  "tool_calls": [
    {"name": "get_account_scores", "arguments": {"account_id": "100428660", "date": "2022-10-23"},
     "ok": true, "elapsed_ms": 180, "source": "auto", "summary": "2022-10-23: score 0.982 (High Risk, TIER_1_HIGH)", "error": null},
    {"name": "get_transactions", "arguments": {"account_id": "100428660", "date": "2022-10-23", "limit": 10},
     "ok": true, "elapsed_ms": 401, "source": "model", "summary": "28878 tx on 2022-10-23; in 0.0, out 4271607921.96", "error": null}
  ],
  "steps": 2,
  "model": "qwen2.5:3b-instruct",
  "degraded": false,
  "warnings": [],
  "elapsed_ms": 184000
}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/cases/{case_id}/chat` | Case-aware assistant; resolves account/day, runs the agent and audits `AI_QUERY`. |
| `POST /chat` | Legacy agentic Q&A with tool trace. |
| `GET /agent/tools` | Tool schemas, model, bound run and readiness. |
| `POST /agent/tool` | `{"name": "...", "arguments": {...}}` — call one tool directly. |
| `POST /agent/review` | Standard investigation memo. |
| `POST /pipeline/analyze` | Unchanged, now also returns `agent_ready` and binds the run. |

`FINDETECT_AGENT_ENABLED=0` restores the original single-shot `/chat` behaviour
(without tools). The new endpoints still exist but `/agent/review` returns 409.
If no run is bound at all, `/chat` and `/agent/review` return 409 instead of
letting the model answer without evidence; run the pipeline first.

---

## 6. CLI

The CLI works against the latest saved run and does not need the web server. It
also imports far less than `main.py` (no TensorFlow), so it starts quickly.

```powershell
# List tools
python agent_cli.py tools

# Call a tool directly
python agent_cli.py call get_account_scores --args '{\"account_id\": \"100428660\", \"date\": \"2022-10-23\"}'

# Ask the investigator
python agent_cli.py chat --case 100428660 --date 2022-10-23 `
  --message "Why was this account flagged?"

# Deterministic evidence memo (no LLM)
python agent_cli.py chat --case 100428660 --date 2022-10-23 `
  --message "Summarise" --no-llm

# Standard memo
python agent_cli.py review --case 100428660 --date 2022-10-23
```

Use `--output-dir` to select a specific run and `--transactions` to override the
transactions parquet. `--json` emits the full result payload.

---

## 7. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FINDETECT_AGENT_ENABLED` | `1` | Master switch for tool-using chat. |
| `FINDETECT_OLLAMA_TOOL_MODEL` | chat model (`qwen2.5`) | Model used for agent calls. `qwen2.5:3b-instruct` is much faster on CPU. |
| `FINDETECT_OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint. |
| `FINDETECT_AGENT_MAX_STEPS` | `5` | Tool-calling rounds before a forced answer. |
| `FINDETECT_AGENT_NATIVE_TOOLS` | `1` | Use Ollama native tool calling. Set to `0` on slow CPU machines to skip tool schemas (the prefetch and JSON protocol still work). |
| `FINDETECT_AGENT_NUM_CTX` | `8192` | Context window. Ollama's qwen2.5 default (4096) is too small for prompt + evidence + tools. |
| `FINDETECT_AGENT_OLLAMA_TIMEOUT` | `300` | Per-call timeout in seconds (separate from the legacy chat timeout). |
| `FINDETECT_AGENT_KEEP_ALIVE` | `30m` | Keep the model resident between questions. |
| `FINDETECT_AGENT_WARMUP` | `1` | Load the model in the background when the API starts. |
| `FINDETECT_AGENT_MAX_HISTORY` | `10` | Prior user/assistant messages forwarded to the model. |
| `FINDETECT_AGENT_TX_LIMIT`, `..._TX_MAX_LIMIT` | `25`, `100` | Default and maximum transaction rows per tool call. |

Graph penalty constants (`FINDETECT_BIDIR_PENALTY`, `FINDETECT_FAN_BOOST`,
`FINDETECT_TERTIARY_PENALTY_FACTOR`, `FINDETECT_GRAPH_PENALTY_CAP`) are now in
`config.py` and shared by `nodes.py` and `tools.py`, so the tool explanation and
the pipeline cannot drift apart.

---

## 8. Performance notes (important on CPU-only machines)

Tool calls are cheap: 0.1–0.5 s against the 3-day sample. The LLM is the slow
part. On the machine used to develop this, with Ollama on **100% CPU**:

| Model | Prompt processing | Generation | Observed agent answer |
|---|---|---|---|
| `qwen2.5:latest` (7B) | ~19 tokens/s | ~4 tokens/s | ~150 s for a 2.0k-token prompt + 180-token answer |
| `qwen2.5:3b-instruct` | ~42 tokens/s | ~7 tokens/s | ~80 s for a 1.9k-token prompt + 230-token answer |

That is why the agent:

- prefetches the evidence pack (avoids a second model round for common questions),
- uses `num_ctx=8192` rather than Ollama's small default,
- warms the model at API startup,
- shows a live timer in the chat UI,
- falls back to the evidence memo on timeout instead of failing.

In testing on the sample data, the 7B model read the direction labels correctly
(`inbound_total=0.00`, `outbound_total=4,271,607,921.96`). The 3B model misread
inbound/outbound in an early run with a JSON evidence block; after switching to
the labelled brief and the explicit `ANSWER CHECK` lines, it read them correctly
in the final end-to-end check. It is still less reliable than 7B on nuance.
**Use 7B when accuracy matters**; use 3B as a speed trade-off and check the
evidence chips in the UI.

For a responsive demo on CPU, set:

```powershell
$env:FINDETECT_OLLAMA_TOOL_MODEL = "qwen2.5:3b-instruct"
$env:FINDETECT_AGENT_NATIVE_TOOLS = "0"
$env:FINDETECT_AGENT_MAX_STEPS = "2"
```

With native tools off, the evidence brief is still prefetched and the model can
request extra evidence through the JSON protocol. The tool layer and trace are
identical either way.

---

## 9. Tests

```powershell
python -m pytest tests -q
```

The suite uses synthetic DataFrames and a scripted fake Ollama client, so it runs
in seconds and needs no model, no network and no pipeline artifacts. It covers:

- all three tools, aggregates, direction/limit/min-amount filters, percentile,
  seed exposure and penalty components;
- validation errors (bad account, bad date, bad direction, unknown tool, bad
  JSON, missing/extra arguments) and JSON-safe envelopes;
- case-date resolution, requested-date fallback, latest-vs-highest day;
- native tool calls, the JSON text protocol, unsupported-tools retry, max-step
  forcing, history sanitisation, timeouts, empty answers and offline memos.

A manual end-to-end check against real data:

```powershell
python agent_cli.py call get_transactions `
  --args '{\"account_id\": \"100428660\", \"date\": \"2022-10-23\", \"limit\": 5}'
```

---

## 10. Example questions the agent can now answer

- "Why was this account flagged on 2022-10-23?" → prefetched scores + flows +
  network, explained with exact numbers.
- "Which counterparties did it pay the most, and are any of them high-risk
  seeds?" → `get_transactions` + `get_graph_exposure`.
- "Is the risk driven by the model or by graph exposure?" → score components and
  graph penalty breakdown.
- "How did the score change across the three days?" → score history.
- "What is missing before we can decide on a SAR?" → the model must state
  evidence gaps instead of inventing facts.

---

## 11. Limitations and next steps

- The evidence is only as good as the pipeline's stored outputs. The agent cannot
  see KYC, customer profiles, external watchlists or case-management notes.
- Two/three-hop exposure is aggregate only; the tool cannot name a two-hop path.
- The agent does not write SARs or close cases. It explains and gathers evidence
  for a human. Generation of SAR drafts remains in the pipeline.
- Chat latency is dominated by local LLM speed. Streaming responses and a smaller
  evidence pack are the next practical improvements.
- If you expose the API beyond localhost, add authentication, rate limits and a
  strict CORS allowlist. The tools are read-only but still expose customer data.
- Possible extensions: a watchlist/PEP tool, a case-notes tool, a
  `search_accounts` tool, and a LangGraph investigation node that runs
  `review()` automatically for TIER_1 cases.
