import { api, qs, param, caseHref, ApiError } from "./api.js";
import {
  renderChrome, setContent, loadingBlock, emptyState, errorState, requireCase,
  escapeHtml, fmtScore, fmtNumber, fmtMoney, fmtDate, fmtDateTime,
  riskBadge, statusBadge, sarBadge, icon, toast, setBusy, renderMarkdown,
} from "./ui.js";

const caseId = param("case_id");
const initialRun = param("run");
let runId = initialRun;
let caseData = null;
let activeTab = (window.location.hash || "#overview").slice(1);
let notesCache = [];
let editingNoteId = null;

const dialog = document.getElementById("detail-dialog");
document.getElementById("dialog-close").textContent = "×";
document.getElementById("dialog-close").addEventListener("click", () => dialog.close());
document.getElementById("dialog-ok").addEventListener("click", () => dialog.close());

function openDialog(title, html) {
  document.getElementById("dialog-title").textContent = title;
  document.getElementById("dialog-body").innerHTML = html;
  if (!dialog.open) dialog.showModal();
}

function closeDialog() { if (dialog.open) dialog.close(); }

if (!requireCase(caseId)) {
  closeDialog();
} else {
  renderChrome({ active: "cases", caseId, runId, title: "Case workspace", subtitle: "Loading case…" });
  loadCase();
}

function runQuery() { return qs({ run_id: runId }); }

async function loadCase() {
  const content = document.getElementById("content");
  content.innerHTML = loadingBlock("Loading case evidence…");
  try {
    caseData = await api(`/api/cases/${encodeURIComponent(caseId)}${runQuery()}`);
    runId = caseData.run_id;
    renderChrome({
      active: "cases",
      caseId,
      runId,
      title: caseData.case_id,
      subtitle: `${caseData.holder_name || "Source-record holder unavailable"} · ${caseData.graph_date}`,
    });
    render();
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      setContent(emptyState("Case not found", "The case id does not exist in this run. Return to the queue and select a real account/day case.", `<a class="btn" href="index.html">${icon("cases")} Case queue</a>`));
    } else if (error instanceof ApiError && error.status === 401) {
      setContent(errorState("Authentication required", "Set the API token in Settings, then reload this case.", "retry") + `<div style="margin-top:12px"><a class="btn" href="settings.html">Open Settings</a></div>`);
      document.getElementById("retry")?.addEventListener("click", loadCase);
    } else {
      setContent(errorState("Could not load case", error.message, "retry"));
      document.getElementById("retry")?.addEventListener("click", loadCase);
    }
  }
}

function render() {
  const c = caseData;
  const sar = c.sar || {};
  const tx = c.transaction_summary || {};
  const tier = String(c.tier || "").replaceAll("_", " ");
  const content = setContent(`
    <div class="between" style="margin-bottom:12px">
      <div class="flex small muted" style="gap:6px">
        <a href="index.html" class="flex" style="gap:4px">${icon("cases")} Cases</a>
        <span>/</span>
        <span class="mono">${escapeHtml(c.case_id)}</span>
      </div>
      <div class="flex">
        <button class="btn btn-secondary" id="refresh-case">${icon("refresh")} Refresh</button>
        <a class="btn btn-secondary" href="${caseHref("graph.html", c.case_id, runId)}">${icon("graph")} Graph</a>
        <a class="btn btn-secondary" href="${caseHref("assistant.html", c.case_id, runId)}">${icon("chat")} AI Assistant</a>
        <a class="btn" href="${caseHref("sar.html", c.case_id, runId)}">${icon("sar")} SAR draft</a>
      </div>
    </div>

    <div class="card" style="margin-bottom:14px">
      <div class="between" style="align-items:flex-start">
        <div style="min-width:0">
          <div class="flex" style="gap:8px;flex-wrap:wrap">
            <span class="badge badge-teal mono">${escapeHtml(c.case_id)}</span>
            ${riskBadge(c.risk_label, c.tier)}
            ${statusBadge(c.status)}
            ${c.is_high_risk_seed ? '<span class="badge badge-high">High-risk seed</span>' : ""}
            ${c.received_from_high_risk ? '<span class="badge badge-medium">One-hop graph link</span>' : ""}
          </div>
          <h1 style="font-size:22px;margin-top:10px">Account ${escapeHtml(c.account_id)}</h1>
          <p class="muted small" style="margin-top:3px">${escapeHtml(c.holder_name || "Source-record holder name unavailable")} · ${escapeHtml(c.graph_date)} · ${escapeHtml(tier)}</p>
          <p class="tiny muted" style="margin-top:6px">Model score is a prioritisation aid, not a legal determination. Transactions and sources are real records from the selected pipeline run.</p>
        </div>
        <div class="right" style="min-width:150px">
          <div class="kpi-label">Final risk score</div>
          <div class="kpi-value" style="font-size:30px">${fmtScore(c.risk_score)}</div>
          <div class="tiny muted">Investigation threshold ${fmtScore((c.thresholds || {}).investigation)}</div>
        </div>
      </div>
      <div class="between" style="margin-top:14px;padding-top:12px;border-top:1px solid var(--line);gap:10px">
        <div class="flex small">
          <label class="muted tiny" for="case-status">CASE STATUS</label>
          <select class="select" id="case-status" style="width:auto">
            ${["NEW", "IN_REVIEW", "WAITING_INFO", "ESCALATED", "CLOSED"].map((value) => `<option value="${value}" ${value === c.status ? "selected" : ""}>${value.replace("_", " ")}</option>`).join("")}
          </select>
          <button class="btn btn-secondary btn-sm" id="save-status">Update status</button>
        </div>
        <div class="tiny muted">Run <span class="mono">${escapeHtml(runId)}</span> · ${c.notes_count || 0} note(s)</div>
      </div>
    </div>

    <div class="tabs" role="tablist">
      <button class="tab ${activeTab === "overview" ? "active" : ""}" data-tab="overview">Overview</button>
      <button class="tab ${activeTab === "transactions" ? "active" : ""}" data-tab="transactions">Transactions</button>
      <button class="tab ${activeTab === "notes" ? "active" : ""}" data-tab="notes">Notes & findings</button>
      <button class="tab ${activeTab === "timeline" ? "active" : ""}" data-tab="timeline">Timeline</button>
    </div>
    <div id="tab-content"></div>
  `);

  content.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  document.getElementById("refresh-case").addEventListener("click", loadCase);
  document.getElementById("save-status").addEventListener("click", updateStatus);
  switchTab(activeTab, false);
}

function switchTab(tab, updateHash = true) {
  activeTab = tab;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  if (updateHash) {
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#${tab}`);
  }
  const target = document.getElementById("tab-content");
  if (!target) return;
  if (tab === "overview") renderOverview(target);
  if (tab === "transactions") renderTransactionsTab(target);
  if (tab === "notes") renderNotesTab(target);
  if (tab === "timeline") renderTimelineTab(target);
}

/* ------------------------------ Overview ------------------------------ */
function renderOverview(target) {
  const c = caseData;
  const tx = c.transaction_summary || {};
  const graph = c.graph_features || {};
  const models = c.model_scores || {};
  const thresholds = c.thresholds || {};
  const sar = c.sar || {};
  const aggregates = tx.aggregates || [];
  target.innerHTML = `
    <div class="grid grid-4">
      <div class="card"><div class="kpi"><div class="kpi-label">Base ML risk</div><div class="kpi-value sm">${fmtScore(c.base_risk)}</div><div class="kpi-note">Models before graph penalty</div></div></div>
      <div class="card"><div class="kpi"><div class="kpi-label">Graph penalty</div><div class="kpi-value sm">${fmtScore(c.graph_penalty)}</div><div class="kpi-note">Capped contribution</div></div></div>
      <div class="card"><div class="kpi"><div class="kpi-label">Transactions</div><div class="kpi-value sm">${fmtNumber(tx.transaction_count)}</div><div class="kpi-note">${escapeHtml((tx.currencies || []).join(", ") || "No currency data")}</div></div></div>
      <div class="card"><div class="kpi"><div class="kpi-label">SAR drafts</div><div class="kpi-value sm">${sar.draft_count || 0}</div><div class="kpi-note">${sar.latest ? `Latest ${escapeHtml(sar.latest.status)}` : "Not drafted"}</div></div></div>
    </div>

    <div class="split" style="margin-top:14px">
      <div class="card">
        <div class="card-header"><div><div class="card-title">Risk score composition</div><div class="card-sub">Stored model outputs and graph features. No per-feature attribution is available.</div></div></div>
        <div class="table-wrap">
          <table class="table">
            <thead><tr><th>Signal</th><th class="num">Value</th><th>Interpretation</th></tr></thead>
            <tbody>
              <tr><td>Final risk score</td><td class="num"><strong>${fmtScore(c.risk_score)}</strong></td><td>Clipped to 1.000; prioritisation aid</td></tr>
              <tr><td>Base ML risk</td><td class="num">${fmtScore(c.base_risk)}</td><td>Fused model score before graph penalty</td></tr>
              <tr><td>Graph penalty</td><td class="num">${fmtScore(c.graph_penalty)}</td><td>Capped graph contribution</td></tr>
              <tr><td>XGBoost</td><td class="num">${fmtScore(models.xgboost)}</td><td>Supervised component</td></tr>
              <tr><td>Anomaly</td><td class="num">${fmtScore(models.anomaly)}</td><td>Autoencoder reconstruction component</td></tr>
              <tr><td>Behavioral</td><td class="num">${fmtScore(models.behavioral)}</td><td>Deviation/velocity component</td></tr>
              <tr><td>Meta score</td><td class="num">${fmtScore(models.meta)}</td><td>Calibrated fusion output</td></tr>
            </tbody>
          </table>
        </div>
        <div class="notice" style="margin-top:10px">The pipeline does not persist SHAP or feature-importance values. This page shows only components that actually exist in the run.</div>
      </div>
      <div class="stack">
        <div class="card">
          <div class="card-title">Thresholds</div>
          <table class="table" style="min-width:0;margin-top:8px"><tbody>
            <tr><td>Investigation</td><td class="num">${fmtScore(thresholds.investigation)}</td></tr>
            <tr><td>Medium risk</td><td class="num">${fmtScore(thresholds.medium_risk)}</td></tr>
            <tr><td>Tier 1 high</td><td class="num">${fmtScore(thresholds.tier_1_high)}</td></tr>
            <tr><td>Tier 2 medium</td><td class="num">${fmtScore(thresholds.tier_2_medium)}</td></tr>
          </tbody></table>
        </div>
        <div class="card">
          <div class="card-title">Network signals</div>
          <table class="table" style="min-width:0;margin-top:8px"><tbody>
            <tr><td>Direct seed links</td><td class="num">${fmtNumber(graph.hop1_count)}</td></tr>
            <tr><td>Two-hop links</td><td class="num">${fmtNumber(graph.hop2_count)}</td></tr>
            <tr><td>Bidirectional</td><td class="num">${fmtNumber(graph.bidir_count)}</td></tr>
            <tr><td>Fan score</td><td class="num">${fmtScore(graph.fan_score)}</td></tr>
            <tr><td>Daily degree</td><td class="num">${fmtNumber(graph.daily_degree)}</td></tr>
          </tbody></table>
          <p class="tiny muted" style="margin-top:8px">Graph links are investigative leads, not proof of common ownership or intent.</p>
        </div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-top:14px">
      <div class="card">
        <div class="card-header"><div><div class="card-title">Activity by currency</div><div class="card-sub">No FX conversion is applied. Totals are per source currency.</div></div></div>
        ${aggregates.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Direction</th><th>Currency</th><th class="num">Count</th><th class="num">Total</th></tr></thead><tbody>
          ${aggregates.map((row) => `<tr><td>${escapeHtml(row.direction)}</td><td>${escapeHtml(row.currency || "—")}</td><td class="num">${fmtNumber(row.count)}</td><td class="num">${fmtMoney(row.total)}</td></tr>`).join("")}
        </tbody></table></div>` : `<p class="muted small">No transaction summary available.</p>`}
        <div style="margin-top:10px"><a class="btn btn-secondary btn-sm" href="#transactions" data-open-tab="transactions">${icon("arrow")} Open transaction explorer</a></div>
      </div>
      <div class="card">
        <div class="card-header"><div><div class="card-title">Case narrative & SAR</div><div class="card-sub">Pipeline draft and investigator workflow</div></div></div>
        <div class="stack">
          <div class="notice ${sar.latest && sar.latest.status === "APPROVED" ? "ok" : "warn"}">
            ${sar.latest ? `Latest SAR draft v${sar.latest.version} is <strong>${escapeHtml(sar.latest.status)}</strong>.` : "No SAR draft has been created for this case."}
          </div>
          <div class="flex flex-wrap">
            <a class="btn" href="${caseHref("sar.html", c.case_id, runId)}">${icon("sar")} ${sar.latest ? "Open SAR workspace" : "Create SAR draft"}</a>
            ${c.legacy_report ? `<button class="btn btn-secondary" id="view-legacy">${icon("external")} View pipeline draft</button>` : ""}
          </div>
          ${c.legacy_report ? `<p class="tiny muted">A legacy template draft exists from the pipeline run. It is shown for context and is not a completed SAR.</p>` : ""}
          <p class="tiny muted">${sar.legacy_available ? "Legacy draft available." : "No legacy report for this case."} Approval is always a human action; no regulator filing occurs.</p>
        </div>
      </div>
    </div>
  `;
  target.querySelectorAll("[data-open-tab]").forEach((anchor) => {
    anchor.addEventListener("click", (event) => { event.preventDefault(); switchTab(anchor.dataset.openTab); });
  });
  const legacy = document.getElementById("view-legacy");
  if (legacy) legacy.addEventListener("click", () => openDialog("Pipeline template draft", `<div class="copy">${renderMarkdown(c.legacy_report)}</div><div class="notice warn" style="margin-top:10px">This legacy draft contains no transaction evidence and was generated from a template. Use the SAR workspace for a new evidence-grounded draft.</div>`));
}

/* ---------------------------- Transactions ---------------------------- */
const txState = { page: 1, direction: "both", min_amount: "", max_amount: "", counterparty: "" };

function renderTransactionsTab(target) {
  target.innerHTML = `
    <div class="card" style="margin-bottom:12px">
      <form id="tx-filters" class="filters">
        <div class="field"><label for="tx-direction">Direction</label><select class="select" id="tx-direction"><option value="both">Both</option><option value="in">Inbound</option><option value="out">Outbound</option></select></div>
        <div class="field"><label for="tx-min">Min amount</label><input class="input" id="tx-min" type="number" min="0" step="0.01" value="${escapeHtml(txState.min_amount)}"></div>
        <div class="field"><label for="tx-max">Max amount</label><input class="input" id="tx-max" type="number" min="0" step="0.01" value="${escapeHtml(txState.max_amount)}"></div>
        <div class="field wide"><label for="tx-cp">Counterparty</label><input class="input" id="tx-cp" placeholder="Account id" value="${escapeHtml(txState.counterparty)}"></div>
        <div class="field" style="flex:0 0 auto"><label>&nbsp;</label><button class="btn" type="submit">${icon("search")} Filter</button></div>
        <div class="field" style="flex:0 0 auto"><label>&nbsp;</label><button class="btn btn-ghost" type="button" id="tx-reset">Reset</button></div>
      </form>
    </div>
    <div class="card card-flush">
      <div class="between" style="padding:12px 16px;border-bottom:1px solid var(--line)">
        <div class="card-title">Transaction evidence</div>
        <div class="tiny muted" id="tx-aggregates">Loading…</div>
      </div>
      <div class="table-wrap" style="border:0;border-radius:0">
        <table class="table"><thead><tr><th>Reference</th><th>Time</th><th>Direction</th><th>Counterparty</th><th class="num">Amount</th><th>Currency</th><th>Format</th></tr></thead><tbody id="tx-body"><tr><td colspan="7" class="empty-row">Loading transactions…</td></tr></tbody></table>
      </div>
      <div class="pagination" style="padding:12px 16px"><div class="tiny muted" id="tx-page-info"></div><div class="page-buttons"><button class="btn btn-secondary btn-sm" id="tx-prev">Previous</button><button class="btn btn-secondary btn-sm" id="tx-next">Next</button></div></div>
    </div>
    <p class="tiny muted" style="margin-top:8px">Amounts are direction-specific source amounts. The source has no native transaction id; <span class="mono">TX-…</span> is a deterministic provenance reference.</p>
  `;
  const form = document.getElementById("tx-filters");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    txState.direction = document.getElementById("tx-direction").value;
    txState.min_amount = document.getElementById("tx-min").value;
    txState.max_amount = document.getElementById("tx-max").value;
    txState.counterparty = document.getElementById("tx-cp").value.trim();
    txState.page = 1;
    loadTransactions();
  });
  document.getElementById("tx-reset").addEventListener("click", () => {
    Object.assign(txState, { page: 1, direction: "both", min_amount: "", max_amount: "", counterparty: "" });
    renderTransactionsTab(target);
  });
  document.getElementById("tx-prev").addEventListener("click", () => { if (txState.page > 1) { txState.page -= 1; loadTransactions(); } });
  document.getElementById("tx-next").addEventListener("click", () => { if (txState.page < txPages) { txState.page += 1; loadTransactions(); } });
  loadTransactions();
}

let txPages = 1;

async function loadTransactions() {
  const body = document.getElementById("tx-body");
  if (!body) return;
  body.innerHTML = `<tr><td colspan="7" class="empty-row">Loading transactions…</td></tr>`;
  try {
    const data = await api(`/api/cases/${encodeURIComponent(caseId)}/transactions${qs({
      run_id: runId,
      direction: txState.direction,
      min_amount: txState.min_amount,
      max_amount: txState.max_amount,
      counterparty: txState.counterparty,
      page: txState.page,
      page_size: 25,
    })}`);
    txPages = data.page_count;
    document.getElementById("tx-aggregates").textContent = (data.aggregates || []).map((row) => `${row.direction} ${fmtMoney(row.total, row.currency)} (${row.count})`).join(" · ") || "No rows";
    document.getElementById("tx-page-info").textContent = data.total ? `Page ${data.page} of ${data.page_count} · ${fmtNumber(data.total)} rows` : "No rows";
    document.getElementById("tx-prev").disabled = data.page <= 1;
    document.getElementById("tx-next").disabled = data.page >= data.page_count;
    body.innerHTML = data.items.length ? data.items.map((item) => `
      <tr class="clickable" data-tx='${escapeHtml(JSON.stringify(item))}'>
        <td class="mono tiny">${escapeHtml(item.transaction_ref)}</td>
        <td class="nowrap">${escapeHtml(item.timestamp.replace("T", " "))}</td>
        <td>${item.direction === "in" ? '<span class="badge badge-low">In</span>' : item.direction === "out" ? '<span class="badge badge-medium">Out</span>' : '<span class="badge badge-neutral">Self</span>'}</td>
        <td class="mono">${escapeHtml(item.counterparty)}</td>
        <td class="num">${fmtMoney(item.amount)}</td>
        <td>${escapeHtml(item.currency || "—")}</td>
        <td>${escapeHtml(item.payment_format || "—")}</td>
      </tr>`).join("") : `<tr><td colspan="7" class="empty-row">No transactions match these filters.</td></tr>`;
    body.querySelectorAll("tr[data-tx]").forEach((row) => {
      row.addEventListener("click", () => {
        const item = JSON.parse(row.dataset.tx);
        openDialog("Transaction detail", `
          <dl class="graph-detail">
            <dt>Reference</dt><dd class="mono">${escapeHtml(item.transaction_ref)}</dd>
            <dt>Timestamp</dt><dd>${escapeHtml(item.timestamp)}</dd>
            <dt>Direction</dt><dd>${escapeHtml(item.direction)}</dd>
            <dt>From</dt><dd class="mono">${escapeHtml(item.from_account)} ${item.from_entity_name ? `· ${escapeHtml(item.from_entity_name)}` : ""}</dd>
            <dt>To</dt><dd class="mono">${escapeHtml(item.to_account)} ${item.to_entity_name ? `· ${escapeHtml(item.to_entity_name)}` : ""}</dd>
            <dt>Amount paid / received</dt><dd>${fmtMoney(item.amount_paid)} ${escapeHtml(item.payment_currency || "")} / ${fmtMoney(item.amount_received)} ${escapeHtml(item.receiving_currency || "")}</dd>
            <dt>Payment format</dt><dd>${escapeHtml(item.payment_format || "—")}</dd>
            <dt>Banks</dt><dd>${escapeHtml(item.from_bank)} → ${escapeHtml(item.to_bank)}</dd>
            <dt>Source file</dt><dd class="mono">${escapeHtml(item.source_file)}</dd>
          </dl>
          <div class="notice" style="margin-top:12px">This reference is derived from immutable transaction fields because the source data has no native transaction id.</div>
          <div style="margin-top:12px"><a class="btn btn-secondary btn-sm" href="${caseHref("graph.html", caseId, runId)}&node=${encodeURIComponent(item.counterparty)}">${icon("graph")} Open counterparty in graph</a></div>
        `);
      });
    });
  } catch (error) {
    body.innerHTML = `<tr><td colspan="7" class="empty-row">${escapeHtml(error.message)}</td></tr>`;
    document.getElementById("tx-aggregates").textContent = "Could not load transactions";
  }
}

/* ------------------------------- Notes -------------------------------- */
function renderNotesTab(target) {
  target.innerHTML = `
    <div class="split">
      <div class="card">
        <div class="card-header"><div><div class="card-title">Investigator notes & findings</div><div class="card-sub">Author, timestamp and version are stored server-side. Nothing is saved until you confirm.</div></div></div>
        <div id="notes-list" class="list"><p class="muted small">Loading notes…</p></div>
      </div>
      <div class="card">
        <div class="card-header"><div><div class="card-title" id="note-form-title">Add a note</div><div class="card-sub">Notes are investigator-entered; AI suggestions are never saved automatically.</div></div></div>
        <div class="stack">
          <div class="field"><label for="note-type">Type</label><select class="select" id="note-type"><option value="NOTE">Note</option><option value="FINDING">Confirmed finding</option><option value="HYPOTHESIS">Hypothesis</option></select></div>
          <div class="field"><label for="note-body">Body</label><textarea class="textarea" id="note-body" maxlength="20000" placeholder="Record what you verified, what you infer, and what remains unknown."></textarea></div>
          <div class="flex"><button class="btn" id="save-note">${icon("save")} Save note</button><button class="btn btn-ghost" id="cancel-note" hidden>Cancel edit</button></div>
          <p class="tiny muted">Do not record an AI hypothesis as a confirmed finding. Mark hypotheses and findings explicitly.</p>
        </div>
      </div>
    </div>
  `;
  document.getElementById("save-note").addEventListener("click", saveNote);
  document.getElementById("cancel-note").addEventListener("click", resetNoteForm);
  loadNotes();
}

function resetNoteForm() {
  editingNoteId = null;
  document.getElementById("note-body").value = "";
  document.getElementById("note-type").value = "NOTE";
  document.getElementById("note-form-title").textContent = "Add a note";
  document.getElementById("cancel-note").hidden = true;
}

async function loadNotes() {
  const list = document.getElementById("notes-list");
  if (!list) return;
  try {
    notesCache = await api(`/api/cases/${encodeURIComponent(caseId)}/notes${runQuery()}`);
    list.innerHTML = notesCache.length ? notesCache.map((note) => `
      <div class="list-item">
        <div class="between"><div class="meta">${escapeHtml(note.note_type)} · v${note.version} · ${escapeHtml(note.author)}</div><button class="btn btn-ghost btn-sm" data-edit-note="${note.id}">Edit</button></div>
        <p style="white-space:pre-wrap">${escapeHtml(note.body)}</p>
        <div class="tiny muted">${escapeHtml(fmtDateTime(note.updated_at))}</div>
      </div>`).join("") : `<p class="muted small">No notes have been recorded for this case.</p>`;
    list.querySelectorAll("[data-edit-note]").forEach((button) => {
      button.addEventListener("click", () => editNote(Number(button.dataset.editNote)));
    });
  } catch (error) {
    list.innerHTML = `<p class="muted small">${escapeHtml(error.message)}</p>`;
  }
}

function editNote(id) {
  const note = notesCache.find((item) => item.id === id);
  if (!note) return;
  editingNoteId = id;
  document.getElementById("note-body").value = note.body;
  document.getElementById("note-type").value = note.note_type;
  document.getElementById("note-form-title").textContent = `Edit note #${id}`;
  document.getElementById("cancel-note").hidden = false;
  document.getElementById("note-body").focus();
}

async function saveNote() {
  const body = document.getElementById("note-body").value.trim();
  const noteType = document.getElementById("note-type").value;
  if (!body) { toast("Note body is required.", "error"); return; }
  if (!window.confirm(editingNoteId ? "Save changes to this note?" : "Save this investigator note?")) return;
  const button = document.getElementById("save-note");
  setBusy(button, true, "Saving…");
  try {
    if (editingNoteId) {
      await api(`/api/notes/${editingNoteId}`, { method: "PATCH", body: JSON.stringify({ body, note_type: noteType, confirm: true }) });
    } else {
      await api(`/api/cases/${encodeURIComponent(caseId)}/notes${runQuery()}`, { method: "POST", body: JSON.stringify({ body, note_type: noteType, confirm: true }) });
    }
    toast("Note saved.", "ok");
    resetNoteForm();
    await loadNotes();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

/* ------------------------------ Timeline ----------------------------- */
function renderTimelineTab(target) {
  target.innerHTML = `<div class="card"><div class="card-header"><div><div class="card-title">Case activity</div><div class="card-sub">Server-side audit events, newest first.</div></div><button class="btn btn-secondary btn-sm" id="load-timeline">${icon("refresh")} Refresh</button></div><div id="timeline-list"><p class="muted small">Loading timeline…</p></div></div>`;
  document.getElementById("load-timeline").addEventListener("click", loadTimeline);
  loadTimeline();
}

async function loadTimeline() {
  const list = document.getElementById("timeline-list");
  if (!list) return;
  try {
    const events = await api(`/api/cases/${encodeURIComponent(caseId)}/timeline${runQuery()}`);
    list.innerHTML = events.length ? `<div class="timeline">${events.map((event) => `
      <div class="timeline-item">
        <div class="when">${escapeHtml(fmtDateTime(event.created_at))}</div>
        <div class="what">${escapeHtml(event.action.replaceAll("_", " "))}</div>
        <div class="who">${escapeHtml(event.actor)} · ${escapeHtml(event.outcome)}</div>
        ${event.details && Object.keys(event.details).length ? `<div class="tiny muted mono" style="margin-top:3px">${escapeHtml(JSON.stringify(event.details))}</div>` : ""}
      </div>`).join("")}</div>` : `<p class="muted small">No audit events yet.</p>`;
  } catch (error) {
    list.innerHTML = `<p class="muted small">${escapeHtml(error.message)}</p>`;
  }
}

/* ------------------------------- Status ------------------------------ */
async function updateStatus() {
  const status = document.getElementById("case-status").value;
  if (!window.confirm(`Change this case status to ${status.replace("_", " ")}? This records an audit event.`)) return;
  const button = document.getElementById("save-status");
  setBusy(button, true, "Updating…");
  try {
    await api(`/api/cases/${encodeURIComponent(caseId)}/status${runQuery()}`, { method: "PATCH", body: JSON.stringify({ status, confirm: true }) });
    toast("Case status updated.", "ok");
    await loadCase();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}
