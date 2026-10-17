"""
Command-line access to the agentic AML investigator.

Examples
--------
List the tools the LLM can use::

    python agent_cli.py tools

Call one evidence tool against the latest saved run::

    python agent_cli.py call get_account_scores --args "{\"account_id\": \"100428660\", \"date\": \"2022-10-23\"}"

Ask the investigator a question (uses Ollama)::

    python agent_cli.py chat --case 100428660 --date 2022-10-23 \
        --message "Why was this account flagged?"

Produce the standard investigation memo::

    python agent_cli.py review --case 100428660 --date 2022-10-23

Run without an LLM (deterministic evidence memo, useful offline)::

    python agent_cli.py chat --case 100428660 --date 2022-10-23 \
        --message "summarise" --no-llm

The store defaults to the newest run under ``outputs/FINDETECT_Pipeline_Output``
and to the transactions parquet recorded in that run's ``summary.json``.
Override with ``--output-dir`` and ``--transactions``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

import config
from agent import AMLInvestigatorAgent
from tools import (
    TOOL_SPECS,
    dispatch_tool,
    store_from_latest_run,
    tool_names,
)


class _OfflineClient:
    """Client double that forces the agent's deterministic offline memo."""

    def chat(self, **kwargs):  # noqa: D401 - protocol method
        raise ConnectionError("offline mode requested (--no-llm)")


def _print_trace(result) -> None:
    if not result.tool_calls:
        return
    print("\n[evidence trace]", file=sys.stderr)
    for call in result.tool_calls:
        status = "ok" if call.ok else "error"
        print(
            f"  - {call.name} ({call.source}, {status}, {call.elapsed_ms} ms): "
            f"{call.summary}",
            file=sys.stderr,
        )


def _load_store(args):
    store = store_from_latest_run(
        output_dir=args.output_dir,
        transactions_path=args.transactions,
    )
    if not store.has_risk_data:
        raise SystemExit(
            "No saved pipeline run found. Run the pipeline first, or pass "
            "--output-dir pointing at a folder that contains risk_scores.parquet."
        )
    return store


def _print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_tools(args) -> int:
    if args.json:
        _print_json({"tools": TOOL_SPECS})
        return 0
    for spec in TOOL_SPECS:
        print(f"\n{spec['name']}")
        print(f"  {spec['description']}")
        params = spec["parameters"]["properties"]
        for name, meta in params.items():
            print(f"    - {name}: {meta.get('description', '')}")
    return 0


def cmd_call(args) -> int:
    try:
        arguments = json.loads(args.args) if args.args else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--args is not valid JSON: {exc}")
    store = _load_store(args)
    result = dispatch_tool(args.tool, arguments, store=store)
    _print_json(result)
    return 0 if result.get("ok") else 1


def cmd_chat(args) -> int:
    store = _load_store(args)
    if args.no_llm:
        agent = AMLInvestigatorAgent(store=store, client=_OfflineClient())
    else:
        agent = AMLInvestigatorAgent(
            store=store,
            model=args.model,
            max_steps=args.max_steps,
        )
    result = agent.chat(args.case, args.message, case_date=args.date)
    if args.json:
        _print_json(result.to_dict())
    else:
        print(result.answer)
        _print_trace(result)
        if result.degraded:
            print(
                "\n[offline] Language model unavailable; deterministic memo shown.",
                file=sys.stderr,
            )
    return 0


def cmd_review(args) -> int:
    store = _load_store(args)
    if args.no_llm:
        agent = AMLInvestigatorAgent(store=store, client=_OfflineClient())
    else:
        agent = AMLInvestigatorAgent(
            store=store,
            model=args.model,
            max_steps=args.max_steps,
        )
    result = agent.review(args.case, focus=args.focus, case_date=args.date)
    if args.json:
        _print_json(result.to_dict())
    else:
        print(result.answer)
        _print_trace(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tool-using AML investigator (evidence tools + local LLM)."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=None, help="Saved pipeline run folder.")
    common.add_argument(
        "--transactions",
        default=None,
        help="Transactions parquet; defaults to the run's recorded input.",
    )
    common.add_argument("--json", action="store_true", help="Emit JSON only.")
    common.add_argument("--no-llm", action="store_true", help="Deterministic memo, no model.")
    common.add_argument("--model", default=config.OLLAMA_TOOL_MODEL)
    common.add_argument("--max-steps", type=int, default=config.AGENT_MAX_STEPS)

    sub = parser.add_subparsers(dest="command", required=True)

    p_tools = sub.add_parser("tools", help="List the evidence tools.")
    p_tools.add_argument("--json", action="store_true")
    p_tools.set_defaults(func=cmd_tools)

    p_call = sub.add_parser("call", help="Call one evidence tool.")
    p_call.add_argument("tool", choices=tool_names())
    p_call.add_argument("--args", default="{}", help="JSON arguments object.")
    p_call.add_argument("--output-dir", default=None)
    p_call.add_argument("--transactions", default=None)
    p_call.add_argument("--json", action="store_true")
    p_call.set_defaults(func=cmd_call)

    p_chat = sub.add_parser("chat", parents=[common], help="Ask the investigator.")
    p_chat.add_argument("--case", required=True, help="Account id.")
    p_chat.add_argument("--message", required=True, help="Investigator question.")
    p_chat.add_argument("--date", default=None, help="Case date YYYY-MM-DD.")
    p_chat.set_defaults(func=cmd_chat)

    p_review = sub.add_parser("review", parents=[common], help="Investigation memo.")
    p_review.add_argument("--case", required=True)
    p_review.add_argument("--date", default=None)
    p_review.add_argument("--focus", default=None)
    p_review.set_defaults(func=cmd_review)

    return parser


def main(argv: Optional[list] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
