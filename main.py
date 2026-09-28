# run fastapi:  uvicorn main:app --host 127.0.0.1 --port 8000
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
import config

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from LangGraph import app as langgraph_app

# Reuse the exact same input preparation and output saving as the terminal script,
# so web runs and command-line runs give identical results and files.
from run_full_pipeline import build_daily_parquet, save_outputs, OUTPUT_ROOT

log = config.get_logger("main")

REPO_ROOT = Path(__file__).resolve().parent

app = FastAPI()

# CORS: any origin may call this local API. Credentials must stay disabled,
# because allow_origins=["*"] combined with allow_credentials=True is invalid
# per the CORS spec and browsers reject it. For a real deployment, replace "*"
# with an explicit allowlist of your frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache is per-process and in-memory by design (this API has no external store
# configured). A lock keeps concurrent requests from tearing a read mid-clear;
# it does NOT make results consistent across multiple worker processes — run
# this behind a single worker, or swap CASE_CACHE for a real store (e.g.
# SQLite/Redis) before scaling out.
_CACHE_LOCK = threading.Lock()
CASE_CACHE: dict[str, dict] = {}


class PipelineRequest(BaseModel):
    # Path to the parquet file to analyze (relative paths are resolved
    # from the project folder, not from where the server was started).
    final_parquet_path: str = "Data/processed/prod_sample_oct23_25.parquet"
    # Optional: analyze only this day (YYYY-MM-DD).
    date: Optional[str] = None


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@app.post("/pipeline/analyze")
def analyze(request: PipelineRequest):
    try:
        if request.date:
            try:
                datetime.strptime(request.date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"date must be YYYY-MM-DD, got: {request.date}",
                )

        input_path = _resolve_input_path(request.final_parquet_path)

        # Same preparation step as run_full_pipeline.py (adds derived features).
        daily_path = build_daily_parquet(request.date, str(input_path))

        final_state = langgraph_app.invoke({"final_parquet_path": str(daily_path)})

        alerts = final_state.get("final_insights", [])
        reports = final_state.get("sar_reports", [])

        # Save the same output files the terminal script produces.
        output_folder = None
        try:
            run_label = (
                f"{request.date or input_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
            )
            save_outputs(run_label, final_state)
            output_folder = str(OUTPUT_ROOT / run_label)
        except Exception:
            log.exception("Could not save output files (results still returned)")

        new_cache = {
            alert["Account"]: {"alert": alert, "report": report}
            for alert, report in zip(alerts, reports)
            if alert.get("Account")
        }

        with _CACHE_LOCK:
            CASE_CACHE.clear()
            CASE_CACHE.update(new_cache)

        return {
            "status": "success",
            "high_risk_alerts_found": len(alerts),
            "case_data": alerts,
            "final_insights": alerts,
            "sar_reports": reports,
            "reports": reports,
            "output_folder": output_folder,
        }

    except HTTPException:
        raise
    except FileNotFoundError as e:
        log.exception("Input file not found: %s", request.final_parquet_path)
        raise HTTPException(status_code=404, detail=str(e).strip())
    except ValueError as e:
        log.exception("Invalid input for pipeline: %s", request.final_parquet_path)
        raise HTTPException(status_code=400, detail=str(e).strip())
    except Exception:
        log.exception(
            "Pipeline analysis failed for final_parquet_path=%s",
            request.final_parquet_path,
        )
        raise HTTPException(
            status_code=500,
            detail="Pipeline analysis failed. See server logs.",
        )


class ChatRequest(BaseModel):
    case_id: str
    message: str


_CHAT_SYSTEM_PROMPT = (
    "You are an expert Anti-Money Laundering (AML) Compliance Investigator. "
    "Be direct, concise, and professional. Base answers strictly on the provided case telemetry."
)

_CHAT_ESSENTIAL_KEYS = [
    "Account", "Account_duplicated_0", "Amount Paid", "Amount Received",
    "final_risk_score", "risk_label", "base_risk", "graph_penalty",
    "is_high_risk_seed", "received_from_high_risk",
]


def _build_chat_context(account_id: str) -> str:
    case_info = CASE_CACHE.get(account_id)
    if not case_info:
        return f"Target Account ID: {account_id} (No telemetry cache found.)"

    raw_alert = case_info.get("alert", {})
    slim_alert = {k: raw_alert[k] for k in _CHAT_ESSENTIAL_KEYS if k in raw_alert}
    return (
        f"=== CASE METRICS ===\n{slim_alert}\n\n"
        f"=== INITIAL SAR REPORT ===\n{case_info.get('report', 'No report available.')}"
    )


@app.post("/chat")
def chat_with_case(request: ChatRequest):
    try:
        context_text = _build_chat_context(request.case_id)
        user_content = f"[CASE CONTEXT]\n{context_text}\n\n[INVESTIGATOR QUERY]\n{request.message}"

        response = ollama.chat(
            model=config.OLLAMA_CHAT_MODEL,
            messages=[
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            options={
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_predict": config.OLLAMA_NUM_PREDICT,
            },
        )

        return {
            "message": "success",
            "case_id": request.case_id,
            "response": response["message"]["content"],
        }

    except Exception:
        log.exception("Chat endpoint failed for case_id=%s", request.case_id)
        raise HTTPException(status_code=500, detail="Chat request failed. See server logs.")


@app.get("/case/{case_id}")
def get_case_details(case_id: str):
    case = CASE_CACHE.get(case_id)
    if case is not None:
        return case

    raise HTTPException(
        status_code=404,
        detail="Case telemetry not found. Please re-run the pipeline from the dashboard.",
    )

# 