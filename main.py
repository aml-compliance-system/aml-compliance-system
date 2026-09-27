# run fastapi
import threading
import ollama
import config

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from LangGraph import app as langgraph_app

log = config.get_logger("main")
 
app = FastAPI()
 
# CORS: no origins are allowed by default. Set FINDETECT_CORS_ORIGINS to a
# comma-separated allowlist (e.g. "https://app.example.com") for your deployment.
# allow_origins=["*"] + allow_credentials=True is invalid per the CORS spec and
# most browsers reject it outright, so we never combine them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all local frontends/origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
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
    csv_path: str = "input/unlabeled_transactions.csv"
 
 
@app.post("/pipeline/analyze")
def analyze(request: PipelineRequest):
    try:
        initial_state = {"raw_csv_path": request.csv_path}
        final_state = langgraph_app.invoke(initial_state)
 
        alerts = final_state.get("final_insights", [])
        reports = final_state.get("sar_reports", [])
 
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
            "reports": reports,
        }
 
    except Exception:
        log.exception("Pipeline analysis failed for csv_path=%s", request.csv_path)
        raise HTTPException(status_code=500, detail="Pipeline analysis failed. See server logs.")
 
 
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