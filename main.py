# run fastapi:  uvicorn main:app --host 127.0.0.1 --port 8000
import mimetypes
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import config

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from LangGraph import app as langgraph_app

# Reuse the exact same input preparation and output saving as the terminal script,
# so web runs and command-line runs give identical results and files.
from run_full_pipeline import build_daily_parquet, save_outputs, OUTPUT_ROOT

from agent import AMLInvestigatorAgent
from tools import (
    TOOL_SPECS,
    AMLDataStore,
    set_default_store,
    store_from_latest_run,
    tool_names,
)
from api_cases import router as cases_router
from api_graph import router as graph_router
from api_notes import router as notes_router
from api_sar import router as sar_router
from case_service import CaseNotFound, CaseService, DataUnavailable, InvalidCase
from graph_service import GraphService
from sar_service import SARService
from security import resolve_input_path
from workflow_store import WorkflowNotFound, WorkflowStore, WorkflowValidation

log = config.get_logger("main")

REPO_ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app):
    # AGENT is created below; it exists by the time the server starts.
    if config.AGENT_ENABLED and config.AGENT_WARMUP:
        threading.Thread(target=AGENT.warmup, name="agent-warmup", daemon=True).start()
    yield


app = FastAPI(title="FINDETECT AML Investigator", version="1.0.0", lifespan=lifespan)

# Mutable workflow state and read-only evidence services. The Parquet outputs
# remain the source of truth for scores and transactions.
WORKFLOW = WorkflowStore(config.WORKFLOW_DB)
CASES = CaseService(WORKFLOW)
GRAPHS = GraphService(CASES)
SAR = SARService(CASES, WORKFLOW)
app.state.workflow = WORKFLOW
app.state.cases = CASES
app.state.graphs = GRAPHS
app.state.sar = SAR
app.state.agent_cache = {}

# One tool-using investigator shared by the API process. The store is bound to
# the latest saved run at startup and re-bound after each /pipeline/analyze call,
# so /chat always reasons over the run currently on screen.
DATA_STORE: AMLDataStore = store_from_latest_run()
set_default_store(DATA_STORE)
AGENT = AMLInvestigatorAgent(store=DATA_STORE)

# CORS: default to the local dashboard origins; configure explicitly for a
# networked deployment. Credentials stay disabled for the bearer-token model.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases_router)
app.include_router(graph_router)
app.include_router(notes_router)
app.include_router(sar_router)

# Serve the investigator UI from the API process as well. This avoids the
# Windows registry MIME issue and the need for a second server:
#   http://127.0.0.1:8000/ui/
FRONTEND_DIR = REPO_ROOT / "frontend"
if FRONTEND_DIR.is_dir():
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root_redirect():
    if (FRONTEND_DIR / "index.html").is_file():
        return RedirectResponse(url="/ui/")
    return JSONResponse(
        content={
            "service": "FINDETECT AML Investigator",
            "docs": "/docs",
            "ui": "frontend/index.html not found",
        }
    )


@app.exception_handler(CaseNotFound)
async def _case_not_found(request: Request, exc: CaseNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidCase)
async def _invalid_case(request: Request, exc: InvalidCase):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DataUnavailable)
async def _data_unavailable(request: Request, exc: DataUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(WorkflowNotFound)
async def _workflow_not_found(request: Request, exc: WorkflowNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(WorkflowValidation)
async def _workflow_validation(request: Request, exc: WorkflowValidation):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

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
    """Backward-compatible wrapper around the sandboxed resolver."""
    return resolve_input_path(raw_path)


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
        run_label = (
            f"{request.date or input_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        try:
            save_outputs(run_label, final_state, input_parquet=input_path)
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

        # Bind this run's evidence to the agent. Prefer the saved parquet (keeps
        # memory flat); fall back to the in-memory DataFrame if saving failed.
        global DATA_STORE
        risk_path = OUTPUT_ROOT / run_label / "risk_scores.parquet"
        reports_by_account = {
            alert["Account"]: report
            for alert, report in zip(alerts, reports)
            if alert.get("Account")
        }
        try:
            if risk_path.is_file():
                DATA_STORE.register_pipeline_run(
                    risk_scores_path=risk_path,
                    transactions_path=daily_path,
                    reports=reports_by_account,
                    run_label=run_label,
                    input_parquet=input_path,
                )
            else:
                DATA_STORE.register_pipeline_run(
                    final_state=final_state,
                    transactions_path=daily_path,
                    reports=reports_by_account,
                    run_label=run_label,
                    input_parquet=input_path,
                )
            AGENT.store = DATA_STORE
            log.info(
                "Agent bound to run %s (saved risk scores=%s)",
                run_label,
                risk_path.is_file(),
            )
        except Exception:
            log.exception("Could not bind pipeline run to the agent")

        return {
            "status": "success",
            "high_risk_alerts_found": len(alerts),
            "case_data": alerts,
            "final_insights": alerts,
            "sar_reports": reports,
            "reports": reports,
            "output_folder": output_folder,
            "agent_ready": DATA_STORE.has_risk_data,
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
    # Optional prior turns, e.g. [{"role": "user", "content": "..."}]. Bounded
    # server-side; useful for follow-up questions in the chat UI.
    history: Optional[List[Dict[str, Any]]] = None
    # The account/day selected in the dashboard (YYYY-MM-DD). Falls back to the
    # alert currently cached for the account.
    case_date: Optional[str] = None


class AgentReviewRequest(BaseModel):
    case_id: str
    focus: Optional[str] = None


class ToolRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


_CHAT_SYSTEM_PROMPT = (
    "You are an expert Anti-Money Laundering (AML) Compliance Investigator. "
    "Be direct, concise, and professional. Base answers strictly on the provided case telemetry."
)

_CHAT_ESSENTIAL_KEYS = [
    "Account", "Account_duplicated_0", "Amount Paid", "Amount Received",
    "final_risk_score", "risk_label", "base_risk", "graph_penalty",
    "is_high_risk_seed", "received_from_high_risk",
]

MAX_MESSAGE_CHARS = 4_000


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


def _case_date(account_id: str) -> Optional[str]:
    """The account/day date currently selected in the dashboard, if any."""
    case_info = CASE_CACHE.get(account_id) or {}
    alert = case_info.get("alert") or {}
    graph_date = alert.get("graph_date")
    if graph_date:
        text = str(graph_date).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", text):
            return text[:10]
    return None


def _legacy_chat(request: ChatRequest) -> str:
    """Original single-shot chat, kept for FINDETECT_AGENT_ENABLED=0."""
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
    return response["message"]["content"]


@app.post("/chat")
def chat_with_case(request: ChatRequest):
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"message is too long (max {MAX_MESSAGE_CHARS} characters).",
        )

    try:
        if config.AGENT_ENABLED:
            if not DATA_STORE.has_risk_data:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "No pipeline run is bound yet. Run the pipeline from the "
                        "dashboard (or save a run under outputs/) before chatting."
                    ),
                )
            result = AGENT.chat(
                request.case_id,
                message,
                history=request.history,
                case_date=request.case_date or _case_date(request.case_id),
            )
            payload = result.to_dict()
            payload["message"] = "success"
            payload["case_id"] = request.case_id
            return payload

        answer = _legacy_chat(request)
        return {
            "message": "success",
            "case_id": request.case_id,
            "response": answer,
            "answer": answer,
            "degraded": False,
            "tool_calls": [],
        }

    except HTTPException:
        raise
    except Exception:
        log.exception("Chat endpoint failed for case_id=%s", request.case_id)
        raise HTTPException(status_code=500, detail="Chat request failed. See server logs.")


@app.get("/agent/tools")
def list_agent_tools():
    """Describe the read-only evidence tools exposed to the LLM."""
    return {
        "enabled": bool(config.AGENT_ENABLED),
        "model": config.OLLAMA_TOOL_MODEL,
        "tools": TOOL_SPECS,
        "run_label": DATA_STORE.run_label,
        "agent_ready": DATA_STORE.has_risk_data,
    }


@app.post("/agent/tool")
def call_agent_tool(request: ToolRequest):
    """Execute one evidence tool directly (useful for debugging and demos)."""
    if request.name not in tool_names():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool. Available: {', '.join(tool_names())}",
        )
    # `store` is bound by the server; never let a caller override it.
    arguments = {k: v for k, v in (request.arguments or {}).items() if k != "store"}
    return AGENT.run_tool(request.name, arguments)


@app.post("/agent/review")
def review_case(request: AgentReviewRequest):
    """Run a standard tool-using investigation memo for one case."""
    if not config.AGENT_ENABLED:
        raise HTTPException(status_code=409, detail="The agent is disabled.")
    if not DATA_STORE.has_risk_data:
        raise HTTPException(
            status_code=409,
            detail="No pipeline run is bound. Run the pipeline first.",
        )
    result = AGENT.review(
        request.case_id,
        focus=request.focus,
        case_date=_case_date(request.case_id),
    )
    payload = result.to_dict()
    payload["message"] = "success"
    return payload


@app.get("/case/{case_id}")
def get_case_details(case_id: str):
    case = CASE_CACHE.get(case_id)
    if case is not None:
        return case

    # The new workspace uses account/day case ids; support them here too so the
    # legacy route does not become a dead end.
    try:
        return CASES.get_case(case_id)
    except InvalidCase:
        pass
    except CaseNotFound:
        pass

    raise HTTPException(
        status_code=404,
        detail="Case telemetry not found. Select an account/day case from the case queue.",
    )

# 