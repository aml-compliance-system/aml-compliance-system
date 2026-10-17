"""Case, transaction, chat and meta API routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

import config
from agent import AMLInvestigatorAgent
from case_service import CaseNotFound, CaseService, DataUnavailable, InvalidCase
from schemas import (
    CaseChatRequest,
    CaseChatResponse,
    CaseDetail,
    CaseListResponse,
    CaseStatusUpdate,
    MetaResponse,
    RunInfo,
    TimelineEvent,
    TransactionPage,
)
from security import Actor, auth_mode, get_actor
from tools import AMLDataStore
from workflow_store import CASE_STATUSES, WorkflowStore, WorkflowValidation

router = APIRouter(prefix="/api", tags=["cases"])


def _cases(request: Request) -> CaseService:
    return request.app.state.cases


def _workflow(request: Request) -> WorkflowStore:
    return request.app.state.workflow


def _agent(request: Request, run) -> AMLInvestigatorAgent:
    cache: Dict[Any, AMLInvestigatorAgent] = request.app.state.agent_cache
    key = (run.run_id, run.risk_path.stat().st_mtime_ns)
    agent = cache.get(key)
    if agent is None:
        store = AMLDataStore(
            risk_scores_path=run.risk_path,
            transactions_path=run.transactions_path,
            run_label=run.run_id,
            input_parquet=run.transactions_path,
        )
        agent = AMLInvestigatorAgent(store=store)
        cache[key] = agent
        if len(cache) > 4:
            cache.pop(next(iter(cache)))
    return agent


@router.get("/runs", response_model=List[RunInfo])
def list_runs(request: Request, actor: Actor = Depends(get_actor)):
    runs = []
    for run in _cases(request).list_runs():
        runs.append(
            {
                "run_id": run.run_id,
                "risk_path": str(run.risk_path),
                "transactions_path": str(run.transactions_path) if run.transactions_path else None,
                "summary": run.summary,
                "created_at": run.created_at,
                "has_legacy_reports": bool(run.legacy_reports),
            }
        )
    return runs


@router.get("/cases", response_model=CaseListResponse)
def list_cases(
    request: Request,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
    tier: Optional[str] = None,
    risk_label: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    max_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    account: Optional[str] = Query(default=None, max_length=64),
    q: Optional[str] = Query(default=None, max_length=128),
    sort: str = Query(default="score_desc", pattern="^(score_desc|score_asc|date_desc|date_asc|account)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(
        default=config.CASE_PAGE_SIZE_DEFAULT,
        ge=1,
        le=config.CASE_PAGE_SIZE_MAX,
    ),
    actor: Actor = Depends(get_actor),
):
    if status and status.upper() not in CASE_STATUSES:
        raise HTTPException(status_code=400, detail=f"Unknown case status: {status}")
    try:
        return _cases(request).list_cases(
            run_id=run_id,
            status=status,
            tier=tier,
            risk_label=risk_label,
            date_from=date_from,
            date_to=date_to,
            min_score=min_score,
            max_score=max_score,
            account=account,
            q=q,
            sort=sort,
            page=page,
            page_size=page_size,
        )
    except (InvalidCase, WorkflowValidation) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(
    case_id: str,
    request: Request,
    run_id: Optional[str] = None,
    track: bool = True,
    actor: Actor = Depends(get_actor),
):
    try:
        case = _cases(request).get_case(case_id, run_id)
        if track:
            _workflow(request).add_event(
                case["run_id"], case_id, actor.id, "CASE_VIEWED",
                details={"source": "case_detail"},
            )
        return case
    except InvalidCase as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/cases/{case_id}/transactions", response_model=TransactionPage)
def list_transactions(
    case_id: str,
    request: Request,
    run_id: Optional[str] = None,
    direction: str = Query(default="both", pattern="^(both|in|out)$"),
    counterparty: Optional[str] = Query(default=None, max_length=64),
    min_amount: Optional[float] = Query(default=None, ge=0),
    max_amount: Optional[float] = Query(default=None, ge=0),
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=config.CASE_PAGE_SIZE_MAX),
    actor: Actor = Depends(get_actor),
):
    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        raise HTTPException(status_code=400, detail="min_amount must be <= max_amount")
    try:
        return _cases(request).list_transactions(
            case_id,
            run_id=run_id,
            direction=direction,
            counterparty=counterparty,
            min_amount=min_amount,
            max_amount=max_amount,
            from_value=from_value,
            to_value=to_value,
            page=page,
            page_size=page_size,
        )
    except InvalidCase as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.patch("/cases/{case_id}/status")
def update_case_status(
    case_id: str,
    payload: CaseStatusUpdate,
    request: Request,
    run_id: Optional[str] = None,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    try:
        case = _cases(request).get_case(case_id, run_id)
        state = _workflow(request).set_case_status(
            case["run_id"], case_id, payload.status, actor.id, payload.assigned_to
        )
        return {"message": "Case status updated", "case_id": case_id, "state": state}
    except InvalidCase as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/cases/{case_id}/timeline", response_model=List[TimelineEvent])
def case_timeline(
    case_id: str,
    request: Request,
    run_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    actor: Actor = Depends(get_actor),
):
    try:
        case = _cases(request).get_case(case_id, run_id)
        return _workflow(request).list_audit(case["run_id"], case_id, limit=limit)
    except InvalidCase as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/cases/{case_id}/chat", response_model=CaseChatResponse)
def case_chat(
    case_id: str,
    payload: CaseChatRequest,
    request: Request,
    run_id: Optional[str] = None,
    actor: Actor = Depends(get_actor),
):
    if not config.AGENT_ENABLED:
        raise HTTPException(status_code=409, detail="The AI investigator is disabled.")
    try:
        cases = _cases(request)
        run = cases.get_run(run_id)
        case = cases.get_case(case_id, run.run_id)
        agent = _agent(request, run)
        result = agent.chat(
            case["account_id"],
            payload.message,
            history=payload.history,
            case_date=case["graph_date"],
        )
        _workflow(request).add_event(
            run.run_id,
            case_id,
            actor.id,
            "AI_QUERY",
            details={
                "tools": [call["name"] for call in result.to_dict().get("tool_calls", [])],
                "steps": result.steps,
                "degraded": result.degraded,
                "elapsed_ms": result.elapsed_ms,
            },
        )
        body = result.to_dict()
        body.update(
            {
                "message": "success",
                "case_id": case_id,
                "run_id": run.run_id,
            }
        )
        return body
    except InvalidCase as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CaseNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # The agent normally handles model failures; this is a last-resort guard.
        raise HTTPException(status_code=500, detail=f"Investigation request failed: {type(exc).__name__}")


@router.get("/meta", response_model=MetaResponse)
def meta(request: Request, actor: Actor = Depends(get_actor)):
    cases = _cases(request)
    run = None
    summary: Dict[str, Any] = {}
    try:
        run = cases.get_run()
        summary = run.summary
    except CaseNotFound:
        run = None
    policy = request.app.state.sar.policy_validation()
    return {
        "run_id": run.run_id if run else None,
        "run_summary": summary,
        "auth_mode": auth_mode(),
        "investigator": actor.id,
        "model": config.OLLAMA_TOOL_MODEL,
        "native_tools": bool(config.AGENT_NATIVE_TOOLS),
        "policy_validation": policy,
        "limits": {
            "page_size_max": config.CASE_PAGE_SIZE_MAX,
            "graph_nodes": config.GRAPH_MAX_NODES,
            "graph_edges": config.GRAPH_MAX_EDGES,
            "graph_depth": config.GRAPH_MAX_DEPTH,
            "graph_paths": config.GRAPH_MAX_PATHS,
            "graph_cycles": config.GRAPH_MAX_CYCLES,
            "sar_evidence_rows": config.SAR_MAX_EVIDENCE_ROWS,
        },
        "capabilities": {
            "cases": run is not None,
            "transactions": bool(run and run.transactions_path),
            "graph": run is not None,
            "notes": True,
            "sar": True,
            "sar_export": True,
            "policy_validation": False,
            "authentication": auth_mode() == "token",
        },
    }
