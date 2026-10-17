"""Bounded graph exploration and analysis API routes."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Request

import config
from schemas import (
    ComponentRequest,
    CycleRequest,
    ExpandRequest,
    GraphAnalysisResponse,
    GraphResponse,
    PathRequest,
    SharedCounterpartyRequest,
)
from security import Actor, get_actor
from workflow_store import WorkflowStore

router = APIRouter(prefix="/api", tags=["graph"])


def _graphs(request: Request):
    return request.app.state.graphs


def _workflow(request: Request) -> WorkflowStore:
    return request.app.state.workflow


@router.get("/cases/{case_id}/graph", response_model=GraphResponse)
def case_graph(
    case_id: str,
    request: Request,
    run_id: Optional[str] = None,
    depth: int = Query(default=1, ge=1, le=config.GRAPH_MAX_DEPTH),
    limit_nodes: int = Query(default=config.GRAPH_MAX_NODES, ge=1, le=config.GRAPH_MAX_NODES),
    limit_edges: int = Query(default=config.GRAPH_MAX_EDGES, ge=1, le=config.GRAPH_MAX_EDGES),
    include_holders: bool = False,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    actor: Actor = Depends(get_actor),
):
    result = _graphs(request).case_graph(
        case_id,
        run_id=run_id,
        depth=depth,
        limit_nodes=limit_nodes,
        limit_edges=limit_edges,
        from_value=from_value,
        to_value=to_value,
        include_holders=include_holders,
    )
    _workflow(request).add_event(
        result["run_id"], case_id, actor.id, "GRAPH_EXPLORED",
        details={
            "nodes": len(result["nodes"]),
            "edges": len(result["edges"]),
            "truncated": result["truncated"],
            "depth": depth,
        },
    )
    return result


@router.post("/cases/{case_id}/graph/paths", response_model=GraphAnalysisResponse)
def graph_paths(
    case_id: str,
    payload: PathRequest,
    request: Request,
    run_id: Optional[str] = None,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    actor: Actor = Depends(get_actor),
):
    result = _graphs(request).paths(
        case_id,
        run_id=run_id,
        source=payload.source,
        target=payload.target,
        max_depth=payload.max_depth,
        from_value=from_value,
        to_value=to_value,
    )
    _workflow(request).add_event(
        result["run_id"], case_id, actor.id, "GRAPH_ANALYSIS_RUN",
        details={"algorithm": "paths", "findings": len(result["findings"])},
    )
    return result


@router.post("/cases/{case_id}/graph/cycles", response_model=GraphAnalysisResponse)
def graph_cycles(
    case_id: str,
    payload: CycleRequest,
    request: Request,
    run_id: Optional[str] = None,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    actor: Actor = Depends(get_actor),
):
    result = _graphs(request).cycles(
        case_id,
        run_id=run_id,
        max_length=payload.max_length,
        from_value=from_value,
        to_value=to_value,
    )
    _workflow(request).add_event(
        result["run_id"], case_id, actor.id, "GRAPH_ANALYSIS_RUN",
        details={"algorithm": "cycles", "findings": len(result["findings"])},
    )
    return result


@router.post(
    "/cases/{case_id}/graph/shared-counterparties",
    response_model=GraphAnalysisResponse,
)
def graph_shared(
    case_id: str,
    payload: SharedCounterpartyRequest,
    request: Request,
    run_id: Optional[str] = None,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    actor: Actor = Depends(get_actor),
):
    result = _graphs(request).shared(
        case_id,
        run_id=run_id,
        accounts=payload.accounts,
        min_shared=payload.min_shared,
        from_value=from_value,
        to_value=to_value,
    )
    _workflow(request).add_event(
        result["run_id"], case_id, actor.id, "GRAPH_ANALYSIS_RUN",
        details={"algorithm": "shared-counterparties", "findings": len(result["findings"])},
    )
    return result


@router.post("/cases/{case_id}/graph/components", response_model=GraphAnalysisResponse)
def graph_components(
    case_id: str,
    payload: ComponentRequest,
    request: Request,
    run_id: Optional[str] = None,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    actor: Actor = Depends(get_actor),
):
    result = _graphs(request).components(
        case_id,
        run_id=run_id,
        account=payload.account,
        from_value=from_value,
        to_value=to_value,
    )
    _workflow(request).add_event(
        result["run_id"], case_id, actor.id, "GRAPH_ANALYSIS_RUN",
        details={"algorithm": "components", "findings": len(result["findings"])},
    )
    return result


@router.post("/cases/{case_id}/graph/expand", response_model=GraphResponse)
def graph_expand(
    case_id: str,
    payload: ExpandRequest,
    request: Request,
    run_id: Optional[str] = None,
    from_value: Optional[str] = Query(default=None, alias="from"),
    to_value: Optional[str] = Query(default=None, alias="to"),
    actor: Actor = Depends(get_actor),
):
    result = _graphs(request).expand(
        case_id,
        payload.account,
        run_id=run_id,
        direction=payload.direction,
        from_value=from_value,
        to_value=to_value,
    )
    _workflow(request).add_event(
        result["run_id"], case_id, actor.id, "GRAPH_EXPLORED",
        details={
            "expanded_account": payload.account,
            "nodes": len(result["nodes"]),
            "edges": len(result["edges"]),
            "truncated": result["truncated"],
        },
    )
    return result
