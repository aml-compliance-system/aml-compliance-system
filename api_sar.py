"""SAR draft, review, approval and export API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from sar_service import SARService
from schemas import SarApprove, SarConfirm, SarDraftCreate, SarDraftRead, SarDraftUpdate
from security import Actor, get_actor, safe_filename
from workflow_store import WorkflowStore, WorkflowValidation

router = APIRouter(prefix="/api", tags=["sar"])


def _sar(request: Request) -> SARService:
    return request.app.state.sar


def _workflow(request: Request) -> WorkflowStore:
    return request.app.state.workflow


@router.get("/cases/{case_id}/sar")
def list_sar(
    case_id: str,
    request: Request,
    run_id: Optional[str] = None,
    actor: Actor = Depends(get_actor),
):
    return _sar(request).list_drafts(case_id, run_id)


@router.post("/cases/{case_id}/sar/drafts", response_model=SarDraftRead, status_code=201)
def create_sar(
    case_id: str,
    payload: SarDraftCreate,
    request: Request,
    run_id: Optional[str] = None,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    try:
        return _sar(request).build_draft(
            case_id,
            actor.id,
            run_id=run_id,
            include_legacy_report=payload.include_legacy_report,
        )
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/sar/drafts/{draft_id}", response_model=SarDraftRead)
def get_sar(draft_id: int, request: Request, actor: Actor = Depends(get_actor)):
    return _sar(request).get_draft(draft_id)


@router.patch("/sar/drafts/{draft_id}", response_model=SarDraftRead)
def update_sar(
    draft_id: int,
    payload: SarDraftUpdate,
    request: Request,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    try:
        return _sar(request).update_draft(draft_id, actor.id, payload.content)
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sar/drafts/{draft_id}/submit-review", response_model=SarDraftRead)
def submit_review(
    draft_id: int,
    payload: SarConfirm,
    request: Request,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    try:
        return _sar(request).submit_for_review(draft_id, actor.id)
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sar/drafts/{draft_id}/approve", response_model=SarDraftRead)
def approve_sar(
    draft_id: int,
    payload: SarApprove,
    request: Request,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    try:
        return _sar(request).approve(
            draft_id, actor.id, acknowledge_missing=payload.acknowledge_missing
        )
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sar/drafts/{draft_id}/export")
def export_sar(
    draft_id: int,
    request: Request,
    format: str = Query(default="pdf", pattern="^(pdf|docx)$"),
    actor: Actor = Depends(get_actor),
):
    draft = request.app.state.sar.get_draft(draft_id)
    payload, media_type, filename = _sar(request).export(draft_id, format)
    _workflow(request).add_event(
        draft["run_id"],
        draft["case_id"],
        actor.id,
        "SAR_EXPORTED",
        details={"format": format, "draft_id": draft_id},
    )
    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{safe_filename(filename, "sar-draft." + format)}"'
            ),
            "Cache-Control": "no-store",
        },
    )
