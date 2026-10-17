"""Investigator notes API routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from schemas import NoteCreate, NoteRead, NoteUpdate
from security import Actor, get_actor
from workflow_store import WorkflowNotFound, WorkflowStore, WorkflowValidation

router = APIRouter(prefix="/api", tags=["notes"])


def _workflow(request: Request) -> WorkflowStore:
    return request.app.state.workflow


@router.get("/cases/{case_id}/notes", response_model=List[NoteRead])
def list_notes(
    case_id: str,
    request: Request,
    run_id: Optional[str] = None,
    actor: Actor = Depends(get_actor),
):
    cases = request.app.state.cases
    case = cases.get_case(case_id, run_id)
    return _workflow(request).list_notes(case["run_id"], case_id)


@router.post("/cases/{case_id}/notes", response_model=NoteRead, status_code=201)
def create_note(
    case_id: str,
    payload: NoteCreate,
    request: Request,
    run_id: Optional[str] = None,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    case = request.app.state.cases.get_case(case_id, run_id)
    try:
        return _workflow(request).create_note(
            case["run_id"], case_id, actor.id, payload.body, payload.note_type
        )
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.patch("/notes/{note_id}", response_model=NoteRead)
def update_note(
    note_id: int,
    payload: NoteUpdate,
    request: Request,
    actor: Actor = Depends(get_actor),
):
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    try:
        return _workflow(request).update_note(
            note_id, actor.id, payload.body, payload.note_type
        )
    except WorkflowNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except WorkflowValidation as exc:
        raise HTTPException(status_code=400, detail=str(exc))
