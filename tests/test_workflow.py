"""Tests for SQLite workflow state, notes, SAR versions and audit."""

from __future__ import annotations

import sqlite3

import pytest

from workflow_store import WorkflowNotFound, WorkflowStore, WorkflowValidation


def test_case_status_and_audit(workflow):
    state = workflow.get_case_state("run", "CASE-A-20220101")
    assert state["status"] == "NEW"
    updated = workflow.set_case_status("run", "CASE-A-20220101", "IN_REVIEW", "alice")
    assert updated["status"] == "IN_REVIEW"
    events = workflow.list_audit("run", "CASE-A-20220101")
    assert events[0]["action"] == "CASE_STATUS_CHANGED"
    assert events[0]["details"]["to"] == "IN_REVIEW"
    with pytest.raises(WorkflowValidation):
        workflow.set_case_status("run", "CASE-A-20220101", "BOGUS", "alice")


def test_notes_are_versioned_and_audited(workflow):
    note = workflow.create_note("run", "CASE-A-20220101", "alice", "First observation", "NOTE")
    assert note["version"] == 1
    assert note["author"] == "alice"
    updated = workflow.update_note(note["id"], "bob", "Confirmed finding", "FINDING")
    assert updated["version"] == 2
    assert updated["note_type"] == "FINDING"
    notes = workflow.list_notes("run", "CASE-A-20220101")
    assert len(notes) == 1
    conn = sqlite3.connect(workflow.path)
    try:
        versions = conn.execute("SELECT version, author FROM note_versions ORDER BY version").fetchall()
    finally:
        conn.close()
    assert versions == [(1, "alice"), (2, "bob")]
    actions = [event["action"] for event in workflow.list_audit("run", "CASE-A-20220101")]
    assert actions[:2] == ["NOTE_UPDATED", "NOTE_CREATED"]


def test_note_validation(workflow):
    with pytest.raises(WorkflowValidation):
        workflow.create_note("run", "case", "alice", "   ")
    with pytest.raises(WorkflowValidation):
        workflow.create_note("run", "case", "alice", "body", "BOGUS")
    with pytest.raises(WorkflowNotFound):
        workflow.update_note(999, "alice", "body")


def test_sar_draft_lifecycle_and_approval_gate(workflow):
    draft = workflow.create_sar_draft(
        "run", "CASE-A-20220101", {"case": {"case_id": "CASE-A-20220101"}},
        ["subject_date_of_birth"], [{"transaction_ref": "TX-1"}], "alice",
    )
    assert draft["version"] == 1
    assert draft["status"] == "DRAFT"
    with pytest.raises(WorkflowValidation):
        workflow.approve_sar_draft(draft["id"], "bob", acknowledge_missing=False)
    reviewed = workflow.submit_sar_for_review(draft["id"], "alice")
    assert reviewed["status"] == "IN_REVIEW"
    approved = workflow.approve_sar_draft(draft["id"], "bob", acknowledge_missing=True)
    assert approved["status"] == "APPROVED"
    assert approved["approved_by"] == "bob"
    updated = workflow.update_sar_draft(
        draft["id"], "bob", {"case": {"case_id": "CASE-A-20220101"}, "narrative": {"summary": "edited"}},
        [], [],
    )
    assert updated["version"] == 2
    assert updated["status"] == "DRAFT"
    assert updated["approved_by"] is None


def test_audit_events_are_append_only(workflow):
    workflow.add_event("run", "case", "alice", "CASE_VIEWED")
    conn = sqlite3.connect(workflow.path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE audit_events SET actor = 'mallory'")
            conn.commit()
        conn.rollback()
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM audit_events")
            conn.commit()
    finally:
        conn.close()
