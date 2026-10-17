"""Tests for evidence-grounded SAR drafting, approval and exports."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO

import pytest

from sar_service import SARService
from workflow_store import WorkflowValidation


def test_draft_is_case_specific_and_evidence_grounded(synthetic_cases, workflow):
    workflow.create_note("run-test", "CASE-A1-20220101", "alice", "Source of funds verified.", "FINDING")
    service = SARService(synthetic_cases, workflow)
    draft = service.build_draft("CASE-A1-20220101", "alice")
    content = draft["content"]
    assert draft["status"] == "DRAFT"
    assert draft["version"] == 1
    assert content["case"]["case_id"] == "CASE-A1-20220101"
    assert content["case"]["account_id"] == "A1"
    assert content["case"]["holder_name"] == "Alpha LLC"
    assert content["risk"]["final_risk_score"] == pytest.approx(0.90)
    assert content["activity"]["transaction_count"] == 4
    assert len(content["evidence"]) == 4
    assert content["evidence_refs"]
    assert content["investigator_findings"][0]["body"] == "Source of funds verified."
    assert content["policy_validation"]["available"] is False
    assert content["legacy_pipeline_report"]
    missing = draft["missing_fields"]
    assert "subject_date_of_birth" in missing
    assert "filing_institution_identity" in missing
    assert "is_laundering" not in str(content).lower()
    assert "pattern type" not in str(content).lower()


def test_approval_requires_missing_acknowledgement(synthetic_cases, workflow):
    service = SARService(synthetic_cases, workflow)
    draft = service.build_draft("CASE-A1-20220101", "alice")
    with pytest.raises(WorkflowValidation):
        service.approve(draft["id"], "alice", acknowledge_missing=False)
    service.submit_for_review(draft["id"], "alice")
    with pytest.raises(WorkflowValidation):
        service.approve(draft["id"], "alice", acknowledge_missing=False)
    approved = service.approve(draft["id"], "alice", acknowledge_missing=True)
    assert approved["status"] == "APPROVED"
    assert approved["approved_by"] == "alice"
    pdf, _, _ = service.export(draft["id"], "pdf")
    assert b"APPROVED" in pdf


def test_update_resets_approval_and_creates_version(synthetic_cases, workflow):
    service = SARService(synthetic_cases, workflow)
    draft = service.build_draft("CASE-A1-20220101", "alice")
    service.submit_for_review(draft["id"], "alice")
    approved = service.approve(draft["id"], "alice", acknowledge_missing=True)
    content = approved["content"]
    content["narrative"]["conclusion"] = "Investigator edited conclusion."
    updated = service.update_draft(approved["id"], "bob", content)
    assert updated["version"] == 2
    assert updated["status"] == "DRAFT"
    assert updated["approved_by"] is None


def test_pdf_export_is_valid(synthetic_cases, workflow):
    service = SARService(synthetic_cases, workflow)
    draft = service.build_draft("CASE-A1-20220101", "alice")
    payload, media, filename = service.export(draft["id"], "pdf")
    assert media == "application/pdf"
    assert filename.endswith(".pdf")
    text = payload.decode("latin-1")
    assert text.startswith("%PDF-1.4")
    assert text.rstrip().endswith("%%EOF")
    assert "/Type /Page" in text
    assert re.search(r"/Count \d+", text)
    start = int(re.search(r"startxref\s+(\d+)", text).group(1))
    assert text[start:start + 4] == "xref"
    # Every xref entry points at an object header.
    header = text[:text.index("xref")]
    for match in re.finditer(r"^(\d{10}) 00000 n", text, re.M):
        offset = int(match.group(1))
        assert re.match(r"\d+ 0 obj", header[offset:offset + 20])
    assert "DRAFT" in text


def test_docx_export_is_valid(synthetic_cases, workflow):
    service = SARService(synthetic_cases, workflow)
    draft = service.build_draft("CASE-A1-20220101", "alice")
    payload, media, filename = service.export(draft["id"], "docx")
    assert "wordprocessingml" in media
    assert filename.endswith(".docx")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = set(archive.namelist())
        assert "word/document.xml" in names
        assert "[Content_Types].xml" in names
        document = archive.read("word/document.xml").decode("utf-8")
        assert "CASE-A1-20220101" in document
        assert "DRAFT" in document
        assert "Transaction Evidence" in document


def test_invalid_export_format(synthetic_cases, workflow):
    service = SARService(synthetic_cases, workflow)
    draft = service.build_draft("CASE-A1-20220101", "alice")
    with pytest.raises(ValueError):
        service.export(draft["id"], "xlsx")
