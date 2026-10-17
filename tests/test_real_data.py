"""Real-pipeline integration tests; skipped when no saved run exists."""

from __future__ import annotations

import os

import pytest

import config
from case_service import CaseNotFound, CaseService
from graph_service import GraphService
from sar_service import SARService
from workflow_store import WorkflowStore


@pytest.fixture()
def real_cases(tmp_path):
    workflow = WorkflowStore(tmp_path / "real-workflow.db")
    try:
        cases = CaseService(workflow, config.OUTPUT_ROOT)
        run = cases.get_run()
    except CaseNotFound:
        pytest.skip("No saved pipeline run is available")
    return cases, workflow, run


def test_real_case_queue_has_cases(real_cases):
    cases, _, run = real_cases
    result = cases.list_cases(page=1, page_size=5)
    assert result["total"] > 1000
    assert len(result["items"]) == 5
    assert result["items"][0]["risk_score"] >= result["items"][1]["risk_score"]


def test_real_case_evidence(real_cases):
    cases, _, run = real_cases
    case_id = "CASE-8201A5710-20221024"
    try:
        case = cases.get_case(case_id)
    except CaseNotFound:
        pytest.skip("Reference case is not in the latest run")
    assert case["risk_score"] == pytest.approx(1.0)
    assert case["transaction_summary"]["transaction_count"] == 13
    assert case["transaction_summary"]["currencies"] == ["Yuan"]
    assert case["transaction_summary"]["aggregates"] == [
        {"direction": "in", "currency": "Yuan", "count": 3, "total": 54417.07},
        {"direction": "out", "currency": "Yuan", "count": 10, "total": 78728.31},
    ]
    tx = cases.list_transactions(case_id, page_size=50)
    assert tx["total"] == 13
    assert len({item["transaction_ref"] for item in tx["items"]}) == 13
    assert "Is Laundering" not in str(tx)
    assert "Pattern Type" not in str(tx)


def test_real_graph_is_case_scoped(real_cases):
    cases, _, run = real_cases
    graph = GraphService(cases)
    try:
        result = graph.case_graph("CASE-8201A5710-20221024", depth=1)
    except CaseNotFound:
        pytest.skip("Reference case is not in the latest run")
    assert result["case_id"] == "CASE-8201A5710-20221024"
    assert len(result["edges"]) > 0
    assert any(node["is_focal"] for node in result["nodes"])
    assert all(edge["transaction_ref"].startswith("TX-") for edge in result["edges"])


def test_real_sar_draft_and_export(real_cases):
    cases, workflow, run = real_cases
    try:
        cases.get_case("CASE-8201A5710-20221024")
    except CaseNotFound:
        pytest.skip("Reference case is not in the latest run")
    service = SARService(cases, workflow)
    draft = service.build_draft("CASE-8201A5710-20221024", "integration-tester")
    assert draft["content"]["activity"]["transaction_count"] == 13
    assert draft["content"]["activity"]["currencies"] == ["Yuan"]
    assert draft["missing_fields"]
    pdf, media, _ = service.export(draft["id"], "pdf")
    docx, media2, _ = service.export(draft["id"], "docx")
    assert pdf.startswith(b"%PDF")
    assert docx.startswith(b"PK")
    assert media == "application/pdf"
    assert "wordprocessingml" in media2
