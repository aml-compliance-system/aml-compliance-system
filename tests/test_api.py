"""FastAPI endpoint, validation, auth and workflow tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import config
from agent import AgentResult, ToolCallRecord
from case_service import CaseService
from graph_service import GraphService
from sar_service import SARService
from tests.conftest import build_synthetic_run
from workflow_store import WorkflowStore


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import main

    root = tmp_path / "outputs"
    build_synthetic_run(root)
    workflow = WorkflowStore(tmp_path / "workflow.db")
    cases = CaseService(workflow, root)
    graphs = GraphService(cases)
    sar = SARService(cases, workflow)
    main.app.state.workflow = workflow
    main.app.state.cases = cases
    main.app.state.graphs = graphs
    main.app.state.sar = sar
    main.app.state.agent_cache = {}
    monkeypatch.setattr(config, "AGENT_WARMUP", False)
    with TestClient(main.app) as client:
        yield client


def test_cases_and_case_detail(api_client):
    response = api_client.get("/api/cases")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 4
    assert payload["available_dates"] == ["2022-01-01", "2022-01-02"]
    assert payload["items"][0]["case_id"].startswith("CASE-")
    assert "transaction_summary" not in payload["items"][0]

    detail = api_client.get("/api/cases/CASE-A1-20220101")
    assert detail.status_code == 200
    data = detail.json()
    assert data["account_id"] == "A1"
    assert data["transaction_summary"]["transaction_count"] == 4
    assert data["thresholds"]["investigation"] == pytest.approx(0.411)
    assert data["model_scores"]["xgboost"] == pytest.approx(0.80)

    missing = api_client.get("/api/cases/CASE-NOPE-20220101")
    assert missing.status_code == 404


def test_transactions_endpoint_and_ground_truth_exclusion(api_client):
    response = api_client.get("/api/cases/CASE-A1-20220101/transactions?page_size=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4
    assert len(data["items"]) == 2
    assert data["items"][0]["transaction_ref"].startswith("TX-")
    serialized = json.dumps(data).lower()
    assert "is_laundering" not in serialized
    assert "pattern type" not in serialized
    inbound = api_client.get("/api/cases/CASE-A1-20220101/transactions?direction=in")
    assert inbound.status_code == 200
    assert inbound.json()["total"] == 2


def test_graph_endpoints(api_client):
    graph = api_client.get("/api/cases/CASE-A1-20220101/graph?depth=1")
    assert graph.status_code == 200
    data = graph.json()
    assert len(data["edges"]) == 4
    assert data["nodes"][0]["is_focal"] is True

    paths = api_client.post("/api/cases/CASE-A1-20220101/graph/paths", json={
        "source": "A2", "target": "A3", "max_depth": 3
    })
    assert paths.status_code == 200
    assert paths.json()["findings"][0]["nodes"] == ["A2", "A1", "A3"]

    cycles = api_client.post("/api/cases/CASE-A1-20220101/graph/cycles", json={"max_length": 3})
    assert cycles.status_code == 200
    assert cycles.json()["findings"]


def test_notes_require_confirmation_and_persist(api_client):
    refused = api_client.post("/api/cases/CASE-A1-20220101/notes", json={
        "body": "Not confirmed", "note_type": "NOTE", "confirm": False
    })
    assert refused.status_code == 400
    created = api_client.post("/api/cases/CASE-A1-20220101/notes", json={
        "body": "Confirmed finding", "note_type": "FINDING", "confirm": True
    })
    assert created.status_code == 201
    note = created.json()
    assert note["version"] == 1
    updated = api_client.patch(f"/api/notes/{note['id']}", json={
        "body": "Edited finding", "note_type": "FINDING", "confirm": True
    })
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    listed = api_client.get("/api/cases/CASE-A1-20220101/notes")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    timeline = api_client.get("/api/cases/CASE-A1-20220101/timeline")
    actions = [item["action"] for item in timeline.json()]
    assert "NOTE_CREATED" in actions
    assert "NOTE_UPDATED" in actions


def test_status_update_requires_confirmation(api_client):
    refused = api_client.patch("/api/cases/CASE-A1-20220101/status", json={
        "status": "IN_REVIEW", "confirm": False
    })
    assert refused.status_code == 400
    updated = api_client.patch("/api/cases/CASE-A1-20220101/status", json={
        "status": "IN_REVIEW", "confirm": True
    })
    assert updated.status_code == 200
    assert updated.json()["state"]["status"] == "IN_REVIEW"


def test_sar_create_and_export(api_client):
    created = api_client.post("/api/cases/CASE-A1-20220101/sar/drafts", json={"confirm": True})
    assert created.status_code == 201
    draft = created.json()
    assert draft["status"] == "DRAFT"
    assert draft["content"]["case"]["case_id"] == "CASE-A1-20220101"
    assert draft["missing_fields"]
    assert draft["validation"]["valid"] is True

    pdf = api_client.post(f"/api/sar/drafts/{draft['id']}/export?format=pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    assert "attachment" in pdf.headers["content-disposition"]

    docx = api_client.post(f"/api/sar/drafts/{draft['id']}/export?format=docx")
    assert docx.status_code == 200
    assert docx.content[:2] == b"PK"


def test_sar_review_and_approval_flow(api_client):
    created = api_client.post("/api/cases/CASE-A1-20220101/sar/drafts", json={"confirm": True})
    draft = created.json()
    early = api_client.post(f"/api/sar/drafts/{draft['id']}/approve", json={"confirm": True, "acknowledge_missing": True})
    assert early.status_code == 400  # review is mandatory
    reviewed = api_client.post(f"/api/sar/drafts/{draft['id']}/submit-review", json={"confirm": True})
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "IN_REVIEW"
    approved = api_client.post(f"/api/sar/drafts/{draft['id']}/approve", json={"confirm": True, "acknowledge_missing": True})
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"


def test_sar_requires_confirmation(api_client):
    response = api_client.post("/api/cases/CASE-A1-20220101/sar/drafts", json={"confirm": False})
    assert response.status_code == 400


def test_auth_gate(api_client, monkeypatch):
    monkeypatch.setattr(config, "API_TOKEN", "test-secret")
    unauthenticated = api_client.get("/api/cases")
    assert unauthenticated.status_code == 401
    authenticated = api_client.get("/api/cases", headers={"Authorization": "Bearer test-secret"})
    assert authenticated.status_code == 200
    wrong = api_client.get("/api/cases", headers={"Authorization": "Bearer wrong"})
    assert wrong.status_code == 401


def test_pipeline_input_is_restricted(api_client, tmp_path):
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(outside)
    response = api_client.post("/pipeline/analyze", json={"final_parquet_path": str(outside)})
    assert response.status_code == 403


def test_case_chat_returns_trace_and_audits(api_client, monkeypatch):
    import api_cases

    class FakeAgent:
        def chat(self, account, message, *, history=None, case_date=None):
            return AgentResult(
                answer=f"Evidence answer for {account} on {case_date}",
                case_id=account,
                tool_calls=[ToolCallRecord("get_account_scores", {"account_id": account}, True, 5, "auto", "ok")],
                steps=1,
                model="fake",
            )

    monkeypatch.setattr(api_cases, "_agent", lambda request, run: FakeAgent())
    response = api_client.post("/api/cases/CASE-A1-20220101/chat", json={"message": "Why flagged?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("Evidence answer")
    assert data["tool_calls"][0]["name"] == "get_account_scores"
    timeline = api_client.get("/api/cases/CASE-A1-20220101/timeline")
    assert "AI_QUERY" in [item["action"] for item in timeline.json()]


def test_meta_endpoint(api_client):
    response = api_client.get("/api/meta")
    assert response.status_code == 200
    data = response.json()
    assert data["policy_validation"]["available"] is False
    assert "sar_export" in data["capabilities"]
    assert data["limits"]["graph_nodes"] > 0


def test_ui_is_served_from_the_api_with_correct_mime(api_client):
    index = api_client.get("/ui/")
    assert index.status_code == 200
    assert "FINDETECT" in index.text
    js = api_client.get("/ui/assets/cases.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("text/javascript")
    css = api_client.get("/ui/assets/app.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    root = api_client.get("/", follow_redirects=False)
    assert root.status_code in (307, 308)
    assert root.headers["location"] == "/ui/"


def test_cors_allows_local_frontend_origin(api_client):
    origin = "http://127.0.0.1:5600"
    response = api_client.get("/api/meta", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
