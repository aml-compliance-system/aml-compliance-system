"""Unit tests for the three evidence tools and the tool registry."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import tools
from tests.conftest import assert_json_safe


def test_tool_registry_exposes_three_tools():
    assert tools.tool_names() == [
        "get_account_scores",
        "get_transactions",
        "get_graph_exposure",
    ]
    assert len(tools.TOOL_SCHEMAS) == 3
    assert {schema["function"]["name"] for schema in tools.TOOL_SCHEMAS} == set(
        tools.tool_names()
    )


def test_get_account_scores_single_day(store):
    result = tools.get_account_scores("A1", "2022-01-01", store=store)
    assert result["ok"] is True
    assert result["tool"] == "get_account_scores"
    assert result["account_id"] == "A1"
    summary = result["data"]["summary"]
    assert summary["date"] == "2022-01-01"
    assert summary["final_risk_score"] == pytest.approx(0.90)
    assert summary["risk_label"] == "High Risk"
    assert summary["investigation_tier"] == "TIER_1_HIGH"
    assert summary["model_scores"] == {
        "xgboost": 0.80,
        "anomaly": 0.90,
        "behavioral": 0.60,
    }
    assert summary["graph_features"]["hop1_count"] == 1
    assert result["data"]["thresholds"]["investigation"] == pytest.approx(0.411, abs=1e-6)
    assert_json_safe(result)


def test_get_account_scores_latest_and_highest_day(store):
    result = tools.get_account_scores("A1", store=store)
    assert result["ok"] is True
    # No date -> latest score is the summary, highest-risk day is reported.
    assert result["data"]["summary"]["date"] == "2022-01-02"
    assert result["data"]["summary"]["final_risk_score"] == pytest.approx(0.30)
    assert result["data"]["highest_risk_day"]["date"] == "2022-01-01"
    assert result["data"]["highest_risk_day"]["final_risk_score"] == pytest.approx(0.90)
    assert len(result["data"]["history"]) == 2
    # 2022-01-02 has a single account/day row, so its percentile is 100.
    assert result["data"]["summary"]["population_percentile"] == pytest.approx(100.0)


def test_get_account_scores_unknown_account(store):
    result = tools.get_account_scores("NOPE", "2022-01-01", store=store)
    assert result["ok"] is False
    assert result["error"]["code"] == "ACCOUNT_NOT_FOUND"
    assert result["account_id"] == "NOPE"
    assert_json_safe(result)


def test_get_account_scores_invalid_date(store):
    result = tools.get_account_scores("A1", "01/02/2022", store=store)
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_DATE"


def test_get_transactions_summary_and_rows(store):
    result = tools.get_transactions("A1", "2022-01-01", store=store)
    assert result["ok"] is True
    data = result["data"]
    summary = data["summary"]
    assert summary["transaction_count"] == 3
    assert summary["inbound"] == {"count": 2, "total": 150.0}
    assert summary["outbound"] == {"count": 1, "total": 300.0}
    assert summary["net_flow"] == pytest.approx(-150.0)
    assert summary["largest_transaction"] == pytest.approx(300.0)
    assert summary["unique_counterparties"] == 2

    directions = {(row["direction"], row["counterparty"]) for row in data["transactions"]}
    assert ("in", "A2") in directions
    assert ("out", "A3") in directions

    top = {item["counterparty"]: item for item in data["top_counterparties"]}
    assert top["A3"]["outbound_total"] == pytest.approx(300.0)
    assert top["A3"]["inbound_total"] == pytest.approx(50.0)
    assert_json_safe(result)


def test_get_transactions_direction_and_limit(store):
    inbound = tools.get_transactions(
        "A1", "2022-01-01", direction="in", limit=1, store=store
    )
    assert inbound["ok"] is True
    assert inbound["data"]["rows_available"] == 2
    assert inbound["data"]["rows_returned"] == 1
    assert all(row["direction"] == "in" for row in inbound["data"]["transactions"])

    outbound = tools.get_transactions("A1", "2022-01-01", direction="out", store=store)
    assert outbound["data"]["rows_available"] == 1
    assert outbound["data"]["summary"]["inbound"]["count"] == 0


def test_get_transactions_min_amount_and_limit_clamp(store):
    filtered = tools.get_transactions(
        "A1", "2022-01-01", min_amount=120, store=store
    )
    assert filtered["data"]["rows_available"] == 1  # only the 300 outbound
    clamped = tools.get_transactions(
        "A1", "2022-01-01", limit=10_000, store=store
    )
    assert clamped["meta"]["limit"] == tools.MAX_TX_LIMIT


def test_get_transactions_defaults_to_latest_scored_day(store):
    result = tools.get_transactions("A1", store=store)
    assert result["ok"] is True
    assert result["data"]["date"] == "2022-01-02"
    assert result["meta"]["date_source"] == "latest_risk_date"
    assert result["data"]["summary"]["transaction_count"] == 1


def test_get_transactions_no_rows_is_not_an_error(store):
    result = tools.get_transactions("A9", "2022-01-01", store=store)
    assert result["ok"] is True
    assert result["data"]["summary"]["transaction_count"] == 0
    assert result["data"]["transactions"] == []


def test_get_transactions_invalid_direction(store):
    result = tools.get_transactions("A1", direction="sideways", store=store)
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_DIRECTION"


def test_get_graph_exposure_one_hop_seed(store):
    result = tools.get_graph_exposure("A1", "2022-01-01", store=store)
    assert result["ok"] is True
    data = result["data"]
    assert data["date"] == "2022-01-01"
    assert data["exposure"]["hop1_direct_seed_count"] == 1
    assert data["exposure"]["seed_inbound_accounts"] == 1
    assert data["exposure"]["seed_outbound_accounts"] == 0
    assert data["flags"]["direct_seed_exposure"] is True
    assert data["flags"]["received_from_high_risk"] is True
    assert data["seed_neighbours"] == [
        {
            "seed_account": "A2",
            "direction": "inbound_from_seed",
            "total": 100.0,
            "count": 1,
        }
    ]
    # hop1=1, volume_factor=1 -> direct component is not capped
    assert data["penalty_cap_applied"] is False
    assert data["penalty_components"]["direct_hop1"] == pytest.approx(
        0.3046, abs=1e-4
    )
    assert_json_safe(result)


def test_get_graph_exposure_seed_itself(store):
    result = tools.get_graph_exposure("A2", "2022-01-01", store=store)
    assert result["ok"] is True
    assert result["data"]["flags"]["is_high_risk_seed"] is True
    assert any(
        "itself an original high-risk seed" in note
        for note in result["data"]["interpretation"]
    )


def test_get_graph_exposure_unknown_account(store):
    result = tools.get_graph_exposure("NOPE", "2022-01-01", store=store)
    assert result["ok"] is False
    assert result["error"]["code"] == "ACCOUNT_NOT_FOUND"


def test_dispatch_tool_unknown_and_bad_arguments(store):
    unknown = tools.dispatch_tool("do_something", {}, store=store)
    assert unknown["ok"] is False
    assert unknown["error"]["code"] == "UNKNOWN_TOOL"

    bad_json = tools.dispatch_tool("get_account_scores", "{not json", store=store)
    assert bad_json["error"]["code"] == "INVALID_ARGUMENTS"

    missing = tools.dispatch_tool("get_account_scores", {}, store=store)
    assert missing["error"]["code"] == "INVALID_ARGUMENTS"

    wrong_type = tools.dispatch_tool("get_account_scores", ["A1"], store=store)
    assert wrong_type["error"]["code"] == "INVALID_ARGUMENTS"


def test_dispatch_tool_accepts_json_string_arguments(store):
    result = tools.dispatch_tool(
        "get_transactions",
        json.dumps({"account_id": "A1", "date": "2022-01-01"}),
        store=store,
    )
    assert result["ok"] is True
    assert result["data"]["summary"]["transaction_count"] == 3


def test_case_report_registration(bound_store):
    assert bound_store.case_report("A1") == "SAR draft for A1"
    assert bound_store.case_report("A2") is None
    assert bound_store.source == "pipeline-cache"


def test_store_without_data_returns_structured_error():
    empty = tools.AMLDataStore()
    result = tools.get_account_scores("A1", store=empty)
    assert result["ok"] is False
    assert result["error"]["code"] == "NO_DATA"


def test_scores_and_transactions_agree_on_dates(store):
    scores = tools.get_account_scores("A1", store=store)["data"]
    assert {item["date"] for item in scores["history"]} == {
        "2022-01-01",
        "2022-01-02",
    }
    tx = tools.get_transactions("A1", "2022-01-02", store=store)
    assert tx["data"]["date"] == "2022-01-02"
    assert tx["data"]["summary"]["transaction_count"] == 1
