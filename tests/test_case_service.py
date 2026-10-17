"""Tests for the read-only case and transaction service."""

from __future__ import annotations

import pytest

from case_service import CaseNotFound, CaseService
from workflow_store import WorkflowStore, WorkflowValidation


def test_case_index_and_pagination(synthetic_cases):
    result = synthetic_cases.list_cases(page=1, page_size=2)
    assert result["total"] == 4
    assert result["page_count"] == 2
    assert len(result["items"]) == 2
    assert result["items"][0]["risk_score"] >= result["items"][1]["risk_score"]
    assert result["status_counts"]["NEW"] == 4


def test_case_id_is_account_and_day(synthetic_cases):
    case = synthetic_cases.get_case("CASE-A1-20220101")
    assert case["case_id"] == "CASE-A1-20220101"
    assert case["account_id"] == "A1"
    assert case["graph_date"] == "2022-01-01"
    assert case["risk_score"] == pytest.approx(0.90)
    assert case["holder_name"] == "Alpha LLC"
    assert case["transaction_summary"]["transaction_count"] == 4
    assert case["transaction_summary"]["aggregates"] == [
        {"direction": "in", "currency": "USD", "count": 2, "total": 150.0},
        {"direction": "out", "currency": "USD", "count": 2, "total": 340.0},
    ]
    assert "CASE-LEGACY0001" in case["legacy_report"]


def test_case_list_filters_and_status(synthetic_cases, workflow):
    workflow.set_case_status("run-test", "CASE-A1-20220101", "IN_REVIEW", "tester")
    result = synthetic_cases.list_cases(status="IN_REVIEW")
    assert result["total"] == 1
    assert result["items"][0]["status"] == "IN_REVIEW"
    assert result["items"][0]["case_id"] == "CASE-A1-20220101"
    high = synthetic_cases.list_cases(tier="TIER_1_HIGH")
    assert high["total"] == 2
    account = synthetic_cases.list_cases(account="A3")
    assert account["total"] == 1
    ranged = synthetic_cases.list_cases(date_from="2022-01-02", date_to="2022-01-02")
    assert ranged["total"] == 1


def test_unknown_case_raises(synthetic_cases):
    with pytest.raises(CaseNotFound):
        synthetic_cases.get_case("CASE-A1-20220102")  # not flagged
    with pytest.raises(CaseNotFound):
        synthetic_cases.get_case("CASE-NOPE-20220101")


def test_transactions_pagination_refs_and_filters(synthetic_cases):
    page = synthetic_cases.list_transactions("CASE-A1-20220101", page=1, page_size=2)
    assert page["total"] == 4
    assert page["page_count"] == 2
    assert len(page["items"]) == 2
    assert all(item["transaction_ref"].startswith("TX-") for item in page["items"])
    assert all("is_laundering" not in item for item in page["items"])
    assert all("pattern_type" not in item for item in page["items"])
    assert len({item["transaction_ref"] for item in page["items"]}) == 2

    inbound = synthetic_cases.list_transactions("CASE-A1-20220101", direction="in")
    assert inbound["total"] == 2
    assert all(item["direction"] == "in" for item in inbound["items"])
    outbound = synthetic_cases.list_transactions("CASE-A1-20220101", direction="out")
    assert outbound["total"] == 2

    large = synthetic_cases.list_transactions("CASE-A1-20220101", min_amount=200)
    assert large["total"] == 1
    assert large["items"][0]["amount"] == pytest.approx(300.0)

    by_counterparty = synthetic_cases.list_transactions(
        "CASE-A1-20220101", counterparty="A3"
    )
    assert by_counterparty["total"] == 2


def test_transactions_time_filter(synthetic_cases):
    filtered = synthetic_cases.list_transactions(
        "CASE-A1-20220101",
        from_value="2022-01-01T09:30:00",
        to_value="2022-01-01T10:30:00",
    )
    assert filtered["total"] == 1
    assert filtered["items"][0]["amount"] == pytest.approx(300.0)


def test_case_holder_names_are_source_record_values(synthetic_cases):
    names = synthetic_cases.holder_names(synthetic_cases.get_run(), ["A1", "A2", "A3"])
    assert names == {"A1": "Alpha LLC", "A2": "Seed Corp", "A3": "Gamma Trading"}


def test_transaction_refs_are_unique_in_fixture(synthetic_cases):
    items = synthetic_cases.list_transactions("CASE-A1-20220101", page_size=100)["items"]
    refs = [item["transaction_ref"] for item in items]
    assert len(refs) == len(set(refs)) == 4


def test_invalid_case_id(synthetic_cases):
    with pytest.raises(ValueError):
        synthetic_cases.get_case("A1")
    with pytest.raises(ValueError):
        synthetic_cases.list_transactions("CASE-A1-20220101", direction="sideways")
