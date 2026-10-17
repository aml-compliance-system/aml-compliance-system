"""Shared fixtures for the AML agent tests (synthetic data, no Ollama needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import tools


def build_synthetic_run(root: Path) -> Path:
    """Create one run folder with a small risk/transactions Parquet contract."""
    run = root / "run-test"
    run.mkdir(parents=True, exist_ok=True)
    risk_rows = [
        {"graph_date": "2022-01-01", "Account": "A1", "final_risk_score": 0.90,
         "risk_label": "High Risk", "investigation_tier": "TIER_1_HIGH",
         "base_risk": 0.70, "graph_penalty": 0.20, "hop1_count": 2, "hop2_count": 0,
         "hop3_count": 0, "bidir_count": 1, "fan_score": 0.0, "daily_degree": 3,
         "is_high_risk_seed": False, "received_from_high_risk": True,
         "xgb_prob": 0.80, "anomaly_score": 0.90, "behavioral_score": 0.60,
         "meta_score": 0.90, "investigation_flag": True},
        {"graph_date": "2022-01-02", "Account": "A1", "final_risk_score": 0.30,
         "risk_label": "Medium Risk", "investigation_tier": "NOT_FLAGGED",
         "base_risk": 0.30, "graph_penalty": 0.0, "hop1_count": 0, "hop2_count": 0,
         "hop3_count": 0, "bidir_count": 0, "fan_score": 0.0, "daily_degree": 1,
         "is_high_risk_seed": False, "received_from_high_risk": False,
         "xgb_prob": 0.30, "anomaly_score": 0.30, "behavioral_score": 0.30,
         "meta_score": 0.30, "investigation_flag": False},
        {"graph_date": "2022-01-01", "Account": "A2", "final_risk_score": 0.80,
         "risk_label": "High Risk", "investigation_tier": "TIER_1_HIGH",
         "base_risk": 0.80, "graph_penalty": 0.0, "hop1_count": 0, "hop2_count": 0,
         "hop3_count": 0, "bidir_count": 0, "fan_score": 0.0, "daily_degree": 2,
         "is_high_risk_seed": True, "received_from_high_risk": False,
         "xgb_prob": 0.75, "anomaly_score": 0.85, "behavioral_score": 0.50,
         "meta_score": 0.80, "investigation_flag": True},
        {"graph_date": "2022-01-01", "Account": "A3", "final_risk_score": 0.50,
         "risk_label": "High Risk", "investigation_tier": "TIER_3_LOW",
         "base_risk": 0.45, "graph_penalty": 0.05, "hop1_count": 1, "hop2_count": 0,
         "hop3_count": 0, "bidir_count": 0, "fan_score": 0.0, "daily_degree": 2,
         "is_high_risk_seed": False, "received_from_high_risk": True,
         "xgb_prob": 0.50, "anomaly_score": 0.50, "behavioral_score": 0.40,
         "meta_score": 0.50, "investigation_flag": True},
        {"graph_date": "2022-01-02", "Account": "A4", "final_risk_score": 0.60,
         "risk_label": "High Risk", "investigation_tier": "TIER_3_LOW",
         "base_risk": 0.55, "graph_penalty": 0.05, "hop1_count": 0, "hop2_count": 0,
         "hop3_count": 0, "bidir_count": 0, "fan_score": 0.0, "daily_degree": 1,
         "is_high_risk_seed": False, "received_from_high_risk": False,
         "xgb_prob": 0.60, "anomaly_score": 0.60, "behavioral_score": 0.40,
         "meta_score": 0.60, "investigation_flag": True},
    ]
    pd.DataFrame(risk_rows).to_parquet(run / "risk_scores.parquet", index=False)

    tx_rows = [
        ("2022-01-01 09:00:00", "A2", "A1", 100.0, 100.0, "USD", "USD", "Cash", 1, 2,
         "E-A2", "Seed Corp", "E-A1", "Alpha LLC", 0, "NO_PATTERN"),
        ("2022-01-01 10:00:00", "A1", "A3", 300.0, 300.0, "USD", "USD", "Wire", 2, 3,
         "E-A1", "Alpha LLC", "E-A3", "Gamma Trading", 0, "NO_PATTERN"),
        ("2022-01-01 11:00:00", "A3", "A1", 50.0, 50.0, "USD", "USD", "Credit Card", 3, 2,
         "E-A3", "Gamma Trading", "E-A1", "Alpha LLC", 0, "NO_PATTERN"),
        ("2022-01-01 12:00:00", "A1", "A2", 40.0, 40.0, "USD", "USD", "Wire", 2, 1,
         "E-A1", "Alpha LLC", "E-A2", "Seed Corp", 0, "NO_PATTERN"),
        ("2022-01-02 09:00:00", "A4", "A3", 25.0, 25.0, "EUR", "EUR", "Cash", 4, 3,
         "E-A4", "Delta", "E-A3", "Gamma Trading", 0, "NO_PATTERN"),
        ("2022-01-02 10:00:00", "A1", "A3", 10.0, 10.0, "EUR", "EUR", "Wire", 2, 3,
         "E-A1", "Alpha LLC", "E-A3", "Gamma Trading", 0, "NO_PATTERN"),
    ]
    tx = pd.DataFrame(tx_rows, columns=[
        "Timestamp", "Account", "Account_duplicated_0", "Amount Paid",
        "Amount Received", "Payment Currency", "Receiving Currency",
        "Payment Format", "From Bank", "To Bank", "From_Entity_ID",
        "From_Entity_Name", "To_Entity_ID", "To_Entity_Name",
        "Is Laundering", "Pattern Type",
    ])
    tx["Timestamp"] = pd.to_datetime(tx["Timestamp"])
    tx_path = root / "transactions.parquet"
    tx.to_parquet(tx_path, index=False)

    (run / "summary.json").write_text(json.dumps({
        "graph_date": "run-test",
        "sar_report_count": 1,
        "input_parquet": str(tx_path),
        "account_day_rows": len(risk_rows),
        "investigation_cases": 4,
    }), encoding="utf-8")
    legacy = "- **Case ID**: CASE-LEGACY0001\n- **Customer**: A1 (Redacted Entity) | Segment: Test\n- **Graph Date**: 2022-01-01\n- **Risk Assessment**: Score 0.900"
    (run / "sar_reports.jsonl").write_text(json.dumps({"report_id": 1, "graph_date": "run-test", "report": legacy}) + "\n", encoding="utf-8")
    return run


def _risk_row(**overrides):
    row = {
        "graph_date": "2022-01-01",
        "Account": "A1",
        "base_risk": 0.70,
        "daily_degree": 2.0,
        "is_high_risk_seed": False,
        "hop1_count": 1.0,
        "total_volume": 400.0,
        "volume_factor": 1.0,
        "hop2_count": 0.0,
        "hop3_count": 0.0,
        "bidir_count": 0.0,
        "fan_score": 0.0,
        "graph_penalty": 0.20,
        "final_risk_score": 0.90,
        "received_from_high_risk": True,
        "xgb_prob": 0.80,
        "anomaly_score": 0.90,
        "behavioral_score": 0.60,
        "amount_vs_own_mean": 2.0,
        "amount_vs_own_median": 1.5,
        "amount_zscore_vs_own": 3.0,
        "velocity_vs_own_avg": 1.2,
        "meta_score": 0.90,
        "investigation_flag": True,
        "investigation_tier": "TIER_1_HIGH",
        "risk_label": "High Risk",
    }
    row.update(overrides)
    return row


@pytest.fixture()
def risk_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _risk_row(),
            _risk_row(
                graph_date="2022-01-02",
                base_risk=0.30,
                graph_penalty=0.0,
                final_risk_score=0.30,
                meta_score=0.30,
                investigation_flag=False,
                investigation_tier="NOT_FLAGGED",
                risk_label="Low Risk",
                received_from_high_risk=False,
                hop1_count=0.0,
                total_volume=0.0,
                xgb_prob=0.30,
                anomaly_score=0.30,
                behavioral_score=0.30,
            ),
            _risk_row(
                Account="A2",
                base_risk=0.80,
                daily_degree=1.0,
                is_high_risk_seed=True,
                hop1_count=0.0,
                total_volume=0.0,
                graph_penalty=0.0,
                final_risk_score=0.80,
                received_from_high_risk=False,
                investigation_flag=True,
                risk_label="High Risk",
            ),
            _risk_row(
                Account="A3",
                base_risk=0.40,
                daily_degree=2.0,
                hop1_count=0.0,
                total_volume=0.0,
                graph_penalty=0.0,
                final_risk_score=0.40,
                received_from_high_risk=False,
                investigation_flag=False,
                investigation_tier="TIER_3_LOW",
                risk_label="Medium Risk",
            ),
        ]
    )


@pytest.fixture()
def tx_df() -> pd.DataFrame:
    rows = [
        # seed A2 sends to A1
        ("2022-01-01 09:00", "A2", "A1", 100.0, 100.0, "Cash"),
        # A1 sends out to A3
        ("2022-01-01 10:00", "A1", "A3", 300.0, 300.0, "Wire"),
        # A3 sends back to A1 (not a seed)
        ("2022-01-01 11:00", "A3", "A1", 50.0, 50.0, "Credit Card"),
        # unrelated
        ("2022-01-01 12:00", "A4", "A3", 25.0, 25.0, "Cash"),
        # next day
        ("2022-01-02 09:00", "A1", "A3", 10.0, 10.0, "Wire"),
    ]
    return pd.DataFrame(
        [
            {
                "Timestamp": ts,
                "Account": sender,
                "Account_duplicated_0": receiver,
                "Amount Paid": paid,
                "Amount Received": received,
                "Payment Currency": "USD",
                "Receiving Currency": "USD",
                "Payment Format": fmt,
                "From Bank": 1,
                "To Bank": 2,
                "Is Laundering": 0,
            }
            for ts, sender, receiver, paid, received, fmt in rows
        ]
    )


@pytest.fixture()
def store(risk_df, tx_df) -> tools.AMLDataStore:
    return tools.AMLDataStore(
        risk_scores_df=risk_df,
        transactions_df=tx_df,
        run_label="test-run",
    )


@pytest.fixture()
def bound_store(risk_df, tx_df) -> tools.AMLDataStore:
    store = tools.AMLDataStore(run_label="test-run")
    store.register_pipeline_run(
        final_state={"risk_scores": risk_df},
        transactions_df=tx_df,
        reports={"A1": "SAR draft for A1"},
        run_label="test-run",
    )
    return store


def assert_json_safe(payload) -> None:
    json.dumps(payload, allow_nan=False)


@pytest.fixture()
def workflow(tmp_path):
    from workflow_store import WorkflowStore

    return WorkflowStore(tmp_path / "workflow.db")


@pytest.fixture()
def synthetic_cases(tmp_path, workflow):
    from case_service import CaseService

    root = tmp_path / "outputs"
    build_synthetic_run(root)
    return CaseService(workflow, root)
