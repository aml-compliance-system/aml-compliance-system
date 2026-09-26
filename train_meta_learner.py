"""
One-time training script for the stacking meta-learner.
Run this AFTER your base models + graph have scored the labeled dataset.

Usage:
    python train_meta_learner.py
"""
import pandas as pd
import numpy as np
import config
from meta_learner import train_meta_learner
from nodes import (
    run_xgboost_model,
    run_anomaly_model,
    run_behavioral_model,
    graph_context_node,
)
from customer_deviation import add_customer_deviation_features


def main():
    print("=" * 70)
    print("TRAINING META-LEARNER")
    print("=" * 70)

    # 1. Load labeled data
    df = pd.read_parquet(config.DEFAULT_PARQUET_PATH)
    print(f"Loaded {len(df)} transactions")

    # 2. Add customer deviation features
    df = add_customer_deviation_features(df)

    # 3. Run base models
    state = {"final_df": df, "ml_probabilities": {}}

    xgb_out = run_xgboost_model(state)
    state["ml_probabilities"].update(xgb_out["ml_probabilities"])

    ae_out = run_anomaly_model(state)
    state["ml_probabilities"].update(ae_out["ml_probabilities"])

    beh_out = run_behavioral_model(state)
    state["ml_probabilities"].update(beh_out["ml_probabilities"])

    # 4. Run graph
    graph_out = graph_context_node(state)
    state["subgraph_data"] = graph_out["subgraph_data"]

    # 5. Merge graph features back to transaction-level for training
    account_results = graph_out["subgraph_data"]["account_results"]
    df["_graph_date"] = pd.to_datetime(
        df["Transaction_Date"] if "Transaction_Date" in df.columns else df["Timestamp"]
    ).dt.normalize()

    df = df.merge(
        account_results[["graph_date", "Account", "base_risk", "graph_penalty",
                         "hop1_count", "hop2_count", "hop3_count",
                         "bidir_count", "fan_score", "daily_degree"]],
        left_on=["_graph_date", "Account"],
        right_on=["graph_date", "Account"],
        how="left",
    )

    # Add model scores to df
    for key in ["xgb_prob", "anomaly_score", "behavioral_score"]:
        df[key] = state["ml_probabilities"][key]

    # Fill NaN graph features with 0
    for col in ["base_risk", "graph_penalty", "hop1_count", "hop2_count", "hop3_count",
                "bidir_count", "fan_score", "daily_degree"]:
        df[col] = df[col].fillna(0.0)

    # 6. Train meta-learner
    train_meta_learner(df, label_col="Is Laundering")

    print("\n" + "=" * 70)
    print("✅ DONE. Meta-learner is ready for inference.")
    print("=" * 70)


if __name__ == "__main__":
    main()