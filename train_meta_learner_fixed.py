"""
Final corrected meta-learner training script.

Fixes THREE bugs found across this debugging session:

BUG 1 -- Trained and evaluated on the SAME data.
    config.DEFAULT_PARQUET_PATH points at prod_sample_oct23_25.parquet --
    the held-out window. Training the meta-learner on it means every later
    "held-out" evaluation against that file measures memorization, not
    generalization.
    FIX: --train-window must be a DIFFERENT window than whatever you
    evaluate the pipeline against.

BUG 2 -- Training grain != inference grain (caused the 99.1% flood).
    meta_learner_node MAX-aggregates xgb_prob/anomaly_score/behavioral_score
    across every transaction an account made that day before scoring. The
    original script trained on per-transaction values instead, so the model
    never saw the (systematically higher) max-aggregated values it would be
    scored on at inference.
    FIX: aggregate to ACCOUNT/DAY grain with the same max() aggregation
    meta_learner_node uses, for both features and label, before training.

BUG 3 -- Out-of-memory loading the full 55-day train window.
    Your original training notebook queried with a 10% negative-sample
    filter for exactly this reason: `Is Laundering = 1 OR random() < 0.10`.
    Keep ALL positives (extremely rare), downsample negatives. My first
    version of this script dropped that and tried to load the whole window
    unsampled -> OOM on a 55-day range.
    FIX: --neg-sample-rate (default 0.10) reproduces the original notebook's
    approach. Set to 1.0 to disable sampling if you have the RAM / use a
    shorter window instead.

Usage:
    python train_meta_learner_fixed.py --train-window 2022-08-08:2022-08-25 --input-glob "Data/processed/block_d_numerical_bucket_*.parquet" --neg-sample-rate 0.10
"""
import argparse

import duckdb
import numpy as np
import pandas as pd

import config
from meta_learner import train_meta_learner
from nodes import (
    run_xgboost_model,
    run_anomaly_model,
    run_behavioral_model,
    graph_context_node,
)
from customer_deviation import add_customer_deviation_features


def build_account_day_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce EXACTLY what meta_learner_node does at inference time, so the
    model trains on the same feature distributions it will be scored on.
    """
    state = {"final_df": df, "ml_probabilities": {}}

    print("Running base models (transaction grain)...")
    xgb_out = run_xgboost_model(state)
    state["ml_probabilities"].update(xgb_out["ml_probabilities"])

    ae_out = run_anomaly_model(state)
    state["ml_probabilities"].update(ae_out["ml_probabilities"])

    beh_out = run_behavioral_model(state)
    state["ml_probabilities"].update(beh_out["ml_probabilities"])

    for key in ["xgb_prob", "anomaly_score", "behavioral_score"]:
        df[key] = state["ml_probabilities"][key]

    print("Running graph node...")
    graph_out = graph_context_node(state)
    account_results = graph_out["subgraph_data"]["account_results"].copy()

    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Timestamp"
    df["_graph_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # --- Aggregate model scores to account/day grain, MAX -- matches
    #     meta_learner_node's aggregation exactly. Fix for Bug 2. ---
    account_scores = df.groupby(["_graph_date", "Account"]).agg(
        xgb_prob=("xgb_prob", "max"),
        anomaly_score=("anomaly_score", "max"),
        behavioral_score=("behavioral_score", "max"),
        amount_vs_own_mean=("amount_vs_own_mean", "max") if "amount_vs_own_mean" in df.columns else ("Amount Paid", "size"),
        amount_vs_own_median=("amount_vs_own_median", "max") if "amount_vs_own_median" in df.columns else ("Amount Paid", "size"),
        amount_zscore_vs_own=("amount_zscore_vs_own", "max") if "amount_zscore_vs_own" in df.columns else ("Amount Paid", "size"),
        velocity_vs_own_avg=("velocity_vs_own_avg", "max") if "velocity_vs_own_avg" in df.columns else ("Amount Paid", "size"),
    ).reset_index()

    # --- Label at the SAME account/day grain: positive if any transaction
    #     that account/day is laundering. ---
    account_labels = (
        df.groupby(["_graph_date", "Account"])["Is Laundering"]
        .max()
        .reset_index()
    )

    training_table = account_results.merge(
        account_scores,
        left_on=["graph_date", "Account"],
        right_on=["_graph_date", "Account"],
        how="left",
    ).merge(
        account_labels,
        left_on=["graph_date", "Account"],
        right_on=["_graph_date", "Account"],
        how="left",
        suffixes=("", "_lbl"),
    )

    training_table = training_table.drop(
        columns=[c for c in training_table.columns if c.endswith("_lbl") or c == "_graph_date"],
        errors="ignore",
    )

    for col in ["xgb_prob", "anomaly_score", "behavioral_score",
                "amount_vs_own_mean", "amount_vs_own_median",
                "amount_zscore_vs_own", "velocity_vs_own_avg"]:
        if col not in training_table.columns:
            training_table[col] = 0.0
        training_table[col] = pd.to_numeric(training_table[col], errors="coerce").fillna(0.0)

    training_table["Is Laundering"] = training_table["Is Laundering"].fillna(0).astype(int)

    return training_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-window", required=True,
        help="Date window to train on, format 'YYYY-MM-DD:YYYY-MM-DD'. "
             "MUST differ from whatever window you evaluate the pipeline against.",
    )
    parser.add_argument(
        "--input-glob", default=None,
        help="Glob for bucket parquet files; defaults to config.FINAL_GLOB",
    )
    parser.add_argument(
        "--neg-sample-rate", type=float, default=0.10,
        help="Fraction of NEGATIVE (non-laundering) transactions to keep, to fit in memory. "
             "ALL positive/laundering transactions are always kept. Matches the original "
             "training notebook's approach. Set to 1.0 to disable sampling.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the negative sample, so the run is reproducible.",
    )
    args = parser.parse_args()

    start_str, end_str = args.train_window.split(":")
    input_glob = args.input_glob or config.FINAL_GLOB

    con = duckdb.connect()
    con.execute(f"SELECT setseed({args.seed / 2147483647.0})")  # deterministic random()

    time_filter = (
        f"Timestamp >= '{start_str}' "
        f"AND Timestamp < CAST('{end_str}' AS DATE) + INTERVAL 1 DAY"
    )
    if args.neg_sample_rate >= 1.0:
        where_clause = time_filter
    else:
        where_clause = f"{time_filter} AND (\"Is Laundering\" = 1 OR random() < {args.neg_sample_rate})"

    print(f"Loading training window {start_str} -> {end_str} "
          f"(neg-sample-rate={args.neg_sample_rate}) from {input_glob} ...")
    df = con.execute(f"""
        SELECT * FROM read_parquet('{input_glob}')
        WHERE {where_clause}
    """).df()
    print(f"Loaded {len(df):,} transactions "
          f"({int(df['Is Laundering'].sum()):,} positive).")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    print("Adding customer deviation features...")
    df = add_customer_deviation_features(df)

    print("Building account/day training table (matches inference grain exactly)...")
    training_table = build_account_day_training_table(df)

    print(f"Training rows (account/day grain): {len(training_table):,}")
    print(f"Positive account/days: {int(training_table['Is Laundering'].sum()):,}")

    train_meta_learner(training_table, label_col="Is Laundering",
                       neg_sample_rate=args.neg_sample_rate)

    print("\n✅ DONE. Retrained on account/day grain, matching meta_learner_node's inference features.")
    print(f"   Trained on window {start_str} -> {end_str} with neg-sample-rate={args.neg_sample_rate}.")
    print(f"   Make sure this window is NOT the one you evaluate the pipeline against.")


if __name__ == "__main__":
    main()