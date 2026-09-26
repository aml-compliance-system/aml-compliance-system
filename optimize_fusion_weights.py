
import numpy as np
import pandas as pd


def run_fusion(df: pd.DataFrame, xgb_w: float, anomaly_w: float, behavioral_w: float,
               seed_threshold: float, direct_penalty: float, indirect_penalty: float,
               max_normal_degree: int = 50, corrob_min: float = 0.3, corrob_count: int = 2) -> pd.DataFrame:
    """Recompute base_risk + graph propagation for one candidate weight set."""
    df = df.copy()
    weighted_mean = xgb_w * df["xgb_prob"] + anomaly_w * df["anomaly_score"] + behavioral_w * df["behavioral_score"]
    signals = df[["xgb_prob", "anomaly_score", "behavioral_score"]]
    max_single = signals.max(axis=1)
    corroborated = (signals > corrob_min).sum(axis=1) >= corrob_count
    df["base_risk"] = np.where(corroborated, np.maximum(weighted_mean, max_single * 0.85), weighted_mean)

    sender_base_risk = df.groupby(["_graph_date", "Account"])["base_risk"].max().reset_index()

    sender_accounts = df[["_graph_date", "Account"]].dropna().copy()
    receiver_accounts = df[["_graph_date", "Account_duplicated_0"]].dropna().rename(columns={"Account_duplicated_0": "Account"})
    account_results = pd.concat([sender_accounts, receiver_accounts], ignore_index=True).drop_duplicates()
    account_results = account_results.merge(sender_base_risk, on=["_graph_date", "Account"], how="left")
    account_results["base_risk"] = account_results["base_risk"].fillna(0.0)

    daily_degree = df.groupby(["_graph_date", "Account"])["Account_duplicated_0"].nunique().reset_index(name="daily_degree")
    account_results = account_results.merge(daily_degree, on=["_graph_date", "Account"], how="left")
    account_results["daily_degree"] = account_results["daily_degree"].fillna(1.0)

    account_results["is_high_risk_seed"] = (
        (account_results["base_risk"] > seed_threshold) & (account_results["daily_degree"] <= max_normal_degree)
    )
    seeds = account_results.loc[account_results["is_high_risk_seed"], ["_graph_date", "Account"]].drop_duplicates()
    seeds = seeds.rename(columns={"Account": "Seed_Account", "_graph_date": "graph_date"})

    amount_col = "Amount Paid" if "Amount Paid" in df.columns else "Amount Received"
    df["_clean_amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    edges = df[df["Account"] != df["Account_duplicated_0"]][["_graph_date", "Account", "Account_duplicated_0", "_clean_amount"]].copy()
    edges.columns = ["graph_date", "sender", "receiver", "amount"]

    seed_send = edges.merge(seeds, left_on=["graph_date", "sender"], right_on=["graph_date", "Seed_Account"])
    hop1_receivers = seed_send[["graph_date", "receiver", "Seed_Account", "amount"]].rename(columns={"receiver": "Account"})
    seed_recv = edges.merge(seeds, left_on=["graph_date", "receiver"], right_on=["graph_date", "Seed_Account"])
    hop1_senders = seed_recv[["graph_date", "sender", "Seed_Account", "amount"]].rename(columns={"sender": "Account"})
    hop1_all = pd.concat([hop1_receivers, hop1_senders])
    hop1_stats = hop1_all.groupby(["graph_date", "Account"]).agg(
        hop1_count=("Seed_Account", "nunique"), total_volume=("amount", "sum")
    ).reset_index()
    hop1_stats["volume_factor"] = (np.log1p(hop1_stats["total_volume"]) / 10.0).clip(lower=0.5, upper=1.5)

    hop1_unique = hop1_all[["graph_date", "Account"]].drop_duplicates().rename(columns={"Account": "Hop1_Account"})
    hop2_recv = edges.merge(hop1_unique, left_on=["graph_date", "sender"], right_on=["graph_date", "Hop1_Account"])
    hop2_receivers = hop2_recv[["graph_date", "receiver", "Hop1_Account"]].rename(columns={"receiver": "Account"})
    hop2_send = edges.merge(hop1_unique, left_on=["graph_date", "receiver"], right_on=["graph_date", "Hop1_Account"])
    hop2_senders = hop2_send[["graph_date", "sender", "Hop1_Account"]].rename(columns={"sender": "Account"})
    hop2_all = pd.concat([hop2_receivers, hop2_senders])
    hop2_counts = hop2_all.groupby(["graph_date", "Account"])["Hop1_Account"].nunique().reset_index(name="hop2_count")

    account_results = account_results.rename(columns={"_graph_date": "graph_date"})
    account_results = account_results.merge(hop1_stats, on=["graph_date", "Account"], how="left")
    account_results["hop1_count"] = account_results["hop1_count"].fillna(0.0)
    account_results["volume_factor"] = account_results["volume_factor"].fillna(1.0)
    account_results = account_results.merge(hop2_counts, on=["graph_date", "Account"], how="left")
    account_results["hop2_count"] = account_results["hop2_count"].fillna(0.0)
    account_results.loc[account_results["hop1_count"] > 0, "hop2_count"] = 0.0

    account_results["graph_penalty"] = (
        account_results["hop1_count"] * direct_penalty * account_results["volume_factor"]
        + account_results["hop2_count"] * indirect_penalty
    ).clip(upper=0.55)
    account_results["final_risk_score"] = (account_results["base_risk"] + account_results["graph_penalty"]).clip(0.0, 1.0)
    return account_results


def evaluate_candidate(df, account_results, target_min=0.85, target_max=0.90, objective="precision"):
    """Merge account_results back to transaction level; find the best in-band threshold.

    objective="precision" (old default): among thresholds landing recall in
        [target_min, target_max], pick the one with the HIGHEST PRECISION.
        This tends to land recall near target_min (precision falls as recall rises).
    objective="recall": among thresholds landing recall in [target_min, target_max],
        pick the one with the HIGHEST RECALL (i.e. push toward target_max, more TP,
        more FP too). Use this when you explicitly want more TP within the band.
    """
    merged = df.merge(
        account_results[["graph_date", "Account", "final_risk_score"]],
        left_on=["_graph_date", "Account"], right_on=["graph_date", "Account"], how="left",
    )
    scores = merged["final_risk_score"].fillna(0.0).to_numpy()
    y_true = merged["Is Laundering"].astype(int).to_numpy()

    best = None
    for t in np.arange(0.02, 0.96, 0.005):
        preds = scores >= t
        tp = int((preds & (y_true == 1)).sum())
        fp = int((preds & (y_true == 0)).sum())
        fn = int((~preds & (y_true == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if not (target_min <= recall <= target_max):
            continue
        candidate = {"threshold": round(float(t), 4), "precision": precision, "recall": recall, "TP": tp, "FP": fp, "FN": fn}
        if best is None:
            best = candidate
        elif objective == "recall" and recall > best["recall"]:
            best = candidate
        elif objective == "precision" and precision > best["precision"]:
            best = candidate
    return best


def optimize(df, n_iter=60, target_min=0.85, target_max=0.90, seed=42, objective="precision"):
    """Random search over fusion weights + graph thresholds.

    objective: passed through to evaluate_candidate AND used to rank configs
        against each other. "recall" biases both the per-config threshold pick
        and the overall winner toward more TP (higher recall) within the band,
        at the cost of more FP. "precision" (default) biases toward fewer FP,
        landing recall near target_min.
    """
    rng = np.random.default_rng(seed)
    best_overall = None
    best_config = None

    for i in range(n_iter):
        xgb_w, anomaly_w, behavioral_w = rng.dirichlet([1, 1, 1])
        seed_threshold = rng.uniform(0.25, 0.55)
        direct_penalty = rng.uniform(0.10, 0.35)
        indirect_penalty = rng.uniform(0.02, 0.10)

        account_results = run_fusion(df, xgb_w, anomaly_w, behavioral_w, seed_threshold, direct_penalty, indirect_penalty)
        result = evaluate_candidate(df, account_results, target_min, target_max, objective=objective)
        if result is None:
            continue

        print(f"[{i+1}/{n_iter}] xgb={xgb_w:.3f} ae={anomaly_w:.3f} beh={behavioral_w:.3f} "
              f"seed_thr={seed_threshold:.3f} dp={direct_penalty:.3f} ip={indirect_penalty:.3f} "
              f"-> precision={result['precision']:.4f} recall={result['recall']:.4f} "
              f"TP={result['TP']} FP={result['FP']}")

        is_better = (
            (objective == "recall" and (best_overall is None or result["recall"] > best_overall["recall"]))
            or (objective == "precision" and (best_overall is None or result["precision"] > best_overall["precision"]))
        )
        if is_better:
            best_overall = result
            best_config = {
                "FINDETECT_XGB_WEIGHT": float(xgb_w),
                "FINDETECT_ANOMALY_WEIGHT": float(anomaly_w),
                "FINDETECT_BEHAVIORAL_WEIGHT": float(behavioral_w),
                "FINDETECT_SEED_THRESHOLD": float(seed_threshold),
                "FINDETECT_GRAPH_PENALTY": float(direct_penalty),
                "FINDETECT_INDIRECT_PENALTY": float(indirect_penalty),
                "FINDETECT_INVESTIGATION_THRESHOLD": result["threshold"],
            }

    print("\n=== BEST CONFIG FOUND (set these as env vars / config.py defaults) ===")
    print(best_config)
    print(best_overall)
    return best_config, best_overall


if __name__ == "__main__":
    raise SystemExit(
        "Import optimize(df) from your evaluation_node instead of running this "
        "standalone -- it needs the live scored dataframe from your pipeline state."
    )
