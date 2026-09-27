"""
Threshold / Precision-Recall analysis for the AML risk pipeline.

Answers the question the hand-picked constants (INVESTIGATION_THRESHOLD=0.28,
SEED_THRESHOLD, etc.) were never actually checked against: at each possible
threshold on final_risk_score, how many accounts get flagged, and what
precision/recall do you actually get?

INPUTS (you produce these once from your existing pipeline run):
  1. A parquet/csv of the labeled data used for evaluation, with at least:
       Account, Is Laundering, and a date column (Transaction_Date or Timestamp)
     -> this is just val_df / df_test, or your held-out production sample,
        the same one you pass into evaluation_node today.
  2. A parquet/csv of the CONTINUOUS account-level scores -- state["risk_scores"]
     from risk_agent_node (or account_results straight out of the graph/
     meta-learner node), with at least:
       graph_date, Account, final_risk_score
     IMPORTANT: use the continuous score (before the hard-rule filters turn
     it into a 0/1 investigation_flag). The whole point is to sweep the
     threshold ourselves instead of trusting a single fixed cutoff.

HOW TO PRODUCE INPUT #2 FROM YOUR EXISTING PIPELINE:
    In your pipeline code, right after risk_agent_node runs (or even right
    after graph_context_node / meta_learner_node), add:
        state["risk_scores"].to_parquet("risk_scores_for_threshold_analysis.parquet")

OUTPUT:
  - threshold_table.csv       : sweep of thresholds -> queue size, precision,
                                 recall, f1, TP/FP/FN/TN, at every 0.01 step
  - pr_curve.png               : precision-recall curve, with the current
                                 hard-coded threshold marked for comparison
  - printed summary            : recommended thresholds for a few realistic
                                 "how many cases can compliance actually work"
                                 queue sizes

FIX FOR YOUR NODE CODE:
    You added `state["risk_scores"].to_parquet(...)` inside risk_agent_node,
    but at that point in the function `state` is still the INPUT to the node
    -- "risk_scores" only gets added to state AFTER the node returns (the
    pipeline merges the returned dict back in for the next node to see).
    Use the local variable instead, right before the return statement:

        results.to_parquet("risk_scores_for_threshold_analysis.parquet")
        return {
            "risk_scores": results,
            "final_insights": flagged_accounts.to_dict(orient="records"),
        }

USAGE:
    python threshold_analysis.py \\
        --labels-path val_df.parquet \\
        --scores-path risk_scores_for_threshold_analysis.parquet \\
        --current-threshold 0.28 \\
        --out-dir threshold_report/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score


def _read_any(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_account_day_labels(labels_df: pd.DataFrame, label_col: str = "Is Laundering") -> pd.DataFrame:
    """
    Ground truth for final_risk_score is at ACCOUNT/DAY grain, but raw labels
    are at TRANSACTION grain. An account/day is a true positive if ANY
    transaction for that account, that day, is laundering.
    """
    if label_col not in labels_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in labels data.")

    date_col = "Transaction_Date" if "Transaction_Date" in labels_df.columns else "Timestamp"
    if date_col not in labels_df.columns:
        raise ValueError("Labels data needs a 'Transaction_Date' or 'Timestamp' column.")
    if "Account" not in labels_df.columns:
        raise ValueError("Labels data needs an 'Account' column.")

    df = labels_df.copy()
    df["graph_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    account_day_labels = (
        df.groupby(["graph_date", "Account"])[label_col]
        .max()
        .reset_index()
        .rename(columns={label_col: "y_true"})
    )
    account_day_labels["y_true"] = account_day_labels["y_true"].astype(int)
    return account_day_labels


def merge_scores_and_labels(labels_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    account_day_labels = build_account_day_labels(labels_df)

    scores = scores_df.copy()
    if "graph_date" not in scores.columns:
        raise ValueError("Scores data needs a 'graph_date' column (as produced by graph_context_node).")
    scores["graph_date"] = pd.to_datetime(scores["graph_date"], errors="coerce").dt.normalize()
    if "final_risk_score" not in scores.columns:
        raise ValueError("Scores data needs a 'final_risk_score' column.")

    merged = account_day_labels.merge(
        scores[["graph_date", "Account", "final_risk_score"]],
        on=["graph_date", "Account"],
        how="left",
    )
    # Accounts that never got a score (e.g. no transactions that day) -> score 0
    merged["final_risk_score"] = merged["final_risk_score"].fillna(0.0)

    n_missing = merged["final_risk_score"].isna().sum()
    if n_missing:
        print(f"⚠️  {n_missing} account/day rows had no matching score.", file=sys.stderr)

    return merged


def threshold_sweep_table(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    n_total = len(y_true)
    rows = []
    for t in thresholds:
        pred = y_score >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        tn = int(np.sum(~pred & (y_true == 0)))
        queue_size = tp + fp
        precision = tp / queue_size if queue_size > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision and recall and (precision + recall) > 0)
            else float("nan")
        )
        rows.append(
            {
                "threshold": round(float(t), 4),
                "queue_size": queue_size,
                "queue_pct": round(100 * queue_size / n_total, 4) if n_total else float("nan"),
                "precision": round(precision, 6) if precision == precision else precision,
                "recall": round(recall, 6) if recall == recall else recall,
                "f1": round(f1, 6) if f1 == f1 else f1,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )
    return pd.DataFrame(rows)


def threshold_for_target_queue_pct(sweep_df: pd.DataFrame, target_pct: float) -> dict:
    """Find the threshold whose resulting queue_pct is closest to target_pct."""
    idx = (sweep_df["queue_pct"] - target_pct).abs().idxmin()
    return sweep_df.loc[idx].to_dict()


def main():
    parser = argparse.ArgumentParser(description="Precision-recall / threshold analysis for risk_scores output.")
    parser.add_argument("--labels-path", required=True, help="Parquet/CSV with Account, Is Laundering, date col")
    parser.add_argument("--scores-path", required=True, help="Parquet/CSV with graph_date, Account, final_risk_score")
    parser.add_argument("--label-col", default="Is Laundering")
    parser.add_argument("--current-threshold", type=float, default=None,
                         help="Your existing hard-coded threshold (e.g. 0.28), marked on the PR curve for comparison")
    parser.add_argument("--out-dir", default="threshold_report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading labels...")
    labels_df = _read_any(args.labels_path)
    print("Loading scores...")
    scores_df = _read_any(args.scores_path)

    print("Merging to account/day grain...")
    merged = merge_scores_and_labels(labels_df, scores_df)

    y_true = merged["y_true"].to_numpy()
    y_score = merged["final_risk_score"].to_numpy()

    n_total = len(y_true)
    n_positive = int(y_true.sum())
    print(f"Account/day rows: {n_total:,} | positives: {n_positive:,} ({100*n_positive/n_total:.4f}%)")

    if n_positive == 0:
        print("❌ No positive labels found -- cannot build a PR curve on this slice.", file=sys.stderr)
        sys.exit(1)

    # --- Real PR curve (every distinct score value as a threshold) ---
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"Average Precision (area under PR curve): {ap:.6f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        plt.plot(recall_curve, precision_curve, label=f"Risk Agent (AP={ap:.4f})")
        if args.current_threshold is not None:
            # Find the point on the curve nearest the current threshold
            close_idx = int(np.argmin(np.abs(pr_thresholds - args.current_threshold)))
            plt.scatter(
                recall_curve[close_idx], precision_curve[close_idx],
                color="red", zorder=5,
                label=f"current threshold={args.current_threshold}",
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve — final_risk_score vs Is Laundering")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "pr_curve.png"
        plt.savefig(plot_path, dpi=150)
        print(f"✅ Saved PR curve: {plot_path}")
    except ImportError:
        print("⚠️  matplotlib not installed -- skipping pr_curve.png (pip install matplotlib)", file=sys.stderr)

    # --- Fine-grained threshold sweep table (0.00 to 1.00 step 0.01) ---
    fine_thresholds = np.round(np.arange(0.0, 1.001, 0.01), 2)
    sweep_df = threshold_sweep_table(y_true, y_score, fine_thresholds)
    table_path = out_dir / "threshold_table.csv"
    sweep_df.to_csv(table_path, index=False)
    print(f"✅ Saved full threshold sweep: {table_path}")

    # --- Recommendations for realistic operational queue sizes ---
    print("\n================ THRESHOLD RECOMMENDATIONS ================")
    print("(pick based on how many account/day cases your team can actually review)")
    for target_pct in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        row = threshold_for_target_queue_pct(sweep_df, target_pct)
        print(
            f"  ~{target_pct:>4}% of accounts flagged -> threshold={row['threshold']:.2f} | "
            f"queue={int(row['queue_size']):,} | precision={row['precision']:.4f} | recall={row['recall']:.4f}"
        )

    if args.current_threshold is not None:
        current_row = sweep_df.iloc[(sweep_df["threshold"] - args.current_threshold).abs().argsort()[:1]]
        print("\n---- Your CURRENT threshold, for comparison ----")
        print(current_row.to_string(index=False))

    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()