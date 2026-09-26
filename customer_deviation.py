"""
Customer-deviation features: measures how much a transaction deviates from
the customer's OWN historical baseline (not peer group).

These are powerful because money launderers often break their own patterns,
not just peer patterns.
"""
import numpy as np
import pandas as pd


def add_customer_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features measuring deviation from customer's own historical baseline.
    Expects df to have: Account, Amount Paid, Transaction_Date (or Timestamp),
                        outgoing_count_24h, outgoing_count_72h.
    """
    df = df.copy()

    # Determine date column
    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Timestamp"
    df["_dev_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(["Account", "_dev_date"])

    amount = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # === 1. Amount deviation from customer's 30-day rolling mean ===
    # Use expanding mean (all prior transactions) as baseline
    customer_amount_mean = df.groupby("Account")["Amount Paid"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").expanding().mean().shift(1)
    )
    customer_amount_median = df.groupby("Account")["Amount Paid"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").expanding().median().shift(1)
    )
    customer_amount_std = df.groupby("Account")["Amount Paid"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").expanding().std().shift(1)
    )

    df["amount_vs_own_mean"] = (
        amount / customer_amount_mean.replace(0, np.nan)
    ).fillna(1.0).clip(0, 50).astype("float32")

    df["amount_vs_own_median"] = (
        amount / customer_amount_median.replace(0, np.nan)
    ).fillna(1.0).clip(0, 50).astype("float32")

    # Z-score vs customer's own history
    df["amount_zscore_vs_own"] = (
        (amount - customer_amount_mean) / customer_amount_std.replace(0, 1)
    ).fillna(0.0).clip(-5, 20).astype("float32")

    # === 2. Velocity: tx count today vs customer's average daily count ===
    # Count transactions per account per day
    daily_counts = df.groupby(["Account", "_dev_date"]).size().reset_index(name="daily_tx_count")

    # Customer's historical average daily tx count (expanding mean)
    daily_counts = daily_counts.sort_values(["Account", "_dev_date"])
    daily_counts["customer_avg_daily"] = daily_counts.groupby("Account")["daily_tx_count"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    daily_counts["velocity_vs_own_avg"] = (
        daily_counts["daily_tx_count"] / daily_counts["customer_avg_daily"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 50).astype("float32")

    df = df.merge(
        daily_counts[["Account", "_dev_date", "velocity_vs_own_avg"]],
        on=["Account", "_dev_date"],
        how="left",
    )
    df["velocity_vs_own_avg"] = df["velocity_vs_own_avg"].fillna(1.0).astype("float32")

    # === 3. First-ever transaction flag (new accounts are higher risk) ===
    df["is_first_transaction"] = (
        ~df.duplicated(subset=["Account"], keep="first")
    ).astype("float32")

    # === 4. Amount percentile vs customer's own history ===
    # What percentile does this amount fall into for this customer?
    def customer_percentile(group):
        amounts = pd.to_numeric(group, errors="coerce")
        ranks = amounts.rank(pct=True)
        return ranks

    df["amount_percentile_vs_own"] = (
        df.groupby("Account")["Amount Paid"].transform(customer_percentile)
    ).fillna(0.5).clip(0, 1).astype("float32")

    # Cleanup
    df = df.drop(columns=["_dev_date"])

    print(
        f"Customer deviation features added | "
        f"mean(amount_vs_own_mean)={df['amount_vs_own_mean'].mean():.2f} | "
        f"mean(amount_zscore_vs_own)={df['amount_zscore_vs_own'].mean():.2f}"
    )

    return df