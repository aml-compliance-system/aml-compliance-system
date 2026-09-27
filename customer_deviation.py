"""
Customer-deviation features: measures how much a transaction deviates from
the customer's OWN historical baseline (not peer group).

VECTORIZED REWRITE
-------------------
The original version used `.groupby("Account")[...].transform(lambda x: ...)`
four times. `.transform(lambda ...)` invokes the Python callback ONCE PER
GROUP -- so runtime scales with the NUMBER OF DISTINCT ACCOUNTS, not just row
count. With small/medium data this is invisible; once account count crosses
some threshold, the sheer number of Python-level function calls (each
spinning up a fresh Series, calling .expanding(), .mean(), .shift()) starts
to dominate and can look like a cliff rather than a slope. That's the "fine
at 1M rows, stalls above it" symptom.

This version computes the same features via:
  - cumulative sums (cumsum/cumcount) for expanding mean/std -- true O(n),
    single vectorized pass, no per-group Python calls at all.
  - GroupBy.rolling(...).median() for the median feature -- this IS a
    vectorized Cython path (unlike .transform(lambda: x.expanding().median())),
    bounded to a fixed window so cost doesn't grow with full history length.
  - GroupBy.rank(...) directly (no lambda wrapper) for the percentile feature.

It also flags (see note near the bottom) a pre-existing look-ahead leak in
the percentile feature: ranking a transaction against its ENTIRE account
history includes transactions that happen LATER in time. Left behavior-
compatible with the original for now since fixing it changes what the
model was trained on -- flagged so you can decide whether to retrain.
"""
import numpy as np
import pandas as pd

ROLLING_MEDIAN_WINDOW = 30  # bounded window; expanding median has no cheap vectorized form


def add_customer_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features measuring deviation from customer's own historical baseline.
    Expects df to have: Account, Amount Paid, Transaction_Date (or Timestamp).
    """
    df = df.copy()

    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Timestamp"
    df["_dev_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(["Account", "_dev_date"]).reset_index(drop=True)

    df["_amount_numeric"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["_amount_sq"] = df["_amount_numeric"] ** 2
    amount = df["_amount_numeric"]

    # ------------------------------------------------------------------
    # 1. Expanding mean/std via cumulative sums -- no per-group Python calls
    # ------------------------------------------------------------------
    acct = df.groupby("Account")
    cumsum = acct["_amount_numeric"].cumsum()
    cumsum_sq = acct["_amount_sq"].cumsum()
    cumcount = acct.cumcount() + 1  # count including current row

    # "Prior" (shift-1) aggregates: exclude the current transaction
    prior_sum = cumsum - amount
    prior_sum_sq = cumsum_sq - df["_amount_sq"]
    prior_count = (cumcount - 1).astype("float64")

    customer_amount_mean = prior_sum / prior_count.replace(0, np.nan)
    prior_var = prior_sum_sq / prior_count.replace(0, np.nan) - customer_amount_mean ** 2
    customer_amount_std = np.sqrt(prior_var.clip(lower=0))

    df["amount_vs_own_mean"] = (
        amount / customer_amount_mean.replace(0, np.nan)
    ).fillna(1.0).clip(0, 50).astype("float32")

    df["amount_zscore_vs_own"] = (
        (amount - customer_amount_mean) / customer_amount_std.replace(0, 1)
    ).fillna(0.0).clip(-5, 20).astype("float32")

    # ------------------------------------------------------------------
    # 2. Expanding MEDIAN -- no cumulative-sum shortcut exists for median,
    #    so use a bounded rolling median via GroupBy.rolling (vectorized,
    #    Cython-backed; NOT the same cost profile as transform(lambda)).
    # ------------------------------------------------------------------
    rolling_median = (
        df.groupby("Account")["_amount_numeric"]
        .rolling(window=ROLLING_MEDIAN_WINDOW, min_periods=1)
        .median()
    )
    # shift(1) WITHIN each account group so we don't leak the current row
    # and don't bleed across account boundaries
    rolling_median = rolling_median.groupby(level=0).shift(1).reset_index(level=0, drop=True)

    df["amount_vs_own_median"] = (
        amount / rolling_median.replace(0, np.nan)
    ).fillna(1.0).clip(0, 50).astype("float32")

    # ------------------------------------------------------------------
    # 3. Velocity: tx count today vs customer's average daily count
    #    (same cumsum/cumcount trick, applied at the daily-aggregate level)
    # ------------------------------------------------------------------
    daily_counts = (
        df.groupby(["Account", "_dev_date"]).size().reset_index(name="daily_tx_count")
    )
    daily_counts = daily_counts.sort_values(["Account", "_dev_date"])

    dg = daily_counts.groupby("Account")["daily_tx_count"]
    d_cumsum = dg.cumsum()
    d_cumcount = dg.cumcount() + 1
    prior_daily_sum = d_cumsum - daily_counts["daily_tx_count"]
    prior_daily_count = (d_cumcount - 1).astype("float64")

    daily_counts["customer_avg_daily"] = prior_daily_sum / prior_daily_count.replace(0, np.nan)
    daily_counts["velocity_vs_own_avg"] = (
        daily_counts["daily_tx_count"] / daily_counts["customer_avg_daily"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 50).astype("float32")

    df = df.merge(
        daily_counts[["Account", "_dev_date", "velocity_vs_own_avg"]],
        on=["Account", "_dev_date"],
        how="left",
    )
    df["velocity_vs_own_avg"] = df["velocity_vs_own_avg"].fillna(1.0).astype("float32")

    # ------------------------------------------------------------------
    # 4. First-ever transaction flag (already vectorized in the original)
    # ------------------------------------------------------------------
    df["is_first_transaction"] = (~df.duplicated(subset=["Account"], keep="first")).astype("float32")

    # ------------------------------------------------------------------
    # 5. Amount percentile vs customer's own history
    #
    # NOTE -- LOOK-AHEAD LEAK (pre-existing, not introduced by this rewrite):
    # `.rank(pct=True)` over the whole group ranks each transaction against
    # ALL of that account's transactions, including ones that happen LATER
    # in time. That means the feature value for a transaction on day 1 can
    # depend on a transaction on day 30. Left as-is here for behavioral
    # compatibility with whatever model was trained on it, but this should
    # be fixed (rank only against prior transactions) before your next
    # retrain, or precision/recall you measure in training won't hold up
    # in true production scoring, where the future obviously isn't visible.
    # ------------------------------------------------------------------
    df["amount_percentile_vs_own"] = (
        df.groupby("Account")["_amount_numeric"].rank(pct=True, method="average")
    ).fillna(0.5).clip(0, 1).astype("float32")

    df = df.drop(columns=["_dev_date", "_amount_numeric", "_amount_sq"])

    print(
        f"Customer deviation features added | "
        f"mean(amount_vs_own_mean)={df['amount_vs_own_mean'].mean():.2f} | "
        f"mean(amount_zscore_vs_own)={df['amount_zscore_vs_own'].mean():.2f}"
    )

    return df