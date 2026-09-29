"""
pick_tiers.py
Derive INVESTIGATION / MEDIUM / TIER thresholds from eval_days.parquet
(written by evaluation_node). Run it on your TUNING window (Oct 17-19),
then report results on a different window (Oct 23-25).

Usage:
    python pick_tiers.py
    python pick_tiers.py --recall 0.60 --medium-recall 0.75 --tier1-precision 0.40 --tier2-precision 0.30
"""
import argparse

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(config.REPO_ROOT / "eval_days.parquet"))
ap.add_argument("--recall", type=float, default=0.60,
                help="recall target that sets INVESTIGATION_THRESHOLD")
ap.add_argument("--medium-recall", type=float, default=0.75,
                help="recall target that sets MEDIUM_RISK_THRESHOLD (watchlist)")
ap.add_argument("--tier1-precision", type=float, default=0.40,
                help="cumulative precision the TIER_1 cutoff must still reach")
ap.add_argument("--tier2-precision", type=float, default=0.30,
                help="cumulative precision the TIER_2 cutoff must still reach")
ap.add_argument("--min-flagged", type=int, default=50,
                help="ignore tiny top slices when checking tier precision")
args = ap.parse_args()

d = pd.read_parquet(args.file)
y = d["is_laundering_day"].astype(int).to_numpy()
s = d["final_risk_score"].fillna(0.0).to_numpy()

order = np.argsort(-s, kind="stable")
ys, ss = y[order], s[order]
tp = np.cumsum(ys)
k = np.arange(1, len(ys) + 1)
prec = tp / k
rec = tp / ys.sum()

print(f"account/days={len(y):,} | positives={int(ys.sum()):,} | base rate={y.mean():.4%}")


def thr_at_recall(target):
    i = min(int(np.searchsorted(rec, target)), len(ss) - 1)
    return float(ss[i])


def thr_at_precision(target):
    idx = np.where(prec >= target)[0]
    idx = idx[idx >= args.min_flagged - 1]
    return float(ss[int(idx.max())]) if len(idx) else None


t_inv = thr_at_recall(args.recall)
t_med = min(thr_at_recall(args.medium_recall), t_inv)
t1 = thr_at_precision(args.tier1_precision)
t2 = thr_at_precision(args.tier2_precision)

# enforce ordering: medium <= investigation <= tier2 <= tier1
t2 = max(t2 if t2 is not None else t_inv, t_inv)
t1 = max(t1 if t1 is not None else t2, t2)

t_inv, t_med, t1, t2 = (round(x, 3) for x in (t_inv, t_med, t1, t2))

print("\nSuggested values (rounded):")
print(f"  INVESTIGATION_THRESHOLD = {t_inv}   (recall target {args.recall:.0%})")
print(f"  MEDIUM_RISK_THRESHOLD   = {t_med}   (recall target {args.medium_recall:.0%})")
print(f"  TIER_1_THRESHOLD        = {t1}   (cumulative precision >= {args.tier1_precision:.0%})")
print(f"  TIER_2_THRESHOLD        = {t2}   (cumulative precision >= {args.tier2_precision:.0%})")

print("\nBand check (precision INSIDE each band):")
bands = [
    ("TIER_1_HIGH", t1, np.inf),
    ("TIER_2_MEDIUM", t2, t1),
    ("TIER_3_LOW", t_inv, t2),
    ("WATCHLIST (medium)", t_med, t_inv),
]
for name, lo, hi in bands:
    m = (s >= lo) & (s < hi)
    n = int(m.sum())
    p = int(y[m].sum())
    print(f"  {name:<20} [{lo:.3f}, {hi if np.isfinite(hi) else 1.0:.3f}) | "
          f"alerts={n:>7} | true positives={p:>5} | precision={(p / n if n else 0):.4f}")

m = s >= t_inv
print(f"\nAll investigated (>= {t_inv}): alerts={int(m.sum())} | "
      f"precision={y[m].sum() / max(int(m.sum()), 1):.4f} | recall={y[m].sum() / ys.sum():.4f}")