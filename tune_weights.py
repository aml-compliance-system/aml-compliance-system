import itertools
import numpy as np
import pandas as pd
import config

df = pd.read_parquet(config.REPO_ROOT / "scores_for_tuning.parquet")
g = df.groupby(["_graph_date", "Account"]).agg(
    xgb=("xgb_prob", "max"), ae=("anomaly_score", "max"),
    beh=("behavioral_score", "max"), y=("is_laundering", "max")).reset_index()


def base_risk(x, a, e, b):
    wm = a * x.xgb + e * x.ae + b * x.beh
    mx = x[["xgb", "ae", "beh"]].max(axis=1)
    two = (x[["xgb", "ae", "beh"]] > 0.3).sum(axis=1) >= 2
    return np.where(two, np.maximum(wm, mx * 0.85), wm)


def evaluate(a, e, b, thr):
    flag = base_risk(g, a, e, b) >= thr
    tp = int((flag & (g.y == 1)).sum())
    p = tp / max(int(flag.sum()), 1)
    r = tp / g.y.sum()
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return (round(a, 2), round(e, 2), round(b, 2), thr, int(flag.sum()), p, r, f1)


cols = ["xgb_w", "ae_w", "beh_w", "thr", "flagged", "precision", "recall", "f1"]
rows = []
steps = np.arange(0, 1.01, 0.1)
for a, e in itertools.product(steps, steps):
    b = round(1 - a - e, 2)
    if b < 0:
        continue
    for thr in [0.30, 0.38, 0.45, 0.55]:
        rows.append(evaluate(a, e, b, thr))

res = pd.DataFrame(rows, columns=cols)
print("Top 15 by F1:")
print(res.sort_values("f1", ascending=False).head(15).to_string(index=False))

print("\nCurrent weights (0.15 / 0.65 / 0.20):")
cur = pd.DataFrame(
    [evaluate(0.15, 0.65, 0.20, t) for t in [0.30, 0.38, 0.45, 0.55]], columns=cols
)
print(cur.to_string(index=False))