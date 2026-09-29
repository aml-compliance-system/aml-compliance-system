import glob, pandas as pd
f = sorted(glob.glob("outputs/FINDETECT_Pipeline_Output/*/risk_scores.parquet"))[-1]
r = pd.read_parquet(f)
for t in [0.45, 0.55, 0.65, 0.75, 0.85]:
    print(t, f"{(r.base_risk > t).mean():.2%} of account/days")