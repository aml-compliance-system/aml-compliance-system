"""
meta_learner.py
Meta-learner training and inference.

meta_learner_score       -> raw calibrated score (same scale the pipeline
                            thresholds were set on; unchanged behaviour).
meta_learner_score_prob  -> APPROXIMATE real-world probability, corrected
                            for negative down-sampling at training time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

import config

# --- 1. Configuration ---
META_LEARNER_PATH = config.REPO_ROOT / "models" / "meta_learner_calibrated.pkl"
META_LEARNER_META_PATH = META_LEARNER_PATH.with_suffix(".json")


# --- 2. Feature builder ---
_FEATURE_COLS = ["xgb_prob", "anomaly_score", "behavioral_score"]
_DEVIATION_COLS = [
    "amount_vs_own_mean", "amount_vs_own_median",
    "amount_zscore_vs_own", "velocity_vs_own_avg",
]
_GRAPH_COLS = ["hop1_count", "hop2_count", "bidir_count", "fan_score"]


def _build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in _FEATURE_COLS + _DEVIATION_COLS + _GRAPH_COLS if c in df.columns]
    return df[existing].fillna(0).copy()


# --- 3. Probability correction for negative down-sampling ---
def correct_for_negative_sampling(p, rate: float) -> np.ndarray:
    """Approximate: odds_true = rate * odds_sampled (exact only for
    account/day-level sampling; transaction-level sampling makes it a
    rough guide)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return (rate * p) / (rate * p + (1.0 - p))


def _load_neg_sample_rate() -> float:
    if META_LEARNER_META_PATH.exists():
        try:
            return float(json.loads(META_LEARNER_META_PATH.read_text())["neg_sample_rate"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return 1.0  # no metadata -> no correction


# --- 4. Inference ---
def predict_with_meta_learner(df: pd.DataFrame) -> pd.DataFrame:
    if not META_LEARNER_PATH.exists():
        raise FileNotFoundError(f"Model not found at {META_LEARNER_PATH}. Train it first!")

    model = joblib.load(META_LEARNER_PATH)
    X = _build_meta_features(df)

    df = df.copy()
    raw = model.predict_proba(X)[:, 1]
    df["meta_learner_score"] = raw
    df["meta_learner_score_prob"] = correct_for_negative_sampling(raw, _load_neg_sample_rate())
    return df


# --- 5. Training ---
def train_meta_learner(scored_df: pd.DataFrame, label_col: str = "Is Laundering",
                       neg_sample_rate: float = 1.0):
    print(f"🔧 Training meta-learner on {len(scored_df)} rows...")

    X = _build_meta_features(scored_df)
    y = scored_df[label_col].astype(int).to_numpy()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < 50:
        raise ValueError(f"Only {n_pos} positive rows; too few to train a reliable meta-learner.")

    raw_ratio = n_neg / max(n_pos, 1)
    scale_pos_weight = min(raw_ratio ** 0.7, 40.0)
    print(f"   {n_pos} pos, {n_neg} neg | scale_pos_weight={scale_pos_weight:.1f}")

    base_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=-1,
    )

    print("   Wrapping in CalibratedClassifierCV (isotonic, cv=3)...")
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X, y)

    mean_pred = model.predict_proba(X)[:, 1].mean()
    print(f"   Post-calibration mean prob: {mean_pred:.4f} | "
          f"training base rate (sampled): {n_pos / len(y):.4f}")

    META_LEARNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, META_LEARNER_PATH)
    META_LEARNER_META_PATH.write_text(json.dumps({
        "neg_sample_rate": float(neg_sample_rate),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "features": list(X.columns),
    }, indent=2))
    print(f"✅ Meta-learner saved to {META_LEARNER_PATH}")
    return model