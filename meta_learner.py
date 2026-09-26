"""
Stacking Meta-Learner: Non-linear combination of base model outputs + graph features.
Replaces the simple weighted average with a trained XGBoost that learns interactions.
"""
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

import config

META_LEARNER_PATH = Path(
    os.environ.get("FINDETECT_META_LEARNER_PATH", config.REPO_ROOT / "models" / "meta_learner_v1.joblib")
)

# Features the meta-learner uses (must match training)
META_FEATURE_COLS = [
    # Base model scores
    "xgb_prob", "anomaly_score", "behavioral_score",
    # Graph features
    "base_risk", "graph_penalty", "hop1_count", "hop2_count", "hop3_count",
    "bidir_count", "fan_score", "daily_degree",
    # Customer deviation features (added by customer_deviation_node)
    "amount_vs_own_mean", "amount_vs_own_median", "amount_zscore_vs_own",
    "velocity_vs_own_avg",
    # Interaction features (computed on-the-fly)
]


def _build_meta_features(df: pd.DataFrame) -> pd.DataFrame:

    """Build the feature matrix for the meta-learner from base scores + graph features."""
    X = pd.DataFrame(index=df.index)

    # Base model scores
    for col in ["xgb_prob", "anomaly_score", "behavioral_score"]:
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Graph features
    for col in ["base_risk", "graph_penalty", "hop1_count", "hop2_count", "hop3_count",
                "bidir_count", "fan_score", "daily_degree"]:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            X[col] = 0.0

    # Customer deviation features
    for col in ["amount_vs_own_mean", "amount_vs_own_median", "amount_zscore_vs_own", "velocity_vs_own_avg"]:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            X[col] = 0.0

    # === INTERACTION FEATURES (this is where the magic happens) ===
    # When multiple models agree, risk is multiplicative, not additive
    X["ae_x_beh"] = X["anomaly_score"] * X["behavioral_score"]
    X["ae_x_xgb"] = X["anomaly_score"] * X["xgb_prob"]
    X["beh_x_xgb"] = X["behavioral_score"] * X["xgb_prob"]
    X["all_three_high"] = (
        (X["anomaly_score"] > 0.80) &
        (X["behavioral_score"] > 0.40) &
        (X["xgb_prob"] > 0.05)
    ).astype(float)
    X["two_high"] = (
        ((X["anomaly_score"] > 0.80).astype(int) +
         (X["behavioral_score"] > 0.40).astype(int) +
         (X["xgb_prob"] > 0.05).astype(int)) >= 2
    ).astype(float)

    X["max_score"] = X[["xgb_prob", "anomaly_score", "behavioral_score"]].max(axis=1)
    X["min_score"] = X[["xgb_prob", "anomaly_score", "behavioral_score"]].min(axis=1)
    X["score_spread"] = X["max_score"] - X["min_score"]
    X["mean_score"] = X[["xgb_prob", "anomaly_score", "behavioral_score"]].mean(axis=1)

    # Graph × model interactions
    X["graph_x_ae"] = X["graph_penalty"] * X["anomaly_score"]
    X["hop1_x_base"] = (X["hop1_count"] > 0).astype(float) * X["base_risk"]
    X["bidir_x_ae"] = (X["bidir_count"] > 0).astype(float) * X["anomaly_score"]

    # Customer deviation × model interactions
    X["deviation_x_ae"] = X["amount_zscore_vs_own"].clip(-3, 10) * X["anomaly_score"]

    # Sanitize
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def predict_with_meta_learner(df: pd.DataFrame) -> np.ndarray:
    """
    Use trained meta-learner to produce final risk scores.
    Returns array of probabilities in [0, 1].
    """
    if not META_LEARNER_PATH.exists():
        raise FileNotFoundError(
            f"Meta-learner model not found at {META_LEARNER_PATH}. "
            f"Run `python train_meta_learner.py` first."
        )

    model = joblib.load(META_LEARNER_PATH)
    X = _build_meta_features(df)

    # Ensure feature order matches training
    expected_features = model.feature_names_in_
    missing = [f for f in expected_features if f not in X.columns]
    if missing:
        raise ValueError(f"Meta-learner missing features at inference: {missing}")
    X = X[expected_features]

    probabilities = model.predict_proba(X)[:, 1].astype(np.float32)
    return np.clip(probabilities, 0.0, 1.0)


def train_meta_learner(scored_df: pd.DataFrame, label_col: str = "Is Laundering"):
    """
    Train the meta-learner on labeled, scored data.
    Call this ONCE via `train_meta_learner.py`.
    """
    print(f"🔧 Training meta-learner on {len(scored_df)} rows...")

    X = _build_meta_features(scored_df)
    y = scored_df[label_col].astype(int).to_numpy()

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"   Class distribution: {n_pos} positive, {n_neg} negative, scale_pos_weight={scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="aucpr",          # PR-AUC is better for imbalanced data
        use_label_encoder=False,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y, verbose=50)

    # Save
    META_LEARNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, META_LEARNER_PATH)
    print(f"✅ Meta-learner saved to {META_LEARNER_PATH}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n📊 Top 15 Meta-Learner Features:")
    for feat, imp in importance.head(15).items():
        print(f"   {feat:30s} {imp:.4f}")

    return model