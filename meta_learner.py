"""
meta_learner.py
Complete, self-contained module for meta-learner training and inference.
"""
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV

# --- 1. Configuration ---
META_LEARNER_PATH = Path("models/meta_learner_calibrated.pkl")

# --- 2. Feature Builder ---
def _build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    # Core model scores
    feature_cols = ["xgb_prob", "anomaly_score", "behavioral_score"]
    
    # Customer deviation features
    deviation_cols = [
        "amount_vs_own_mean", "amount_vs_own_median", 
        "amount_zscore_vs_own", "velocity_vs_own_avg"
    ]
    
    # FIX: Match the exact column names output by graph_context_node
    graph_cols = [
        "hop1_count", "hop2_count", 
        "bidir_count", "fan_score"
    ]
    
    # Safely filter to only columns that actually exist
    existing_cols = [col for col in feature_cols + deviation_cols + graph_cols if col in df.columns]
    
    X = df[existing_cols].fillna(0).copy()
    return X

# --- 3. Inference Function (Used by nodes.py) ---
def predict_with_meta_learner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the calibrated model and appends the meta_learner_score to the dataframe.
    """
    if not META_LEARNER_PATH.exists():
        raise FileNotFoundError(f"Model not found at {META_LEARNER_PATH}. Train it first!")
    
    model = joblib.load(META_LEARNER_PATH)
    X = _build_meta_features(df)
    
    # Predict probabilities and assign to the positive class (index 1)
    df = df.copy()
    df["meta_learner_score"] = model.predict_proba(X)[:, 1] 
    return df

# --- 4. Training Function (The Fixed Patch) ---
def train_meta_learner(scored_df: pd.DataFrame, label_col: str = "Is Laundering"):
    """
    Train the meta-learner on labeled, scored data, with probability
    calibration so predict_proba actually reflects real probabilities.
    """
    print(f"🔧 Training meta-learner on {len(scored_df)} rows...")

    X = _build_meta_features(scored_df)
    y = scored_df[label_col].astype(int).to_numpy()

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    raw_ratio = n_neg / max(n_pos, 1)


    scale_pos_weight = min(raw_ratio ** 0.7, 40.0)
    print(f"   Class distribution: {n_pos} pos, {n_neg} neg | scale_pos_weight={scale_pos_weight:.1f} (Balanced Mode)")

    base_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="aucpr", tree_method="hist", random_state=42, n_jobs=-1,
    )

    # FIX 2: calibrate predict_proba using isotonic regression
    print("   Wrapping in CalibratedClassifierCV (isotonic, cv=3)...")
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X, y)

    mean_pred = model.predict_proba(X)[:, 1].mean()
    true_base_rate = n_pos / len(y)
    print(f"   ✅ Post-calibration mean prob: {mean_pred:.4f} (True base rate: {true_base_rate:.4f})")
    
    if abs(mean_pred - true_base_rate) > 0.01:
        print("   ⚠️  WARNING: Calibration gap is > 1%. Check data leakage or try method='sigmoid'.")

    # Save
    META_LEARNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, META_LEARNER_PATH)
    print(f"✅ Meta-learner saved to {META_LEARNER_PATH}")

    return model