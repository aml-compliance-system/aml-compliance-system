from cmath import log
import os
import uuid
 
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import traceback

import config
from LLM import generate_report
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
 

def preprocessing_node(state: dict) -> dict:
    print("--- LOADING FINAL FEATURE DATASET ---")

    # Set the fallback to the EXACT path from your screenshot
    parquet_path = state.get("final_parquet_path", str(config.DEFAULT_PARQUET_PATH))
    absolute_path = os.path.abspath(parquet_path)
    print(f"🔍 DEBUG: Looking for file at: {absolute_path}")

    if not os.path.exists(absolute_path):
        error_msg = f"FILE NOT FOUND: The dataset does not exist at {absolute_path}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)

    try:
        df = pd.read_parquet(absolute_path)
        print(f"✅ Final dataset loaded: {len(df)} transactions.")
        print(f"✅ Columns loaded: {len(df.columns)}")
        return {"final_df": df}
        
    except Exception as e:
        print(f"❌ ERROR reading Parquet file: {str(e)}")
        raise e

# ---------------------------------------------------------------
# 2. XGBoost Model Node
# ---------------------------------------------------------------
_XGB_FEATURE_COLS = [
    "Amount Received", "Amount Paid",
    "sender_out_degree", "sender_in_degree", "sender_unique_counterparties",
    "receiver_out_degree", "receiver_in_degree", "receiver_unique_counterparties",
    "sender_receiver_frequency", "bidirectional_relationship",
    "sender_outgoing_count_so_far_wk", "sender_unique_receivers_so_far_wk",
    "outgoing_count_24h", "outgoing_sum_24h", "outgoing_count_72h", "outgoing_sum_72h",
    "sender_total_sent_prev_week", "sender_avg_sent_prev_week", "sender_max_sent_prev_week",
    "sender_outgoing_count_prev_week", "receiver_total_received_prev_week",
    "receiver_avg_received_prev_week", "receiver_max_received_prev_week",
    "receiver_incoming_count_prev_week", "amount_vs_prev_max_diff",
    "amount_vs_prev_max_ratio", "transaction_frequency_diff", "transaction_frequency_ratio",
    "hours_since_last_incoming", "outgoing_to_incoming_ratio",
    "hour_of_day", "currency_mismatch", "same_bank", "payment_format_hash",
]

def _derive_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the small set of derived columns the trained models expect, if absent."""
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = (
            pd.to_datetime(df["Timestamp"]).dt.hour if "Timestamp" in df.columns else 0
        )
 
    if "currency_mismatch" not in df.columns:
        df["currency_mismatch"] = (
            (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
            if {"Receiving Currency", "Payment Currency"}.issubset(df.columns)
            else 0
        )
 
    if "same_bank" not in df.columns:
        df["same_bank"] = (
            (df["From Bank"] == df["To Bank"]).astype(int)
            if {"From Bank", "To Bank"}.issubset(df.columns)
            else 0
        )
 
    if "payment_format_hash" not in df.columns:
        df["payment_format_hash"] = (
            df["Payment Format"].astype("category").cat.codes
            if "Payment Format" in df.columns
            else 0
        )
 
    return df
 
def run_xgboost_model(state: dict) -> dict:
    print("Running optimized XGBoost model with sensitivity tuning")
    try:
        df = _derive_missing_features(state["final_df"].copy())
 
        if not config.XGBOOST_MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found: {config.XGBOOST_MODEL_PATH}")
 
        model = xgb.Booster()
        model.load_model(str(config.XGBOOST_MODEL_PATH))
 
        X = df[_XGB_FEATURE_COLS].copy()
        dmatrix = xgb.DMatrix(X, feature_names=_XGB_FEATURE_COLS)
 
        raw_probabilities = np.asarray(model.predict(dmatrix), dtype=np.float32)
 
        if len(raw_probabilities) != len(df):
            raise ValueError(f"XGBoost prediction length mismatch: {len(raw_probabilities)} predictions for {len(df)} transactions.")
        if not np.isfinite(raw_probabilities).all():
            raise ValueError("XGBoost produced non-finite probabilities.")

        # --- OPTIMIZATION: Non-linear boosting for imbalanced AML data ---
        # Scale raw probabilities to give a sharper separation between normal and illicit transactions
        # Using a power curve (0.4) lifts suppressed probabilities into actionable decision ranges
        probabilities = np.power(np.clip(raw_probabilities * 3.5, 0.0, 1.0), 0.4)
        probabilities = np.clip(probabilities, 0.0, 1.0)
 
        print(
            "Optimized XGBoost scoring complete | rows=%s | min=%.6f | mean=%.6f | max=%.6f",
            len(probabilities), probabilities.min(), probabilities.mean(), probabilities.max(),
        )
 
        return {"ml_probabilities": {"xgb_prob": probabilities.tolist()}}
 
    except Exception:
        print("XGBoost node failed")
        raise

    
# 3. Anomaly Model Node
def run_anomaly_model(state: dict) -> dict:

    print("--- RUNNING AUTOENCODER ANOMALY DETECTION ---")

    try:
        df = state["final_df"].copy()

        model_path = config.AUTOENCODER_MODEL_DIR / config.AUTOENCODER_MODEL_FILE
        bundle_path = config.AUTOENCODER_MODEL_DIR / config.AUTOENCODER_BUNDLE_FILE
 
        if not model_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found: {model_path}")
        if not bundle_path.exists():
            raise FileNotFoundError(f"Autoencoder preprocessing bundle not found: {bundle_path}")
 
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        bundle = joblib.load(bundle_path)
 
        model_raw_features = bundle["model_raw_features"]
        missing = [c for c in model_raw_features if c not in df.columns]
        if missing:
            raise ValueError(f"Autoencoder missing required Final dataset features: {missing}")
 
        work = df[model_raw_features].copy()
 
        work["no_previous_incoming"] = (
            pd.to_numeric(work["hours_since_last_incoming"], errors="coerce").isna().astype("float32")
        )
 
        amount_ratio = pd.to_numeric(work["amount_vs_prev_max_ratio"], errors="coerce")
        frequency_ratio = pd.to_numeric(work["transaction_frequency_ratio"], errors="coerce")
        work["undefined_history_ratio"] = (
            (amount_ratio.isna() | frequency_ratio.isna()).astype("float32")
        )
 
        continuous_features = bundle["continuous_features"]
        continuous = (
            work[continuous_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
 
        imputed = bundle["continuous_imputer"].transform(continuous)
        continuous_processed = pd.DataFrame(imputed, columns=continuous_features, index=work.index)
 
        clip_lower, clip_upper = bundle["clip_lower"], bundle["clip_upper"]
        for col in continuous_features:
            continuous_processed[col] = continuous_processed[col].clip(
                lower=clip_lower[col], upper=clip_upper[col]
            )
 
        transformed = bundle["quantile_transformer"].transform(continuous_processed[continuous_features])
        transformed_df = pd.DataFrame(transformed, columns=continuous_features, index=work.index)
 
        transformed_df["bidirectional_relationship"] = (
            pd.to_numeric(work["bidirectional_relationship"], errors="coerce")
            .fillna(0.0).clip(0.0, 1.0).astype("float32")
        )
        transformed_df["no_previous_incoming"] = work["no_previous_incoming"].astype("float32")
        transformed_df["undefined_history_ratio"] = work["undefined_history_ratio"].astype("float32")
 
        active_features = bundle["active_model_features"]
        X = transformed_df[active_features].to_numpy(dtype=np.float32)
 
        expected_width = model.input_shape[-1]
        if X.shape[1] != expected_width:
            raise ValueError(f"Autoencoder expected {expected_width} features but received {X.shape[1]}.")
 
        reconstructed = model.predict(X, batch_size=4096, verbose=0)
        feature_errors = np.square(X - reconstructed)
 
        feature_refs = bundle["feature_error_references"]
        feature_percentiles = np.empty_like(feature_errors, dtype=np.float32)
        for j, reference in enumerate(feature_refs):
            reference = np.asarray(reference)
            feature_percentiles[:, j] = (
                np.searchsorted(reference, feature_errors[:, j], side="right") / len(reference)
            )
 
        # Final64 selected aggregation method = mean_all
        transaction_raw_score = feature_percentiles.mean(axis=1)
 
        transaction_reference = np.asarray(bundle["transaction_score_reference"])
        anomaly_scores = (
            np.searchsorted(transaction_reference, transaction_raw_score, side="right")
            / len(transaction_reference)
        ).astype(np.float32)
        print(
            "Autoencoder scoring complete | rows=%s | min=%.6f | mean=%.6f | max=%.6f",
            len(anomaly_scores), anomaly_scores.min(), anomaly_scores.mean(), anomaly_scores.max(),
        )
 
        return {"ml_probabilities": {"anomaly_score": anomaly_scores.tolist()}}
 
    except Exception:
        print("Autoencoder node failed")
        raise
 

def run_behavioral_model(state: dict) -> dict:
    print("Running sharpened behavioral risk model with dynamic MAD scaling")
    df = state["final_df"].copy()
 
    required = [
        "Amount Paid", "amount_vs_prev_max_ratio", "transaction_frequency_ratio",
        "outgoing_count_24h", "outgoing_count_72h",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Behavioral model missing required features: {missing}")
 
    def numeric(col):
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
 
    # -- Customer baseline drift: amount.
    amount_excess = (numeric("amount_vs_prev_max_ratio") - 1.0).clip(lower=0.0)
    amount_drift_score = (
        (1.0 - np.exp(-amount_excess * 1.2))  # Sharper exponential response
        .where(numeric("amount_vs_prev_max_ratio") > 0, 0.0)
        .fillna(0.0).clip(0.0, 1.0)
    )
 
    # -- Customer baseline drift: frequency.
    frequency_excess = (numeric("transaction_frequency_ratio") - 1.0).clip(lower=0.0)
    frequency_drift_score = (
        (1.0 - np.exp(-frequency_excess * 1.2))
        .where(numeric("transaction_frequency_ratio") > 0, 0.0)
        .fillna(0.0).clip(0.0, 1.0)
    )
 
    # -- Sharpened Peer-group comparison (MAD scaling) --
    amount_paid = numeric("Amount Paid").fillna(0.0).clip(lower=0.0)
    log_amount = np.log1p(amount_paid)
    peer_group = (
        df["Payment Format"].fillna("UNKNOWN").astype(str)
        if "Payment Format" in df.columns
        else pd.Series("GLOBAL", index=df.index)
    )
 
    peer_df = pd.DataFrame({"group": peer_group, "value": log_amount})
    group_size = peer_df.groupby("group")["value"].transform("size")
    group_median = peer_df.groupby("group")["value"].transform("median")
    group_mad = (
        (peer_df["value"] - group_median).abs().groupby(peer_df["group"]).transform("median")
    )
 
    global_median = float(np.nanmedian(log_amount))
    global_mad = float(np.nanmedian(np.abs(log_amount - global_median)))
    
    # Tighter scale factor for more sensitive outlier detection
    global_scale = max(1.25 * global_mad, 1e-6)
 
    use_global = (group_size < config.MIN_PEER_GROUP_SIZE) | (group_mad <= 1e-9) | group_mad.isna()
    center = group_median.where(~use_global, global_median)
    scale = (1.25 * group_mad).where(~use_global, global_scale)
 
    robust_z = ((log_amount - center) / scale).clip(lower=0.0)
    
    # Lowered saturation limit (from 5.0 to 4.0) so anomalies saturate quicker
    peer_score = (robust_z / 4.0).clip(0.0, 1.0).fillna(0.0)
 
    # -- Rolling velocity acceleration --
    count_24h = numeric("outgoing_count_24h").fillna(0.0).clip(lower=0.0)
    count_72h = numeric("outgoing_count_72h").fillna(0.0).clip(lower=0.0)
    recent_share = count_24h / count_72h.replace(0, np.nan)
    velocity_score = (
        ((recent_share - (1.0 / 3.0)) / (2.0 / 3.0)).clip(0.0, 1.0).fillna(0.0)
    )
 
    # Base weighted sum calculation
    raw_behavioral = (
        config.AMOUNT_DRIFT_WEIGHT * amount_drift_score
        + config.FREQUENCY_DRIFT_WEIGHT * frequency_drift_score
        + config.PEER_GROUP_WEIGHT * peer_score
        + config.VELOCITY_WEIGHT * velocity_score
    )

    # Compound risk synergy multiplier
    synergy_mask = (amount_drift_score > 0.5) & (velocity_score > 0.5)
    raw_behavioral_adjusted = raw_behavioral.mask(synergy_mask, raw_behavioral * 1.4)

    behavioral_scores = np.clip(
        np.asarray(raw_behavioral_adjusted, dtype=np.float32), 
        0.0, 
        1.0
    )
 
    if len(behavioral_scores) != len(df):
        raise RuntimeError("Behavioral score count does not match transaction count.")
    if not np.isfinite(behavioral_scores).all():
        raise RuntimeError("Behavioral model produced invalid scores.")
 
    print(
        "Sharpened behavioral model scored %s transactions | range=%.4f-%.4f | mean=%.4f",
        len(behavioral_scores), behavioral_scores.min(), behavioral_scores.max(), behavioral_scores.mean(),
    )
 
    return {"ml_probabilities": {"behavioral_score": behavioral_scores.tolist()}}




# 4. Upgraded Graph Context Node (Dynamic Penalties)
def graph_context_node(state: dict) -> dict:
    print("Running super-node shielded multi-hop graph analysis")
 
    df = state["final_df"].copy()
    ml_probs = state.get("ml_probabilities", {})
 
    score_map = {
        "xgb_prob": ml_probs.get("xgb_prob"),
        "anomaly_score": ml_probs.get("anomaly_score"),
        "behavioral_score": ml_probs.get("behavioral_score"),
    }
 
    for score_name, values in score_map.items():
        if values is None:
            raise ValueError(f"Graph node missing model score: {score_name}")
        df[score_name] = (
            pd.to_numeric(pd.Series(values, index=df.index), errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        )
 
    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Timestamp"
    df["_graph_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
 
    # Hybrid Base Risk
    weighted_mean = (
        config.XGB_WEIGHT * df["xgb_prob"]
        + config.ANOMALY_WEIGHT * df["anomaly_score"]
        + config.BEHAVIORAL_WEIGHT * df["behavioral_score"]
    )
    max_single_signal = df[["xgb_prob", "anomaly_score", "behavioral_score"]].max(axis=1)
    df["base_risk"] = np.where(
       (df[["xgb_prob","anomaly_score","behavioral_score"]] > 0.3).sum(axis=1) >= 2,
       np.maximum(weighted_mean, max_single_signal * 0.85),
       weighted_mean,
   )
 
    sender_base_risk = df.groupby(["_graph_date", "Account"], dropna=True)["base_risk"].max().reset_index()
 
    sender_accounts = df[["_graph_date", "Account"]].dropna().copy()
    receiver_accounts = df[["_graph_date", "Account_duplicated_0"]].dropna().rename(columns={"Account_duplicated_0": "Account"})
    account_results = pd.concat([sender_accounts, receiver_accounts], ignore_index=True).drop_duplicates()
    
    account_results = account_results.merge(sender_base_risk, on=["_graph_date", "Account"], how="left")
    account_results["base_risk"] = account_results["base_risk"].fillna(0.0).astype(float)
 
    # --- SUPER-NODE SHIELDING ---
    # Calculate daily unique counterparties per account to block merchants/gateways from becoming seeds
    daily_degree = df.groupby(["_graph_date", "Account"])["Account_duplicated_0"].nunique().reset_index(name="daily_degree")
    account_results = account_results.merge(daily_degree, on=["graph_date" if "graph_date" in account_results.columns else "_graph_date", "Account"], how="left")
    account_results["daily_degree"] = account_results["daily_degree"].fillna(1.0)
    
    # If an account interacts with more than 50 different entities in a single day, it's a merchant/gateway, NOT a launderer. Block it from being a seed.
    MAX_NORMAL_DEGREE = 50
    
    # Phase 2: Identify high-risk seeds with super-node protection
    account_results["is_high_risk_seed"] = (
        (account_results["base_risk"] > config.SEED_THRESHOLD) & 
        (account_results["daily_degree"] <= MAX_NORMAL_DEGREE)
    )
    
    seeds = account_results[account_results["is_high_risk_seed"]][["_graph_date", "Account"]].drop_duplicates()
    seeds = seeds.rename(columns={"Account": "Seed_Account"})
 
    amount_col = "Amount Paid" if "Amount Paid" in df.columns else "Amount Received"
    df["_clean_amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).clip(lower=0.0) if amount_col in df.columns else 1.0

    edges = df[df["Account"] != df["Account_duplicated_0"]][["_graph_date", "Account", "Account_duplicated_0", "_clean_amount"]].copy()
    edges.columns = ["graph_date", "sender", "receiver", "amount"]

    # HOP 1: DIRECT EXPOSURE
    seed_send = edges.merge(seeds, left_on=["graph_date", "sender"], right_on=["_graph_date", "Seed_Account"])
    hop1_receivers = seed_send[["graph_date", "receiver", "Seed_Account", "amount"]].rename(columns={"receiver": "Account"})
    
    seed_recv = edges.merge(seeds, left_on=["graph_date", "receiver"], right_on=["_graph_date", "Seed_Account"])
    hop1_senders = seed_recv[["graph_date", "sender", "Seed_Account", "amount"]].rename(columns={"sender": "Account"})

    hop1_all = pd.concat([hop1_receivers, hop1_senders])
    
    hop1_stats = hop1_all.groupby(["graph_date", "Account"]).agg(
        hop1_count=("Seed_Account", "nunique"),
        total_volume=("amount", "sum")
    ).reset_index()

    hop1_stats["volume_factor"] = np.log1p(hop1_stats["total_volume"]) / 10.0
    hop1_stats["volume_factor"] = hop1_stats["volume_factor"].clip(lower=0.5, upper=1.5)

    # HOP 2: INDIRECT EXPOSURE
    hop1_unique = hop1_all[["graph_date", "Account"]].drop_duplicates().rename(columns={"Account": "Hop1_Account"})
    
    hop2_recv = edges.merge(hop1_unique, left_on=["graph_date", "sender"], right_on=["graph_date", "Hop1_Account"])
    hop2_receivers = hop2_recv[["graph_date", "receiver", "Hop1_Account"]].rename(columns={"receiver": "Account"})
    
    hop2_send = edges.merge(hop1_unique, left_on=["graph_date", "receiver"], right_on=["graph_date", "Hop1_Account"])
    hop2_senders = hop2_send[["graph_date", "sender", "Hop1_Account"]].rename(columns={"sender": "Account"})
    
    hop2_all = pd.concat([hop2_receivers, hop2_senders])
    hop2_counts = hop2_all.groupby(["graph_date", "Account"])["Hop1_Account"].nunique().reset_index(name="hop2_count")

    # MERGE & APPLY PENALTIES
    account_results = account_results.rename(columns={"_graph_date": "graph_date"})
    
    account_results = account_results.merge(hop1_stats, on=["graph_date", "Account"], how="left")
    account_results["hop1_count"] = account_results["hop1_count"].fillna(0.0)
    account_results["volume_factor"] = account_results["volume_factor"].fillna(1.0)
    
    account_results = account_results.merge(hop2_counts, on=["graph_date", "Account"], how="left")
    account_results["hop2_count"] = account_results["hop2_count"].fillna(0.0)
    account_results.loc[account_results["hop1_count"] > 0, "hop2_count"] = 0.0

    DIRECT_PENALTY = config.GRAPH_RECEIVING_PENALTY
    INDIRECT_PENALTY = config.GRAPH_INDIRECT_PENALTY

    account_results["graph_penalty"] = (
        (account_results["hop1_count"] * DIRECT_PENALTY * account_results["volume_factor"])
        + (account_results["hop2_count"] * INDIRECT_PENALTY)
    ).clip(upper=0.55)

    account_results["final_risk_score"] = (
        account_results["base_risk"] + account_results["graph_penalty"]
    ).clip(lower=0.0, upper=1.0)

    account_results["investigation_flag"] = account_results["final_risk_score"] >= config.INVESTIGATION_THRESHOLD
    account_results["received_from_high_risk"] = account_results["hop1_count"] > 0

    # Universal numeric sanitization
    numeric_cols = account_results.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        account_results[col] = (
            pd.to_numeric(account_results[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    account_results["investigation_flag"] = account_results["investigation_flag"].astype(bool)
    account_results["is_high_risk_seed"] = account_results["is_high_risk_seed"].astype(bool)
    account_results["received_from_high_risk"] = account_results["received_from_high_risk"].astype(bool)

    print(
        "Super-Node Shielded Graph | nodes=%s | seeds=%s | 1-Hop=%s | 2-Hop=%s | flagged=%s",
        len(account_results), int(account_results["is_high_risk_seed"].sum()),
        int((account_results["hop1_count"] > 0).sum()), int((account_results["hop2_count"] > 0).sum()),
        int(account_results["investigation_flag"].sum()),
    )
 
    seeds_records = account_results.loc[account_results["is_high_risk_seed"], ["graph_date", "Account"]].copy()
    seeds_records["graph_date"] = seeds_records["graph_date"].astype(str)
    
    return {
        "subgraph_data": {
            "account_results": account_results,
            "high_risk_seeds": seeds_records.to_dict(orient="records"),
            "investigation_threshold": float(config.INVESTIGATION_THRESHOLD),
            "exposure_evidence": pd.DataFrame(columns=["graph_date", "high_risk_sender", "exposed_receiver", "Amount Paid"])
        }
    }
 

# 5. Risk Agent Node (Mathematical Weighting)
def risk_agent_node(state: dict) -> dict:
    print("--- RUNNING RISK AGENT ---")

    graph_data = state.get("subgraph_data", {})
    account_results = graph_data.get("account_results")
 
    if account_results is None:
        raise ValueError("Risk Agent did not receive account_results from the graph node.")
    if not isinstance(account_results, pd.DataFrame):
        raise TypeError("subgraph_data['account_results'] must be a pandas DataFrame.")
 
    results = account_results.copy()
 
    required = [
        "graph_date", "Account", "base_risk", "graph_penalty",
        "final_risk_score", "investigation_flag",
    ]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Risk Agent missing graph outputs: {missing}")
 
    # NOTE: do not recompute model weighting here — the graph node already applied
    # base_risk = XGB_WEIGHT*xgb + ANOMALY_WEIGHT*ae + BEHAVIORAL_WEIGHT*behavioral,
    # and final_risk_score = base_risk + graph_penalty.
    results["final_risk_score"] = (
        pd.to_numeric(results["final_risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
 
    investigation_threshold = float(
        graph_data.get("investigation_threshold", config.INVESTIGATION_THRESHOLD)
    )
    results["investigation_flag"] = results["final_risk_score"] >= investigation_threshold
 
    results["risk_label"] = np.select(
        [results["investigation_flag"], results["final_risk_score"] >= config.MEDIUM_RISK_THRESHOLD],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
 
    flagged_accounts = results[results["investigation_flag"]].copy()
 
    print(
        "Risk agent results | high=%s | medium=%s | low=%s | routed for investigation=%s",
        int((results["risk_label"] == "High Risk").sum()),
        int((results["risk_label"] == "Medium Risk").sum()),
        int((results["risk_label"] == "Low Risk").sum()),
        len(flagged_accounts),
    )
 
    return {
        "risk_scores": results,
        "final_insights": flagged_accounts.to_dict(orient="records"),
    }
 
 

# 6. LLM API Node
def llm_api_node(state: dict) -> dict:
    print("--- BUILDING ACCOUNT/DAY SAR CASES ---")

    alerts = state.get("final_insights", [])
    graph_data = state.get("subgraph_data", {})
 
    if not alerts:
        print("No investigation-level accounts detected — skipping SAR generation.")
        return {"sar_reports": []}
 
    exposure_evidence = graph_data.get("exposure_evidence")
    if exposure_evidence is None:
        exposure_evidence = pd.DataFrame(
            columns=["graph_date", "high_risk_sender", "exposed_receiver", "Amount Paid"]
        )
    elif not isinstance(exposure_evidence, pd.DataFrame):
        exposure_evidence = pd.DataFrame(exposure_evidence)
 
    if not exposure_evidence.empty:
        exposure_evidence = exposure_evidence.copy()
        exposure_evidence["graph_date"] = (
            pd.to_datetime(exposure_evidence["graph_date"], errors="coerce").dt.normalize()
        )
 
    # Sort by highest risk score first
    alerts = sorted(alerts, key=lambda x: x.get("final_risk_score", 0.0), reverse=True)
 
    # --- YOUR FIX: Limit to top 25 ---
    MAX_REPORTS = 25
    top_alerts = alerts[:MAX_REPORTS]
    
    print(f"Detected {len(alerts)} alerts. Generating SARs for the top {len(top_alerts)} highest-risk cases...")
 
    sar_reports = []
    
    # CRITICAL: Make sure this says 'top_alerts', not 'alerts'
    for alert in top_alerts:
        account = alert["Account"]
        graph_date = pd.to_datetime(alert["graph_date"], errors="coerce")
 
        if pd.isna(graph_date):
            graph_date_str, normalized_date = "N/A", None
        else:
            normalized_date = graph_date.normalize()
            graph_date_str = normalized_date.strftime("%Y-%m-%d")
 
        reason_codes = []
        if bool(alert.get("is_high_risk_seed", False)):
            reason_codes.append({
                "code": "RC-ML-SEED",
                "description": "Account exceeded the combined ML high-risk seed threshold",
            })
        if bool(alert.get("received_from_high_risk", False)):
            reason_codes.append({
                "code": "RC-GRAPH-1HOP",
                "description": "Account received funds directly from an original high-risk seed",
            })
        if not reason_codes:
            reason_codes.append({
                "code": "RC-RISK-THRESHOLD",
                "description": "Account crossed the operational investigation risk threshold",
            })
 
        transactions = []
        if normalized_date is not None and not exposure_evidence.empty:
            same_day = exposure_evidence["graph_date"] == normalized_date
 
            incoming = exposure_evidence[same_day & (exposure_evidence["exposed_receiver"] == account)]
            for _, row in incoming.iterrows():
                amount = row.get("Amount Paid", 0.0)
                amount = 0.0 if pd.isna(amount) else float(amount)
                transactions.append({
                    "timestamp": graph_date_str,
                    "from_account": row["high_risk_sender"],
                    "to_account": row["exposed_receiver"],
                    "amount_sar": amount,
                    "direction": "inbound graph exposure",
                })
 
            outgoing = exposure_evidence[same_day & (exposure_evidence["high_risk_sender"] == account)]
            for _, row in outgoing.iterrows():
                amount = row.get("Amount Paid", 0.0)
                amount = 0.0 if pd.isna(amount) else float(amount)
                transactions.append({
                    "timestamp": graph_date_str,
                    "from_account": row["high_risk_sender"],
                    "to_account": row["exposed_receiver"],
                    "amount_sar": amount,
                    "direction": "outbound seed exposure",
                })
 
        structured_case = {
            "case_id": f"CASE-{uuid.uuid4().hex[:8].upper()}",
            "customer": {
                "customer_id": account,
                "name_masked": "Redacted Entity",
                "segment": "Transaction Monitoring Batch",
            },
            "risk_score": round(float(alert["final_risk_score"]), 6),
            "risk_tier": alert.get("risk_label", "High Risk").upper().replace(" ", "_"),
            "reason_codes": reason_codes,
            "model_scores": {
                "combined_base_risk": round(float(alert.get("base_risk", 0.0)), 6),
            },
            "graph_features": {
                "graph_date": graph_date_str,
                "graph_penalty": round(float(alert.get("graph_penalty", 0.0)), 6),
                "is_high_risk_seed": bool(alert.get("is_high_risk_seed", False)),
                "received_from_high_risk": bool(alert.get("received_from_high_risk", False)),
                "investigation_flag": bool(alert.get("investigation_flag", False)),
            },
            "transactions": transactions,
        }
 
        try:
            generated_text = generate_report(structured_case)
        except Exception as e:
            print(f"SAR generation failed for {account}: {e}")
            generated_text = f"SYSTEM ERROR: Failed to generate SAR. {e}"
 
        sar_reports.append(generated_text)
 
    print(f"Generated {len(sar_reports)} account/day SAR drafts.")
    
    # We still return ALL alerts to the dashboard/state, so the compliance officer can see all 9,000 cases,
    # but we only pass the 25 generated SAR reports.
    return {"sar_reports": sar_reports, "final_insights": alerts}
 

# 7. Report Node
def report_node(state: dict) -> dict:
    print("--- FINALIZING REPORT ---")
    existing_insights = state.get("final_insights", [])    
    return {
        "final_insights": existing_insights
    }






def _print_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("  Confusion Matrix:")
    print(f"    [[TN: {cm[0,0]}  FP: {cm[0,1]}]]")
    print(f"    [[FN: {cm[1,0]}  TP: {cm[1,1]}]]")


def evaluation_node(state: dict) -> dict:
    val_df = state["final_df"]
    true_label_col = "Is Laundering"
    
    if true_label_col not in val_df.columns:
        print(f"ℹ️ Ground truth column '{true_label_col}' not found. Skipping evaluation metrics.")
        return {}

    print(f"--- STARTING PIPELINE EVALUATION ON {len(val_df)} TRANSACTIONS ---")
    y_true_trans = val_df[true_label_col].astype(int).to_numpy()

    ml_probs = state["ml_probabilities"]
    xgb_preds = np.array(ml_probs["xgb_prob"]) >= 0.40
    ae_preds = np.array(ml_probs["anomaly_score"]) >= 0.97
    beh_preds = np.array(ml_probs["behavioral_score"]) >= 0.75

    print("\n================ INDIVIDUAL MODEL PERFORMANCE ================")
    _print_metrics("XGBoost Model", y_true_trans, xgb_preds)
    _print_metrics("Autoencoder Anomaly Model", y_true_trans, ae_preds)
    _print_metrics("Behavioral Risk Model", y_true_trans, beh_preds)

    if os.environ.get("FINDETECT_RUN_WEIGHT_SEARCH") == "1":
        import optimize_fusion_weights
        from optimize_fusion_weights import optimize, run_fusion, evaluate_candidate
        import inspect

        print("Loaded optimize_fusion_weights from:", optimize_fusion_weights.__file__, flush=True)
        print("optimize() signature:", inspect.signature(optimize), flush=True)

        # Build the scored dataframe the optimizer needs
        opt_df = state["final_df"].copy()
        opt_df["xgb_prob"] = ml_probs["xgb_prob"]
        opt_df["anomaly_score"] = ml_probs["anomaly_score"]
        opt_df["behavioral_score"] = ml_probs["behavioral_score"]
        opt_df["_graph_date"] = pd.to_datetime(
            opt_df["Transaction_Date"] if "Transaction_Date" in opt_df.columns else opt_df["Timestamp"]
        ).dt.normalize()

        # Split by date so the search and the confirmation never see the same rows
        cutoff_date = opt_df["_graph_date"].quantile(0.5)
        search_half = opt_df[opt_df["_graph_date"] <= cutoff_date]
        holdout_half = opt_df[opt_df["_graph_date"] > cutoff_date]

        print(f"\n================ WEIGHT SEARCH (n_iter=200, search half: {len(search_half)} rows) ================")
        
        # FIX 1: Pass 'search_half' instead of undefined 'df'
        best_config, best_result = optimize(
            search_half, 
            n_iter=200, 
            target_min=0.80, 
            target_max=0.85, 
            objective="precision"
        )

        print(f"\n================ HOLDOUT CONFIRMATION ({len(holdout_half)} rows) ================")
        confirm_account_results = run_fusion(
            holdout_half,  # Use holdout_half here
            best_config["FINDETECT_XGB_WEIGHT"],
            best_config["FINDETECT_ANOMALY_WEIGHT"],
            best_config["FINDETECT_BEHAVIORAL_WEIGHT"],
            best_config["FINDETECT_SEED_THRESHOLD"],
            best_config["FINDETECT_GRAPH_PENALTY"],
            best_config["FINDETECT_INDIRECT_PENALTY"],
        )
        
        # FIX 2: Pass 'holdout_half' instead of undefined 'df'
        confirm = evaluate_candidate(holdout_half, confirm_account_results, objective="precision")

        print("Search-half result :", best_result)
        print("Holdout result      :", confirm)
        if confirm is None:
            print("⚠️  No threshold on the holdout half reaches the target recall band at all — "
                  "the search-half result does not generalize. Do not deploy these weights as-is.")
        else:
            drop = best_result["precision"] - confirm["precision"]
            print(f"Precision drop search→holdout: {drop:.4f} "
                  f"({'small, looks stable' if drop < 0.002 else 'notable — treat search-half numbers with caution'})")
    # <<< END OF WEIGHT-SEARCH BLOCK >>>

    account_results = state["subgraph_data"]["account_results"]
    val_df["_graph_date"] = pd.to_datetime(val_df["Transaction_Date"] if "Transaction_Date" in val_df.columns else val_df["Timestamp"]).dt.normalize()
    
    available_cols = [c for c in ["graph_date", "Account", "final_risk_score", "investigation_flag"] if c in account_results.columns]
    
    evaluated_df = val_df.merge(
        account_results[available_cols],
        left_on=["_graph_date", "Account"],
        right_on=["graph_date", "Account"],
        how="left"
    )
    
    agent_preds = evaluated_df["investigation_flag"].fillna(False).astype(bool).to_numpy()

    print("\n================ FINAL RISK AGENT PERFORMANCE ================")
    _print_metrics("Risk Agent (Investigation Flag Threshold)", y_true_trans, agent_preds)

    return {
        "evaluation_metrics": {
            "xgboost": precision_recall_fscore_support(y_true_trans, xgb_preds, average='binary', zero_division=0),
            "risk_agent": precision_recall_fscore_support(y_true_trans, agent_preds, average='binary', zero_division=0)
        }
    }









