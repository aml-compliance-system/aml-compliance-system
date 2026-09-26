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
    print("Running XGBoost model")
    try:
        df = _derive_missing_features(state["final_df"].copy())
 
        if not config.XGBOOST_MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found: {config.XGBOOST_MODEL_PATH}")
 
        model = xgb.Booster()
        model.load_model(str(config.XGBOOST_MODEL_PATH))
 
        X = df[_XGB_FEATURE_COLS].copy()
        dmatrix = xgb.DMatrix(X, feature_names=_XGB_FEATURE_COLS)
 
        probabilities = np.asarray(model.predict(dmatrix), dtype=np.float32)
 
        if len(probabilities) != len(df):
            raise ValueError(
                f"XGBoost prediction length mismatch: {len(probabilities)} predictions "
                f"for {len(df)} transactions."
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("XGBoost produced non-finite probabilities.")
        if (probabilities < 0.0).any() or (probabilities > 1.0).any():
            raise ValueError("binary:logistic XGBoost produced probabilities outside [0, 1].")
 
        print(
            "XGBoost scoring complete | rows=%s | min=%.6f | mean=%.6f | max=%.6f",
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
    print("Running behavioral risk model")
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
 
    # -- Customer baseline drift: amount. ratio<=1 -> no drift; >1 -> increasingly unusual.
    amount_excess = (numeric("amount_vs_prev_max_ratio") - 1.0).clip(lower=0.0)
    amount_drift_score = (
        (1.0 - np.exp(-amount_excess))
        .where(numeric("amount_vs_prev_max_ratio") > 0, 0.0)
        .fillna(0.0).clip(0.0, 1.0)
    )
 
    # -- Customer baseline drift: frequency.
    frequency_excess = (numeric("transaction_frequency_ratio") - 1.0).clip(lower=0.0)
    frequency_drift_score = (
        (1.0 - np.exp(-frequency_excess))
        .where(numeric("transaction_frequency_ratio") > 0, 0.0)
        .fillna(0.0).clip(0.0, 1.0)
    )
 
    # -- Peer-group comparison (robust z-score against payment-format peer group).
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
    global_scale = max(config.MAD_SCALE_FACTOR * global_mad, 1e-6)
 
    use_global = (group_size < config.MIN_PEER_GROUP_SIZE) | (group_mad <= 1e-9) | group_mad.isna()
    center = group_median.where(~use_global, global_median)
    scale = (config.MAD_SCALE_FACTOR * group_mad).where(~use_global, global_scale)
 
    robust_z = ((log_amount - center) / scale).clip(lower=0.0)
    peer_score = (robust_z / config.ROBUST_Z_SATURATION).clip(0.0, 1.0).fillna(0.0)
 
    # -- Rolling velocity acceleration: uniform 72h activity => ~1/3 falls in last 24h.
    count_24h = numeric("outgoing_count_24h").fillna(0.0).clip(lower=0.0)
    count_72h = numeric("outgoing_count_72h").fillna(0.0).clip(lower=0.0)
    recent_share = count_24h / count_72h.replace(0, np.nan)
    velocity_score = (
        ((recent_share - (1.0 / 3.0)) / (2.0 / 3.0)).clip(0.0, 1.0).fillna(0.0)
    )
 
    behavioral_scores = np.clip(
        (
            config.AMOUNT_DRIFT_WEIGHT * amount_drift_score
            + config.FREQUENCY_DRIFT_WEIGHT * frequency_drift_score
            + config.PEER_GROUP_WEIGHT * peer_score
            + config.VELOCITY_WEIGHT * velocity_score
        ).to_numpy(dtype=np.float32),
        0.0, 1.0,
    )
 
    if len(behavioral_scores) != len(df):
        raise RuntimeError("Behavioral score count does not match transaction count.")
    if not np.isfinite(behavioral_scores).all():
        raise RuntimeError("Behavioral model produced invalid scores.")
 
    print(
        "Behavioral model scored %s transactions | range=%.4f-%.4f | mean=%.4f",
        len(behavioral_scores), behavioral_scores.min(), behavioral_scores.max(), behavioral_scores.mean(),
    )
    print(
        "Component means | amount_drift=%.3f frequency_drift=%.3f peer=%.3f velocity=%.3f",
        amount_drift_score.mean(), frequency_drift_score.mean(), peer_score.mean(), velocity_score.mean(),
    )
 
    return {"ml_probabilities": {"behavioral_score": behavioral_scores.tolist()}}
 
# 4. Graph Context Node (cuGraph)
def graph_context_node(state: dict) -> dict:
    print("Running silent-mule graph analysis")
 
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
        if len(values) != len(df):
            raise ValueError(
                f"{score_name} length mismatch: {len(values)} scores for {len(df)} transactions."
            )
        df[score_name] = (
            pd.to_numeric(pd.Series(values, index=df.index), errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        )
 
    required = ["Account", "Account_duplicated_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Graph node missing required columns: {missing}")
 
    # Graph2_2B uses a daily directed edgelist.
    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Timestamp"
    if date_col not in df.columns:
        raise ValueError("Graph node requires Transaction_Date or Timestamp.")
 
    graph_dates = pd.to_datetime(df[date_col], errors="coerce")
    if graph_dates.isna().any():
        raise ValueError("Graph node found invalid transaction dates.")
    df["_graph_date"] = graph_dates.dt.normalize()
 
    # Phase 1: independent ML base risk (fusion weights from config).
    df["base_risk"] = (
        config.XGB_WEIGHT * df["xgb_prob"]
        + config.ANOMALY_WEIGHT * df["anomaly_score"]
        + config.BEHAVIORAL_WEIGHT * df["behavioral_score"]
    ).clip(lower=0.0, upper=1.0)
 
    # Transaction -> account/day risk: strongest observed sender evidence wins (max).
    sender_base_risk = (
        df.groupby(["_graph_date", "Account"], dropna=True)["base_risk"].max().reset_index()
    )
 
    sender_accounts = df[["_graph_date", "Account"]].dropna().copy()
    receiver_accounts = (
        df[["_graph_date", "Account_duplicated_0"]]
        .dropna()
        .rename(columns={"Account_duplicated_0": "Account"})
    )
 
    account_results = (
        pd.concat([sender_accounts, receiver_accounts], ignore_index=True).drop_duplicates()
    )
    account_results = account_results.merge(sender_base_risk, on=["_graph_date", "Account"], how="left")
    account_results["base_risk"] = account_results["base_risk"].fillna(0.0).astype(float)
 
    # Phase 2: high-risk seeds. NOTE: Graph2_2B specifies strictly-greater-than, not >=.
    account_results["is_high_risk_seed"] = account_results["base_risk"] > config.SEED_THRESHOLD
    seed_keys = (
        account_results.loc[account_results["is_high_risk_seed"], ["_graph_date", "Account"]]
        .drop_duplicates()
    )
 
    seed_edges = df.merge(seed_keys.assign(_is_seed=True), on=["_graph_date", "Account"], how="inner")
    # Self-transfer does not count as graph exposure — filter the edge, not the whole account.
    seed_edges = seed_edges[seed_edges["Account"] != seed_edges["Account_duplicated_0"]].copy()
 
    exposed_keys = (
        seed_edges[["_graph_date", "Account_duplicated_0"]]
        .dropna().drop_duplicates()
        .rename(columns={"Account_duplicated_0": "Account"})
    )
    exposed_keys["received_from_high_risk"] = True
 
    account_results = account_results.merge(exposed_keys, on=["_graph_date", "Account"], how="left")
    account_results["received_from_high_risk"] = account_results["received_from_high_risk"].eq(True)
 
    # Strict one-time graph penalty — no multiplying by transfer count, seed count, or amount.
    account_results["graph_penalty"] = np.where(
        account_results["received_from_high_risk"], config.GRAPH_RECEIVING_PENALTY, 0.0
    )
 
    account_results["final_risk_score"] = (
        account_results["base_risk"] + account_results["graph_penalty"]
    ).clip(lower=0.0, upper=1.0)
 
    account_results["investigation_flag"] = (
        account_results["final_risk_score"] >= config.INVESTIGATION_THRESHOLD
    )
 
    evidence_cols = ["_graph_date", "Account", "Account_duplicated_0"]
    if "Amount Paid" in seed_edges.columns:
        evidence_cols.append("Amount Paid")
 
    exposure_evidence = (
        seed_edges[evidence_cols]
        .rename(columns={
            "_graph_date": "graph_date",
            "Account": "high_risk_sender",
            "Account_duplicated_0": "exposed_receiver",
        })
        .copy()
    )
 
    account_results = account_results.rename(columns={"_graph_date": "graph_date"})
 
    print(
        "Silent-mule graph | transactions=%s | account/day nodes=%s | seeds(>%.2f)=%s | "
        "1-hop receivers=%s | flagged(>=%.2f)=%s",
        len(df), len(account_results), config.SEED_THRESHOLD,
        int(account_results["is_high_risk_seed"].sum()),
        int(account_results["received_from_high_risk"].sum()),
        config.INVESTIGATION_THRESHOLD, int(account_results["investigation_flag"].sum()),
    )
    # NOTE: no propagation from exposed B -> C. Only original ML high-risk seeds generate exposure.
 
    return {
        "subgraph_data": {
            "account_results": account_results,
            "high_risk_seeds": (
                account_results.loc[account_results["is_high_risk_seed"], ["graph_date", "Account"]]
                .to_dict(orient="records")
            ),
            "exposed_receivers": (
                account_results.loc[account_results["received_from_high_risk"], ["graph_date", "Account"]]
                .to_dict(orient="records")
            ),
            "exposure_evidence": exposure_evidence,
            "seed_threshold": config.SEED_THRESHOLD,
            "graph_penalty": config.GRAPH_RECEIVING_PENALTY,
            "investigation_threshold": config.INVESTIGATION_THRESHOLD,
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
        log.info("No investigation-level accounts detected — skipping SAR generation.")
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
 
    print("Generating %s account/day SAR case(s)", len(alerts))
    alerts = sorted(alerts, key=lambda x: x.get("final_risk_score", 0.0), reverse=True)
 
    sar_reports = []
    for alert in alerts:
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
            log.warning("SAR generation failed for %s: %s", account, e)
            generated_text = f"SYSTEM ERROR: Failed to generate SAR. {e}"
 
        sar_reports.append(generated_text)
 
    print("Generated %s account/day SAR drafts.", len(sar_reports))
    return {"sar_reports": sar_reports, "final_insights": alerts}
 

# 7. Report Node
def report_node(state: dict) -> dict:
    print("--- FINALIZING REPORT ---")
    existing_insights = state.get("final_insights", [])    
    return {
        "final_insights": existing_insights
    }
