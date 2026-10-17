# node

import os
import uuid

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

import config
from LLM import generate_report
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from customer_deviation import add_customer_deviation_features
from meta_learner import predict_with_meta_learner


def preprocessing_node(state: dict) -> dict:
    print("--- LOADING FINAL FEATURE DATASET ---")

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

        # Gentle linear scaling (no retraining possible)
        probabilities = np.clip(raw_probabilities * 1.2, 0.0, 1.0).astype(np.float32)

        print(
            f"Optimized XGBoost scoring complete | rows={len(probabilities)} | "
            f"min={probabilities.min():.6f} | mean={probabilities.mean():.6f} | "
            f"max={probabilities.max():.6f}"
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
            f"Autoencoder scoring complete | rows={len(anomaly_scores)} | "
            f"min={anomaly_scores.min():.6f} | mean={anomaly_scores.mean():.6f} | "
            f"max={anomaly_scores.max():.6f}"
        )

        return {"ml_probabilities": {"anomaly_score": anomaly_scores.tolist()}}

    except Exception:
        print("Autoencoder node failed")
        raise


_BEH_DRIFT_RATE = 1.2            # exponential response for amount/frequency drift
_BEH_MAD_SCALE = 1.25            # tighter than config.MAD_SCALE_FACTOR (1.4826) on purpose
_BEH_Z_SATURATION = 4.0          # peer z-score at which peer_score reaches 1.0
_BEH_SYNERGY_MULTIPLIER = 1.4    # amount drift + velocity spike together
_BEH_SYNERGY_THRESHOLD = 0.5


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

    # -- Customer baseline drift: amount
    amount_ratio = numeric("amount_vs_prev_max_ratio")
    amount_excess = (amount_ratio - 1.0).clip(lower=0.0)
    amount_drift_score = (
        (1.0 - np.exp(-amount_excess * _BEH_DRIFT_RATE))
        .where(amount_ratio > 0, 0.0)
        .fillna(0.0).clip(0.0, 1.0)
    )

    # -- Customer baseline drift: frequency
    freq_ratio = numeric("transaction_frequency_ratio")
    frequency_excess = (freq_ratio - 1.0).clip(lower=0.0)
    frequency_drift_score = (
        (1.0 - np.exp(-frequency_excess * _BEH_DRIFT_RATE))
        .where(freq_ratio > 0, 0.0)
        .fillna(0.0).clip(0.0, 1.0)
    )

    # -- Peer-group comparison (MAD scaling on log amount)
    amount_paid = numeric("Amount Paid").fillna(0.0).clip(lower=0.0)
    log_amount = np.log1p(amount_paid)
    peer_group = (
        df["Payment Format"].fillna("UNKNOWN").astype(str)
        if "Payment Format" in df.columns
        else pd.Series("GLOBAL", index=df.index)
    )

    peer_df = pd.DataFrame({"group": peer_group, "value": log_amount})
    grouped = peer_df.groupby("group")["value"]
    group_size = grouped.transform("size")
    group_median = grouped.transform("median")
    group_mad = (peer_df["value"] - group_median).abs().groupby(peer_df["group"]).transform("median")

    global_median = float(np.nanmedian(log_amount))
    global_mad = float(np.nanmedian(np.abs(log_amount - global_median)))
    global_scale = max(_BEH_MAD_SCALE * global_mad, 1e-6)

    use_global = (group_size < config.MIN_PEER_GROUP_SIZE) | (group_mad <= 1e-9) | group_mad.isna()
    center = group_median.where(~use_global, global_median)
    scale = (_BEH_MAD_SCALE * group_mad).where(~use_global, global_scale)

    robust_z = ((log_amount - center) / scale).clip(lower=0.0)
    peer_score = (robust_z / _BEH_Z_SATURATION).clip(0.0, 1.0).fillna(0.0)

    # -- Rolling velocity acceleration (share of 72h activity in the last 24h)
    count_24h = numeric("outgoing_count_24h").fillna(0.0).clip(lower=0.0)
    count_72h = numeric("outgoing_count_72h").fillna(0.0).clip(lower=0.0)
    recent_share = count_24h / count_72h.replace(0, np.nan)
    velocity_score = (
        ((recent_share - (1.0 / 3.0)) / (2.0 / 3.0)).clip(0.0, 1.0).fillna(0.0)
    )

    # -- Weighted sum + compound-risk multiplier
    raw_behavioral = (
        config.AMOUNT_DRIFT_WEIGHT * amount_drift_score
        + config.FREQUENCY_DRIFT_WEIGHT * frequency_drift_score
        + config.PEER_GROUP_WEIGHT * peer_score
        + config.VELOCITY_WEIGHT * velocity_score
    )
    synergy_mask = (
        (amount_drift_score > _BEH_SYNERGY_THRESHOLD) & (velocity_score > _BEH_SYNERGY_THRESHOLD)
    )
    raw_behavioral = raw_behavioral.mask(synergy_mask, raw_behavioral * _BEH_SYNERGY_MULTIPLIER)

    behavioral_scores = np.clip(np.asarray(raw_behavioral, dtype=np.float32), 0.0, 1.0)

    if len(behavioral_scores) != len(df):
        raise RuntimeError("Behavioral score count does not match transaction count.")
    if not np.isfinite(behavioral_scores).all():
        raise RuntimeError("Behavioral model produced invalid scores.")

    print(
        f"Behavioral model scored {len(behavioral_scores)} transactions | "
        f"range={behavioral_scores.min():.4f}-{behavioral_scores.max():.4f} | "
        f"mean={behavioral_scores.mean():.4f}"
    )

    return {"ml_probabilities": {"behavioral_score": behavioral_scores.tolist()}}


# 4. Upgraded Graph Context Node (Dynamic Penalties)
def graph_context_node(state: dict) -> dict:
    print("Running enhanced multi-hop graph analysis with pattern detection")

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
        (df[["xgb_prob", "anomaly_score", "behavioral_score"]] > 0.3).sum(axis=1) >= 2,
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
    daily_degree = df.groupby(["_graph_date", "Account"])["Account_duplicated_0"].nunique().reset_index(name="daily_degree")

    date_merge_col = "_graph_date" if "_graph_date" in account_results.columns else "graph_date"
    account_results = account_results.merge(
        daily_degree,
        left_on=[date_merge_col, "Account"],
        right_on=["_graph_date", "Account"],
        how="left"
    )
    account_results["daily_degree"] = account_results["daily_degree"].fillna(1.0)

    MAX_NORMAL_DEGREE = 50

    account_results["is_high_risk_seed"] = (
        (account_results["base_risk"] > config.SEED_THRESHOLD) &
        (account_results["daily_degree"] <= MAX_NORMAL_DEGREE)
    )

    seeds = account_results[account_results["is_high_risk_seed"]][["_graph_date", "Account"]].drop_duplicates()
    seeds = seeds.rename(columns={"Account": "Seed_Account", "_graph_date": "graph_date"})

    amount_col = "Amount Paid" if "Amount Paid" in df.columns else "Amount Received"
    df["_clean_amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).clip(lower=0.0) if amount_col in df.columns else 1.0

    edges = df[df["Account"] != df["Account_duplicated_0"]][["_graph_date", "Account", "Account_duplicated_0", "_clean_amount"]].copy()
    edges.columns = ["graph_date", "sender", "receiver", "amount"]

    # ============================================================================
    # PATTERN DETECTION - Bidirectional Relationships (Layering)
    # ============================================================================
    edge_pairs = edges.groupby(["graph_date", "sender", "receiver"])["amount"].sum().reset_index()
    reverse_pairs = edge_pairs.rename(columns={"sender": "receiver", "receiver": "sender"})

    bidirectional = edge_pairs.merge(
        reverse_pairs[["graph_date", "sender", "receiver"]],
        on=["graph_date", "sender", "receiver"],
        how="inner"
    )

    bidir_accounts_a = bidirectional[["graph_date", "sender"]].rename(columns={"sender": "Account"})
    bidir_accounts_b = bidirectional[["graph_date", "receiver"]].rename(columns={"receiver": "Account"})
    bidir_all = pd.concat([bidir_accounts_a, bidir_accounts_b]).drop_duplicates()
    bidir_counts = bidir_all.groupby(["graph_date", "Account"]).size().reset_index(name="bidir_count")

    print(f"Pattern Detection: Found {len(bidir_all)} bidirectional relationships (layering indicator)")

    # ============================================================================
    # PATTERN DETECTION - Fan-In/Fan-Out (Mule Detection)
    # ============================================================================
    out_degree = edges.groupby(["graph_date", "sender"])["receiver"].nunique().reset_index(name="out_degree")
    out_degree = out_degree.rename(columns={"sender": "Account"})

    in_degree = edges.groupby(["graph_date", "receiver"])["sender"].nunique().reset_index(name="in_degree")
    in_degree = in_degree.rename(columns={"receiver": "Account"})

    degree_df = out_degree.merge(in_degree, on=["graph_date", "Account"], how="outer").fillna(0)

    degree_df["is_fan_pattern"] = (
        (degree_df["in_degree"] >= 5) & (degree_df["in_degree"] <= 50) &
        (degree_df["out_degree"] >= 5) & (degree_df["out_degree"] <= 50)
    ).astype(int)

    fan_pattern_accounts = degree_df[degree_df["is_fan_pattern"] == 1][["graph_date", "Account", "in_degree", "out_degree"]].copy()
    fan_pattern_accounts["fan_score"] = (fan_pattern_accounts["in_degree"] * fan_pattern_accounts["out_degree"]) / 2500.0
    fan_pattern_accounts["fan_score"] = fan_pattern_accounts["fan_score"].clip(0.0, 1.0)

    print(f"Pattern Detection: Found {len(fan_pattern_accounts)} fan-in/fan-out patterns (mule indicator)")

    # HOP 1: DIRECT EXPOSURE
    seed_send = edges.merge(seeds, left_on=["graph_date", "sender"], right_on=["graph_date", "Seed_Account"])
    hop1_receivers = seed_send[["graph_date", "receiver", "Seed_Account", "amount"]].rename(columns={"receiver": "Account"})

    seed_recv = edges.merge(seeds, left_on=["graph_date", "receiver"], right_on=["graph_date", "Seed_Account"])
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

    # HOP 3: TERTIARY EXPOSURE
    hop2_unique = hop2_all[["graph_date", "Account"]].drop_duplicates().rename(columns={"Account": "Hop2_Account"})

    hop3_recv = edges.merge(hop2_unique, left_on=["graph_date", "sender"], right_on=["graph_date", "Hop2_Account"])
    hop3_receivers = hop3_recv[["graph_date", "receiver", "Hop2_Account"]].rename(columns={"receiver": "Account"})

    hop3_send = edges.merge(hop2_unique, left_on=["graph_date", "receiver"], right_on=["graph_date", "Hop2_Account"])
    hop3_senders = hop3_send[["graph_date", "sender", "Hop2_Account"]].rename(columns={"sender": "Account"})

    hop3_all = pd.concat([hop3_receivers, hop3_senders])
    hop3_counts = hop3_all.groupby(["graph_date", "Account"])["Hop2_Account"].nunique().reset_index(name="hop3_count")

    # ============================================================================
    # MERGE & APPLY PENALTIES
    # ============================================================================
    if "_graph_date" in account_results.columns:
        account_results = account_results.rename(columns={"_graph_date": "graph_date"})

    account_results = account_results.merge(hop1_stats, on=["graph_date", "Account"], how="left")
    account_results["hop1_count"] = account_results["hop1_count"].fillna(0.0)
    account_results["volume_factor"] = account_results["volume_factor"].fillna(1.0)

    account_results = account_results.merge(hop2_counts, on=["graph_date", "Account"], how="left")
    account_results["hop2_count"] = account_results["hop2_count"].fillna(0.0)

    account_results = account_results.merge(hop3_counts, on=["graph_date", "Account"], how="left")
    account_results["hop3_count"] = account_results["hop3_count"].fillna(0.0)

    account_results = account_results.merge(bidir_counts, on=["graph_date", "Account"], how="left")
    account_results["bidir_count"] = account_results["bidir_count"].fillna(0.0)

    account_results = account_results.merge(
        fan_pattern_accounts[["graph_date", "Account", "fan_score"]],
        on=["graph_date", "Account"],
        how="left"
    )
    account_results["fan_score"] = account_results["fan_score"].fillna(0.0)

    # Zero out lower hops if higher hop exists
    account_results.loc[account_results["hop1_count"] > 0, "hop2_count"] = 0.0
    account_results.loc[account_results["hop2_count"] > 0, "hop3_count"] = 0.0

    DIRECT_PENALTY = config.GRAPH_RECEIVING_PENALTY
    INDIRECT_PENALTY = config.GRAPH_INDIRECT_PENALTY
    TERTIARY_PENALTY = INDIRECT_PENALTY * config.TERTIARY_PENALTY_FACTOR

    BIDIR_PENALTY = config.BIDIR_PENALTY     # Each bidirectional relationship adds this
    FAN_BOOST = config.FAN_BOOST             # Fan pattern adds up to this much

    account_results["graph_penalty"] = (
        (account_results["hop1_count"] * DIRECT_PENALTY * account_results["volume_factor"])
        + (account_results["hop2_count"] * INDIRECT_PENALTY)
        + (account_results["hop3_count"] * TERTIARY_PENALTY)
        + (account_results["bidir_count"] * BIDIR_PENALTY)
        + (account_results["fan_score"] * FAN_BOOST)
    ).clip(upper=config.GRAPH_PENALTY_CAP)

    # Pattern-based base_risk boost
    pattern_boost = (
        (account_results["bidir_count"] > 0).astype(float) * 0.08 +
        account_results["fan_score"] * 0.10
    ).clip(0.0, 0.20)

    account_results["base_risk"] = (account_results["base_risk"] + pattern_boost).clip(0.0, 1.0)

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
        f"Enhanced Graph | nodes={len(account_results)} | "
        f"seeds={int(account_results['is_high_risk_seed'].sum())} | "
        f"1-Hop={int((account_results['hop1_count'] > 0).sum())} | "
        f"2-Hop={int((account_results['hop2_count'] > 0).sum())} | "
        f"3-Hop={int((account_results['hop3_count'] > 0).sum())} | "
        f"Bidir={int((account_results['bidir_count'] > 0).sum())} | "
        f"Fan={int((account_results['fan_score'] > 0).sum())} | "
        f"flagged={int(account_results['investigation_flag'].sum())}"
    )

    seeds_records = account_results.loc[account_results["is_high_risk_seed"], ["graph_date", "Account"]].copy()
    seeds_records["graph_date"] = seeds_records["graph_date"].astype(str)

    # Optional: save per-transaction scores for offline weight tuning.
    # Opt-in so training runs / normal runs never overwrite it.
    #   PowerShell:  $env:FINDETECT_SAVE_TUNING_SCORES="1"
    if os.environ.get("FINDETECT_SAVE_TUNING_SCORES") == "1" and "Is Laundering" in state["final_df"].columns:
        df[["_graph_date", "Account", "Account_duplicated_0",
            "xgb_prob", "anomaly_score", "behavioral_score"]] \
            .assign(is_laundering=state["final_df"]["Is Laundering"].astype(int).values) \
            .to_parquet(config.REPO_ROOT / "scores_for_tuning.parquet")

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

    results["final_risk_score"] = (
        pd.to_numeric(results["final_risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )

    investigation_threshold = float(
        graph_data.get("investigation_threshold", config.INVESTIGATION_THRESHOLD)
    )
    results["investigation_flag"] = results["final_risk_score"] >= investigation_threshold

    # ============================================================================
    # HARD RULE AUTO-CLEAR FILTERS (OPTIMIZED FOR MAXIMUM RECALL)
    # ============================================================================
    flagged_accounts = results[results["investigation_flag"]].copy()
    initial_queue_size = len(flagged_accounts)

    # Filter 1: very mild base_risk check to filter absolute noise
    if "base_risk" in flagged_accounts.columns:
        flagged_accounts = flagged_accounts[flagged_accounts["base_risk"] >= 0.05]
        print(f"Filter 1 (base_risk >= 0.05): {initial_queue_size} → {len(flagged_accounts)}")

    # Filter 2: permissive signal check
    if "hop1_count" in flagged_accounts.columns:
        has_anomaly = "anomaly_score" in flagged_accounts.columns

        if has_anomaly:
            flagged_accounts = flagged_accounts[
                (flagged_accounts["hop1_count"] >= 1) |
                (flagged_accounts["base_risk"] >= 0.12) |
                (flagged_accounts["anomaly_score"] >= 0.55) |
                (flagged_accounts["behavioral_score"] >= 0.35)
            ]
            print(f"Filter 2 (permissive signal check): → {len(flagged_accounts)}")

    # Filter 3: enforce the investigation threshold
    flagged_accounts = flagged_accounts[flagged_accounts["final_risk_score"] >= investigation_threshold]
    print(f"Filter 3 (final_risk_score >= {investigation_threshold}): → {len(flagged_accounts)}")

    final_queue_size = len(flagged_accounts)
    reduction_pct = ((initial_queue_size - final_queue_size) / initial_queue_size * 100) if initial_queue_size > 0 else 0
    print(f"Hard Rule Filters: Queue reduced {initial_queue_size} → {final_queue_size} ({reduction_pct:.1f}% reduction)")

    # ============================================================================
    # TIERED INVESTIGATION WORKFLOW
    # ============================================================================
    flagged_accounts["investigation_tier"] = np.select(
        [
            flagged_accounts["final_risk_score"] >= config.TIER_1_THRESHOLD,
            flagged_accounts["final_risk_score"] >= config.TIER_2_THRESHOLD,
        ],
        ["TIER_1_HIGH", "TIER_2_MEDIUM"],
        default="TIER_3_LOW",
    )

    flagged_accounts["risk_label"] = flagged_accounts["investigation_tier"]

    flagged_accounts = flagged_accounts.sort_values(
        by="final_risk_score",
        ascending=False
    ).reset_index(drop=True)

    tier_counts = flagged_accounts["investigation_tier"].value_counts()
    print(
        f"Tiered Queue | TIER_1_HIGH: {tier_counts.get('TIER_1_HIGH', 0)} | "
        f"TIER_2_MEDIUM: {tier_counts.get('TIER_2_MEDIUM', 0)} | "
        f"TIER_3_LOW: {tier_counts.get('TIER_3_LOW', 0)}"
    )

    # ============================================================================
    # PROPAGATE FILTERED FLAGS BACK TO MAIN RESULTS (CRITICAL FOR EVALUATION)
    # ============================================================================
    filtered_flags = flagged_accounts[["graph_date", "Account"]].copy()
    filtered_flags["investigation_flag"] = True

    results = results.drop(columns=["investigation_flag", "investigation_tier"], errors="ignore")

    results = results.merge(filtered_flags, on=["graph_date", "Account"], how="left")
    results["investigation_flag"] = results["investigation_flag"].fillna(False).astype(bool)

    tier_info = flagged_accounts[["graph_date", "Account", "investigation_tier"]].copy()
    results = results.merge(tier_info, on=["graph_date", "Account"], how="left")
    results["investigation_tier"] = results["investigation_tier"].fillna("NOT_FLAGGED")

    results["risk_label"] = np.select(
        [results["investigation_flag"], results["final_risk_score"] >= config.MEDIUM_RISK_THRESHOLD],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )

    high_count = int((results["risk_label"] == "High Risk").sum())
    med_count = int((results["risk_label"] == "Medium Risk").sum())
    low_count = int((results["risk_label"] == "Low Risk").sum())

    print(
        f"Risk agent results | high={high_count} | medium={med_count} | "
        f"low={low_count} | routed for investigation={len(flagged_accounts)}"
    )
    flagged_accounts["graph_date"] = flagged_accounts["graph_date"].astype(str)
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

    alerts = sorted(alerts, key=lambda x: x.get("final_risk_score", 0.0), reverse=True)

    MAX_REPORTS = 25
    top_alerts = alerts[:MAX_REPORTS]

    print(f"Detected {len(alerts)} alerts. Generating SARs for the top {len(top_alerts)} highest-risk cases...")

    sar_reports = []

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

    # Only the top MAX_REPORTS alerts are returned to state / SAR generation.
    return {"sar_reports": sar_reports, "final_insights": top_alerts}


# 7. Report Node
def report_node(state: dict) -> dict:
    print("--- FINALIZING REPORT ---")
    existing_insights = state.get("final_insights", [])
    return {
        "final_insights": existing_insights
    }


def _threshold_sweep(model_name: str, scores, y_true, thresholds=None):
    """Print precision/recall/F1 at several score cutoffs (transaction grain)."""
    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true).astype(int)

    if len(scores) != len(y_true):
        print(f"Sweep skipped for {model_name}: {len(scores)} scores vs {len(y_true)} labels")
        return

    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]

    base_rate = y_true.mean()
    print(f"\n--- Threshold sweep: {model_name} | base rate={base_rate:.4%} ---")
    for t in thresholds:
        pred = scores >= t
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        lift = (p / base_rate) if base_rate > 0 else 0.0
        print(
            f"thr={t:.2f} | flagged={int(pred.sum()):>8} | "
            f"precision={p:.4f} | recall={r:.4f} | f1={f:.4f} | lift={lift:5.1f}x"
        )


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

    print("--- STARTING PIPELINE EVALUATION ---")

    # 1. Individual models are evaluated at transaction level
    y_true_trans = val_df[true_label_col].astype(int).to_numpy()
    ml_probs = state.get("ml_probabilities", {})

    # Operating points picked from the threshold sweeps
    xgb_preds = np.array(ml_probs.get("xgb_prob", [])) >= 0.50
    ae_preds = np.array(ml_probs.get("anomaly_score", [])) >= 0.98
    beh_preds = np.array(ml_probs.get("behavioral_score", [])) >= 0.40

    print("\n================ INDIVIDUAL MODEL PERFORMANCE (Transaction Grain) ================")
    _print_metrics("XGBoost Model", y_true_trans, xgb_preds)
    _print_metrics("Autoencoder Anomaly Model", y_true_trans, ae_preds)
    _print_metrics("Behavioral Risk Model", y_true_trans, beh_preds)

    _threshold_sweep("Behavioral Risk Model", ml_probs.get("behavioral_score", []), y_true_trans)
    _threshold_sweep("XGBoost Model", ml_probs.get("xgb_prob", []), y_true_trans)
    _threshold_sweep("Autoencoder Anomaly Model", ml_probs.get("anomaly_score", []), y_true_trans,
                     thresholds=[0.50, 0.70, 0.80, 0.90, 0.92, 0.95, 0.98])

    # 2. Risk Agent evaluated at Account/Day grain
    print("\n================ FINAL RISK AGENT PERFORMANCE (Account/Day Grain) ================")
    account_results = state.get("risk_scores", state["subgraph_data"]["account_results"])

    date_col = "Transaction_Date" if "Transaction_Date" in val_df.columns else "Timestamp"
    val_df["_graph_date"] = pd.to_datetime(val_df[date_col], errors="coerce").dt.normalize()

    account_day_truth = val_df.groupby(["_graph_date", "Account"])[true_label_col].max().reset_index()
    account_day_truth.rename(columns={true_label_col: "is_laundering_day"}, inplace=True)

    available_cols = [c for c in ["graph_date", "Account", "final_risk_score", "investigation_flag"] if c in account_results.columns]
    evaluated_days = account_day_truth.merge(
        account_results[available_cols],
        left_on=["_graph_date", "Account"],
        right_on=["graph_date", "Account"],
        how="left"
    )

    evaluated_days["investigation_flag"] = evaluated_days["investigation_flag"].fillna(False).astype(bool)

    # Save account/day results so pick_threshold.py can choose a threshold offline
    try:
        evaluated_days.to_parquet(config.REPO_ROOT / "eval_days.parquet")
    except Exception as e:
        print(f"⚠️ Could not save eval_days.parquet: {e}")

    y_true_days = evaluated_days["is_laundering_day"].astype(int).to_numpy()
    y_pred_days = evaluated_days["investigation_flag"].to_numpy()

    _print_metrics("Risk Agent (Account/Day Alerts)", y_true_days, y_pred_days)

    total_days = len(evaluated_days)
    flagged_days = int(y_pred_days.sum())
    tp_days = int((y_true_days & y_pred_days).sum())

    print(f"\n📊 Summary: Flagged {flagged_days} out of {total_days} Account/Days.")
    print(f"   True Positive Days: {tp_days}")
    if flagged_days > 0:
        print(f"   ✅ Real Account/Day Precision: {tp_days / flagged_days:.4f} ({(tp_days / flagged_days)*100:.1f}%)")

    def to_native(metrics):
        return tuple(float(x) if x is not None else None for x in metrics)

    return {
        "evaluation_metrics": {
            "xgboost": to_native(precision_recall_fscore_support(y_true_trans, xgb_preds, average='binary', zero_division=0)),
            "autoencoder": to_native(precision_recall_fscore_support(y_true_trans, ae_preds, average='binary', zero_division=0)),
            "behavioral": to_native(precision_recall_fscore_support(y_true_trans, beh_preds, average='binary', zero_division=0)),
            "risk_agent_account_day": to_native(precision_recall_fscore_support(y_true_days, y_pred_days, average='binary', zero_division=0))
        }
    }


# ---------------------------------------------------------------
# 3.5 Customer Deviation Node
# ---------------------------------------------------------------
def customer_deviation_node(state: dict) -> dict:
    """Add customer-deviation features to final_df BEFORE models run."""
    print("--- ADDING CUSTOMER DEVIATION FEATURES ---")
    df = state["final_df"].copy()
    df = add_customer_deviation_features(df)
    return {"final_df": df}


# ---------------------------------------------------------------
# 4.5 Meta-Learner Node
# ---------------------------------------------------------------
def meta_learner_node(state: dict) -> dict:
    """
    Combine base model scores + graph features + customer deviation features
    using a trained stacking meta-learner.
    """
    print("--- RUNNING STACKING META-LEARNER ---")

    account_results = state["subgraph_data"]["account_results"].copy()

    df = state["final_df"].copy()
    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Timestamp"
    df["_graph_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # Attach model scores from state["ml_probabilities"] (nodes return lists only)
    ml_probs = state.get("ml_probabilities", {})
    for score_name in ["xgb_prob", "anomaly_score", "behavioral_score"]:
        values = ml_probs.get(score_name)
        if values is None:
            raise ValueError(f"Meta-learner missing model score in state: {score_name}")
        df[score_name] = (
            pd.to_numeric(pd.Series(values, index=df.index), errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        )

    # 1. Explicitly initialize missing deviation columns to 0.0 to avoid fragile fallback logic
    deviation_cols = [
        "amount_vs_own_mean", 
        "amount_vs_own_median", 
        "amount_zscore_vs_own", 
        "velocity_vs_own_avg"
    ]
    for col in deviation_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 2. Clean aggregation dictionary
    agg_dict = {
        "xgb_prob": "max",
        "anomaly_score": "max",
        "behavioral_score": "max",
        "amount_vs_own_mean": "max",
        "amount_vs_own_median": "max",
        "amount_zscore_vs_own": "max",
        "velocity_vs_own_avg": "max",
    }

    # Aggregate model scores per account per day (max — worst transaction drives risk)
    # FIX: Pass agg_dict directly instead of using **agg_dict to satisfy strict Pandas typing
    account_scores = df.groupby(["_graph_date", "Account"]).agg(agg_dict).reset_index()

    account_results = account_results.merge(
        account_scores,
        left_on=["graph_date", "Account"],
        right_on=["_graph_date", "Account"],
        how="left",
        suffixes=("", "_drop"),
    )

    # Sanitize merged columns
    for col in ["xgb_prob", "anomaly_score", "behavioral_score"] + deviation_cols:
        if col not in account_results.columns:
            account_results[col] = 0.0
        else:
            account_results[col] = pd.to_numeric(account_results[col], errors="coerce").fillna(0.0)

    # Clean up merge artifacts
    if "_graph_date" in account_results.columns:
        account_results = account_results.drop(columns=["_graph_date"])
    if "Account_drop" in account_results.columns:
        account_results = account_results.drop(columns=["Account_drop"])

    try:
        scored_df = predict_with_meta_learner(account_results)
        account_results["meta_score"] = scored_df["meta_learner_score"]

        print(
            f"Meta-learner complete | min={account_results['meta_score'].min():.4f} | "
            f"mean={account_results['meta_score'].mean():.4f} | max={account_results['meta_score'].max():.4f}"
        )
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Falling back to weighted-average final_risk_score")
        account_results["meta_score"] = account_results["final_risk_score"]

    # Replace final_risk_score with meta_score for flagging
    account_results["final_risk_score"] = account_results["meta_score"].clip(0.0, 1.0)

    state["subgraph_data"]["account_results"] = account_results
    return {"subgraph_data": state["subgraph_data"]}