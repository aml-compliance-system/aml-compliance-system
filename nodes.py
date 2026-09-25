import pandas as pd
import numpy as np
import duckdb
import xgboost as xgb
import pandas as pd
import os
from pathlib import Path

import uuid
# Import the function from your LLM.py file
from LLM import generate_report


def preprocessing_node(state: dict) -> dict:
    print("--- LOADING FINAL FEATURE DATASET ---")

    parquet_path = state.get("final_parquet_path")

    if not parquet_path:
        raise ValueError(
            "final_parquet_path is required."
        )

    parquet_path = Path(parquet_path)

    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Final parquet file not found: {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)

    print(f"✅ Final dataset loaded: {len(df)} transactions.")
    print(f"✅ Columns loaded: {len(df.columns)}")

    return {"final_df": df}


# This list MUST match the exact order of features used when you trained the model on the A100.
FEATURE_COLS = [
    "Amount Received", "Amount Paid", 
    "sender_out_degree", "sender_in_degree", "sender_unique_counterparties",
    "receiver_out_degree", "receiver_in_degree", "receiver_unique_counterparties",
    "sender_receiver_frequency", "bidirectional_relationship",
    "sender_outgoing_count_so_far_wk", "sender_unique_receivers_so_far_wk",
    "outgoing_count_24h", "outgoing_sum_24h", "outgoing_count_72h", "outgoing_sum_72h",
    "sender_total_sent_prev_week", "sender_avg_sent_prev_week",
    "sender_max_sent_prev_week", "sender_outgoing_count_prev_week",
    "receiver_total_received_prev_week", "receiver_avg_received_prev_week",
    "receiver_max_received_prev_week", "receiver_incoming_count_prev_week",
    "amount_vs_prev_max_diff", "amount_vs_prev_max_ratio",
    "transaction_frequency_diff", "transaction_frequency_ratio",
    "hours_since_last_incoming", "outgoing_to_incoming_ratio",
    "hour_of_day", "currency_mismatch", "same_bank", "payment_format_hash"
]

def run_xgboost_model(state: dict) -> dict:
    print("--- RUNNING XGBOOST MODEL ---")

    df = state["final_df"]

    model_path = "models/final_xgboost_aml_model.json"

    # -------------------------------------------------
    # 1. Validate required features
    # -------------------------------------------------
    missing = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"XGBoost missing required features: {missing}"
        )

    # -------------------------------------------------
    # 2. Load trained model
    # -------------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"XGBoost model not found: {model_path}"
        )

    model = xgb.Booster()
    model.load_model(model_path)

    # -------------------------------------------------
    # 3. Exact trained feature order
    # -------------------------------------------------
    X = df[FEATURE_COLS].copy()

    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURE_COLS
    )

    # -------------------------------------------------
    # 4. Direct probability inference
    #
    # Model objective = binary:logistic
    # Therefore model.predict() already returns
    # probabilities on the 0-1 scale.
    #
    # IMPORTANT:
    # No batch Min-Max normalization.
    # -------------------------------------------------
    probabilities = np.asarray(
        model.predict(dmatrix),
        dtype=np.float32
    )

    if len(probabilities) != len(df):
        raise ValueError(
            "XGBoost prediction length mismatch: "
            f"{len(probabilities)} predictions "
            f"for {len(df)} transactions."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "XGBoost produced non-finite probabilities."
        )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "binary:logistic XGBoost produced "
            "probabilities outside [0, 1]."
        )

    print(
        "✅ XGBoost scoring complete | "
        f"rows={len(probabilities):,} | "
        f"min={probabilities.min():.6f} | "
        f"mean={probabilities.mean():.6f} | "
        f"max={probabilities.max():.6f}"
    )

    return {
        "ml_probabilities": {
            "xgb_prob": probabilities.tolist()
        }
    }


# 3. Anomaly Model Node
def run_anomaly_model(state: dict) -> dict:
    print("--- RUNNING AUTOENCODER ANOMALY DETECTION ---")

    import os
    import joblib
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    df = state["final_df"].copy()

    # Portable Autoencoder model directory.
    # Default: models/autoencoder_final64 inside this repository.
    # Can optionally be overridden with AUTOENCODER_MODEL_DIR.
    repo_root = Path(__file__).resolve().parent

    model_dir = Path(
        os.environ.get(
            "AUTOENCODER_MODEL_DIR",
            repo_root / "models" / "autoencoder_final64"
        )
    ).expanduser().resolve()

    model_path = (
        model_dir / "autoencoder_final64.keras"
    )

    bundle_path = (
        model_dir /
        "preprocessing_and_scoring_final64.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Autoencoder model not found: {model_path}"
        )

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Autoencoder preprocessing bundle not found: {bundle_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    bundle = joblib.load(bundle_path)

    # -------------------------------------------------
    # 1. Validate required raw model features
    # -------------------------------------------------
    model_raw_features = bundle["model_raw_features"]

    missing = [
        col for col in model_raw_features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Autoencoder missing required Final dataset features: "
            f"{missing}"
        )

    work = df[model_raw_features].copy()

    # -------------------------------------------------
    # 2. Derived binary features
    #
    # IMPORTANT:
    # These must be calculated BEFORE imputation so that
    # the original missing-history information is retained.
    # -------------------------------------------------
    work["no_previous_incoming"] = (
        pd.to_numeric(
            work["hours_since_last_incoming"],
            errors="coerce"
        )
        .isna()
        .astype("float32")
    )

    amount_ratio = pd.to_numeric(
        work["amount_vs_prev_max_ratio"],
        errors="coerce"
    )

    frequency_ratio = pd.to_numeric(
        work["transaction_frequency_ratio"],
        errors="coerce"
    )

    work["undefined_history_ratio"] = (
        amount_ratio.isna() |
        frequency_ratio.isna()
    ).astype("float32")

    # -------------------------------------------------
    # 3. Clean continuous features
    # -------------------------------------------------
    continuous_features = bundle["continuous_features"]

    continuous = (
        work[continuous_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # Same trained imputer
    imputed = bundle["continuous_imputer"].transform(
        continuous
    )

    continuous_processed = pd.DataFrame(
        imputed,
        columns=continuous_features,
        index=work.index
    )

    # -------------------------------------------------
    # 4. Apply saved clipping limits
    # -------------------------------------------------
    clip_lower = bundle["clip_lower"]
    clip_upper = bundle["clip_upper"]

    for col in continuous_features:
        continuous_processed[col] = (
            continuous_processed[col]
            .clip(
                lower=clip_lower[col],
                upper=clip_upper[col]
            )
        )

    # -------------------------------------------------
    # 5. Apply trained QuantileTransformer
    # -------------------------------------------------
    transformed = bundle["quantile_transformer"].transform(
        continuous_processed[continuous_features]
    )

    transformed_df = pd.DataFrame(
        transformed,
        columns=continuous_features,
        index=work.index
    )

    # -------------------------------------------------
    # 6. Restore binary features
    # -------------------------------------------------
    transformed_df["bidirectional_relationship"] = (
        pd.to_numeric(
            work["bidirectional_relationship"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
        .astype("float32")
    )

    transformed_df["no_previous_incoming"] = (
        work["no_previous_incoming"]
        .astype("float32")
    )

    transformed_df["undefined_history_ratio"] = (
        work["undefined_history_ratio"]
        .astype("float32")
    )

    # -------------------------------------------------
    # 7. Exact active feature order expected by Final64
    # -------------------------------------------------
    active_features = bundle["active_model_features"]

    X = (
        transformed_df[active_features]
        .to_numpy(dtype=np.float32)
    )

    expected_width = model.input_shape[-1]

    if X.shape[1] != expected_width:
        raise ValueError(
            f"Autoencoder expected {expected_width} features "
            f"but received {X.shape[1]}."
        )

    # -------------------------------------------------
    # 8. Reconstruction
    # -------------------------------------------------
    reconstructed = model.predict(
        X,
        batch_size=4096,
        verbose=0
    )

    feature_errors = np.square(
        X - reconstructed
    )

    # -------------------------------------------------
    # 9. Convert each feature error to its empirical
    # percentile using the saved Final64 references
    # -------------------------------------------------
    feature_refs = bundle["feature_error_references"]

    feature_percentiles = np.empty_like(
        feature_errors,
        dtype=np.float32
    )

    for j, reference in enumerate(feature_refs):
        reference = np.asarray(reference)

        feature_percentiles[:, j] = (
            np.searchsorted(
                reference,
                feature_errors[:, j],
                side="right"
            )
            / len(reference)
        )

    # Final64 selected aggregation method = mean_all
    transaction_raw_score = (
        feature_percentiles.mean(axis=1)
    )

    # -------------------------------------------------
    # 10. Calibrate against saved transaction reference
    # -------------------------------------------------
    transaction_reference = np.asarray(
        bundle["transaction_score_reference"]
    )

    anomaly_scores = (
        np.searchsorted(
            transaction_reference,
            transaction_raw_score,
            side="right"
        )
        / len(transaction_reference)
    ).astype(np.float32)

    print(
        "✅ Autoencoder scoring complete | "
        f"rows={len(anomaly_scores):,} | "
        f"min={anomaly_scores.min():.6f} | "
        f"mean={anomaly_scores.mean():.6f} | "
        f"max={anomaly_scores.max():.6f}"
    )

    return {
        "ml_probabilities": {
            "anomaly_score": anomaly_scores.tolist()
        }
    }


def run_behavioral_model(state: dict) -> dict:
    print("--- RUNNING BEHAVIORAL RISK MODEL ---")

    import numpy as np
    import pandas as pd

    df = state["final_df"].copy()

    # -------------------------------------------------
    # 1. Required behavioral features
    # -------------------------------------------------
    required = [
        "Amount Paid",
        "amount_vs_prev_max_ratio",
        "transaction_frequency_ratio",
        "outgoing_count_24h",
        "outgoing_count_72h",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Behavioral model missing required features: "
            f"{missing}"
        )

    def numeric(col):
        return (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    # -------------------------------------------------
    # 2. Customer baseline drift: transaction amount
    #
    # ratio <= 1  -> no positive drift
    # ratio > 1   -> increasingly unusual vs previous max
    # -------------------------------------------------
    amount_ratio = numeric(
        "amount_vs_prev_max_ratio"
    )

    amount_excess = (
        amount_ratio - 1.0
    ).clip(lower=0.0)

    amount_drift_score = (
        1.0 - np.exp(-amount_excess)
    )

    amount_drift_score = (
        amount_drift_score
        .where(amount_ratio > 0, 0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # -------------------------------------------------
    # 3. Customer baseline drift: transaction frequency
    # -------------------------------------------------
    frequency_ratio = numeric(
        "transaction_frequency_ratio"
    )

    frequency_excess = (
        frequency_ratio - 1.0
    ).clip(lower=0.0)

    frequency_drift_score = (
        1.0 - np.exp(-frequency_excess)
    )

    frequency_drift_score = (
        frequency_drift_score
        .where(frequency_ratio > 0, 0.0)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    # -------------------------------------------------
    # 4. Peer-group comparison
    #
    # Compare log transaction amount with peers using
    # median + MAD (robust to extreme outliers).
    #
    # Payment Format is used as the peer group when
    # available. Otherwise use the whole batch.
    # -------------------------------------------------
    amount_paid = (
        numeric("Amount Paid")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    log_amount = np.log1p(amount_paid)

    if "Payment Format" in df.columns:
        peer_group = (
            df["Payment Format"]
            .fillna("UNKNOWN")
            .astype(str)
        )
    else:
        peer_group = pd.Series(
            "GLOBAL",
            index=df.index
        )

    peer_df = pd.DataFrame({
        "group": peer_group,
        "value": log_amount
    })

    group_size = (
        peer_df
        .groupby("group")["value"]
        .transform("size")
    )

    group_median = (
        peer_df
        .groupby("group")["value"]
        .transform("median")
    )

    abs_dev = (
        peer_df["value"] - group_median
    ).abs()

    group_mad = (
        abs_dev
        .groupby(peer_df["group"])
        .transform("median")
    )

    # Global fallback for small peer groups
    global_median = float(
        np.nanmedian(log_amount)
    )

    global_mad = float(
        np.nanmedian(
            np.abs(log_amount - global_median)
        )
    )

    global_scale = max(
        1.4826 * global_mad,
        1e-6
    )

    use_global = (
        (group_size < 20)
        | (group_mad <= 1e-9)
        | group_mad.isna()
    )

    center = group_median.where(
        ~use_global,
        global_median
    )

    scale = (
        1.4826 * group_mad
    ).where(
        ~use_global,
        global_scale
    )

    robust_z = (
        (log_amount - center) / scale
    ).clip(lower=0.0)

    # 5 robust standard deviations -> score 1
    peer_score = (
        robust_z / 5.0
    ).clip(0.0, 1.0).fillna(0.0)

    # -------------------------------------------------
    # 5. Rolling velocity acceleration
    #
    # Uniform activity over 72h:
    # ~ 1/3 of transactions should fall in last 24h.
    #
    # Concentration toward the last 24h increases score.
    # -------------------------------------------------
    count_24h = (
        numeric("outgoing_count_24h")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    count_72h = (
        numeric("outgoing_count_72h")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    recent_share = (
        count_24h
        / count_72h.replace(0, np.nan)
    )

    velocity_score = (
        (recent_share - (1.0 / 3.0))
        / (2.0 / 3.0)
    ).clip(0.0, 1.0).fillna(0.0)

    # -------------------------------------------------
    # 6. Weighted behavioral risk score
    # -------------------------------------------------
    behavioral_scores = (
        0.30 * amount_drift_score
        + 0.30 * frequency_drift_score
        + 0.20 * peer_score
        + 0.20 * velocity_score
    )

    behavioral_scores = np.clip(
        behavioral_scores.to_numpy(
            dtype=np.float32
        ),
        0.0,
        1.0
    )

    # -------------------------------------------------
    # 7. Validation
    # -------------------------------------------------
    if len(behavioral_scores) != len(df):
        raise RuntimeError(
            "Behavioral score count does not match "
            "transaction count."
        )

    if not np.isfinite(
        behavioral_scores
    ).all():
        raise RuntimeError(
            "Behavioral model produced invalid scores."
        )

    print(
        f"✅ Behavioral model scored "
        f"{len(behavioral_scores):,} transactions."
    )

    print(
        "✅ Behavioral score range:",
        round(float(behavioral_scores.min()), 4),
        "→",
        round(float(behavioral_scores.max()), 4)
    )

    print(
        "✅ Mean behavioral score:",
        round(float(behavioral_scores.mean()), 4)
    )

    print(
        "Component means | "
        f"amount drift={amount_drift_score.mean():.3f}, "
        f"frequency drift={frequency_drift_score.mean():.3f}, "
        f"peer={peer_score.mean():.3f}, "
        f"velocity={velocity_score.mean():.3f}"
    )

    return {
        "ml_probabilities": {
            "behavioral_score":
                behavioral_scores.tolist()
        }
    }


# 4. Graph Context Node (cuGraph)
def graph_context_node(state: dict) -> dict:
    print("--- RUNNING SILENT-MULE GRAPH ANALYSIS ---")

    import numpy as np
    import pandas as pd

    # -------------------------------------------------
    # Configuration from Graph2_2B
    # -------------------------------------------------
    XGB_WEIGHT = 0.40
    ANOMALY_WEIGHT = 0.30
    BEHAVIORAL_WEIGHT = 0.30

    SEED_THRESHOLD = 0.75
    GRAPH_RECEIVING_PENALTY = 0.40
    INVESTIGATION_THRESHOLD = 0.50

    df = state["final_df"].copy()
    ml_probs = state.get("ml_probabilities", {})

    # -------------------------------------------------
    # 1. Retrieve model scores
    # -------------------------------------------------
    score_map = {
        "xgb_prob": ml_probs.get("xgb_prob"),
        "anomaly_score": ml_probs.get("anomaly_score"),
        "behavioral_score": ml_probs.get("behavioral_score"),
    }

    for score_name, values in score_map.items():

        if values is None:
            raise ValueError(
                f"Graph node missing model score: {score_name}"
            )

        if len(values) != len(df):
            raise ValueError(
                f"{score_name} length mismatch: "
                f"{len(values)} scores for {len(df)} transactions."
            )

        df[score_name] = (
            pd.to_numeric(
                pd.Series(values, index=df.index),
                errors="coerce"
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    # -------------------------------------------------
    # 2. Validate graph columns
    # -------------------------------------------------
    required = [
        "Account",
        "Account_duplicated_0",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Graph node missing required columns: {missing}"
        )

    # -------------------------------------------------
    # 3. Determine graph day
    #
    # Graph2_2B uses a DAILY directed edgelist.
    # -------------------------------------------------
    if "Transaction_Date" in df.columns:
        graph_dates = pd.to_datetime(
            df["Transaction_Date"],
            errors="coerce"
        )

    elif "Timestamp" in df.columns:
        graph_dates = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

    else:
        raise ValueError(
            "Graph node requires Transaction_Date or Timestamp."
        )

    if graph_dates.isna().any():
        raise ValueError(
            "Graph node found invalid transaction dates."
        )

    df["_graph_date"] = graph_dates.dt.normalize()

    # -------------------------------------------------
    # 4. Phase 1:
    # Independent ML base risk
    # -------------------------------------------------
    df["base_risk"] = (
        XGB_WEIGHT * df["xgb_prob"]
        + ANOMALY_WEIGHT * df["anomaly_score"]
        + BEHAVIORAL_WEIGHT * df["behavioral_score"]
    ).clip(
        lower=0.0,
        upper=1.0
    )

    # -------------------------------------------------
    # 5. Transaction -> account/day risk
    #
    # Same rule used in Graph2_2B:
    # strongest observed sender evidence = max
    # -------------------------------------------------
    sender_base_risk = (
        df
        .groupby(
            ["_graph_date", "Account"],
            dropna=True
        )["base_risk"]
        .max()
        .reset_index()
    )

    # Include accounts appearing only as receivers.
    sender_accounts = (
        df[["_graph_date", "Account"]]
        .dropna()
        .copy()
    )

    receiver_accounts = (
        df[
            [
                "_graph_date",
                "Account_duplicated_0"
            ]
        ]
        .dropna()
        .rename(
            columns={
                "Account_duplicated_0": "Account"
            }
        )
    )

    account_results = (
        pd.concat(
            [
                sender_accounts,
                receiver_accounts
            ],
            ignore_index=True
        )
        .drop_duplicates()
    )

    account_results = account_results.merge(
        sender_base_risk,
        on=["_graph_date", "Account"],
        how="left"
    )

    account_results["base_risk"] = (
        account_results["base_risk"]
        .fillna(0.0)
        .astype(float)
    )

    # -------------------------------------------------
    # 6. Phase 2:
    # High-risk seeds
    #
    # IMPORTANT:
    # Graph2_2B specifies > 0.75, not >= 0.75.
    # -------------------------------------------------
    account_results["is_high_risk_seed"] = (
        account_results["base_risk"]
        > SEED_THRESHOLD
    )

    seed_keys = (
        account_results.loc[
            account_results["is_high_risk_seed"],
            ["_graph_date", "Account"]
        ]
        .drop_duplicates()
    )

    # -------------------------------------------------
    # 7. Extract directed edges from ORIGINAL seeds
    # -------------------------------------------------
    seed_edges = df.merge(
        seed_keys.assign(_is_seed=True),
        on=["_graph_date", "Account"],
        how="inner"
    )

    # Self-transfer does not count as graph exposure.
    # Filter the edge itself, not the whole account.
    seed_edges = seed_edges[
        seed_edges["Account"]
        != seed_edges["Account_duplicated_0"]
    ].copy()

    # -------------------------------------------------
    # 8. Direct 1-hop exposed receivers
    # -------------------------------------------------
    exposed_keys = (
        seed_edges[
            [
                "_graph_date",
                "Account_duplicated_0"
            ]
        ]
        .dropna()
        .drop_duplicates()
        .rename(
            columns={
                "Account_duplicated_0": "Account"
            }
        )
    )

    exposed_keys[
        "received_from_high_risk"
    ] = True

    account_results = account_results.merge(
        exposed_keys,
        on=["_graph_date", "Account"],
        how="left"
    )

    account_results[
        "received_from_high_risk"
    ] = (
        account_results[
            "received_from_high_risk"
        ]
        .eq(True)
    )

    # -------------------------------------------------
    # 9. Strict one-time graph penalty
    #
    # No multiplication by:
    # - number of transfers
    # - number of seed accounts
    # - transaction amount
    # -------------------------------------------------
    account_results["graph_penalty"] = np.where(
        account_results["received_from_high_risk"],
        GRAPH_RECEIVING_PENALTY,
        0.0
    )

    # -------------------------------------------------
    # 10. Final account risk
    # -------------------------------------------------
    account_results["final_risk_score"] = (
        account_results["base_risk"]
        + account_results["graph_penalty"]
    ).clip(
        lower=0.0,
        upper=1.0
    )

    account_results["investigation_flag"] = (
        account_results["final_risk_score"]
        >= INVESTIGATION_THRESHOLD
    )

    # -------------------------------------------------
    # 11. Save A -> B exposure evidence
    # -------------------------------------------------
    evidence_cols = [
        "_graph_date",
        "Account",
        "Account_duplicated_0",
    ]

    if "Amount Paid" in seed_edges.columns:
        evidence_cols.append("Amount Paid")

    exposure_evidence = (
        seed_edges[evidence_cols]
        .rename(
            columns={
                "_graph_date": "graph_date",
                "Account": "high_risk_sender",
                "Account_duplicated_0":
                    "exposed_receiver",
            }
        )
        .copy()
    )

    # -------------------------------------------------
    # 12. Human-readable date column
    # -------------------------------------------------
    account_results = account_results.rename(
        columns={
            "_graph_date": "graph_date"
        }
    )

    # -------------------------------------------------
    # 13. Diagnostics
    # -------------------------------------------------
    print("========================================")
    print("FINDETECT SILENT-MULE GRAPH")
    print("========================================")

    print(
        f"Transactions evaluated : {len(df):,}"
    )

    print(
        f"Account/day nodes      : "
        f"{len(account_results):,}"
    )

    print(
        f"High-risk seeds > {SEED_THRESHOLD}: "
        f"{account_results['is_high_risk_seed'].sum():,}"
    )

    print(
        f"Direct 1-hop receivers : "
        f"{account_results['received_from_high_risk'].sum():,}"
    )

    print(
        f"Flagged >= {INVESTIGATION_THRESHOLD}: "
        f"{account_results['investigation_flag'].sum():,}"
    )

    # IMPORTANT:
    # No propagation from exposed B -> C.
    # Only original ML high-risk seeds generate exposure.

    return {
        "subgraph_data": {
            "account_results": account_results,
            "high_risk_seeds": (
                account_results.loc[
                    account_results["is_high_risk_seed"],
                    ["graph_date", "Account"]
                ]
                .to_dict(orient="records")
            ),
            "exposed_receivers": (
                account_results.loc[
                    account_results[
                        "received_from_high_risk"
                    ],
                    ["graph_date", "Account"]
                ]
                .to_dict(orient="records")
            ),
            "exposure_evidence": exposure_evidence,
            "seed_threshold": SEED_THRESHOLD,
            "graph_penalty": GRAPH_RECEIVING_PENALTY,
            "investigation_threshold":
                INVESTIGATION_THRESHOLD,
        }
    }


# 5. Risk Agent Node (Mathematical Weighting)
def risk_agent_node(state: dict) -> dict:
    print("--- RUNNING RISK AGENT ---")

    import numpy as np
    import pandas as pd

    graph_data = state.get("subgraph_data", {})

    account_results = graph_data.get("account_results")

    if account_results is None:
        raise ValueError(
            "Risk Agent did not receive account_results "
            "from the graph node."
        )

    if not isinstance(account_results, pd.DataFrame):
        raise TypeError(
            "subgraph_data['account_results'] "
            "must be a pandas DataFrame."
        )

    results = account_results.copy()

    # -------------------------------------------------
    # 1. Validate graph output
    # -------------------------------------------------
    required = [
        "graph_date",
        "Account",
        "base_risk",
        "graph_penalty",
        "final_risk_score",
        "investigation_flag",
    ]

    missing = [
        col for col in required
        if col not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Risk Agent missing graph outputs: {missing}"
        )

    # -------------------------------------------------
    # 2. Use graph final score directly
    #
    # IMPORTANT:
    # Do NOT recompute model weighting here.
    #
    # Graph node already calculated:
    #
    # base risk =
    # 0.40 XGB
    # + 0.30 Autoencoder
    # + 0.30 Behavioral
    #
    # final risk =
    # base risk + graph penalty
    # -------------------------------------------------
    results["final_risk_score"] = (
        pd.to_numeric(
            results["final_risk_score"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    investigation_threshold = float(
        graph_data.get(
            "investigation_threshold",
            0.50
        )
    )

    results["investigation_flag"] = (
        results["final_risk_score"]
        >= investigation_threshold
    )

    # -------------------------------------------------
    # 3. Operational risk labels
    #
    # High Risk = crosses investigation threshold
    # Medium Risk = 0.30 up to threshold
    # Low Risk = below 0.30
    # -------------------------------------------------
    results["risk_label"] = np.select(
        [
            results["investigation_flag"],
            results["final_risk_score"] >= 0.30,
        ],
        [
            "High Risk",
            "Medium Risk",
        ],
        default="Low Risk"
    )

    # -------------------------------------------------
    # 4. Accounts routed to investigation / LLM
    # -------------------------------------------------
    flagged_accounts = results[
        results["investigation_flag"]
    ].copy()

    high_count = (
        results["risk_label"]
        .eq("High Risk")
        .sum()
    )

    medium_count = (
        results["risk_label"]
        .eq("Medium Risk")
        .sum()
    )

    low_count = (
        results["risk_label"]
        .eq("Low Risk")
        .sum()
    )

    print(
        f"🚨 Risk Agent results | "
        f"High={high_count:,} | "
        f"Medium={medium_count:,} | "
        f"Low={low_count:,}"
    )

    print(
        f"✅ {len(flagged_accounts):,} account/day "
        f"cases routed for investigation."
    )

    return {
        "risk_scores": results,
        "final_insights": (
            flagged_accounts
            .to_dict(orient="records")
        )
    }


# 6. LLM API Node
def llm_api_node(state: dict) -> dict:
    print("--- BUILDING ACCOUNT/DAY SAR CASES ---")

    import uuid
    import pandas as pd

    alerts = state.get("final_insights", [])
    graph_data = state.get("subgraph_data", {})

    if not alerts:
        print(
            "✅ No investigation-level accounts detected. "
            "Skipping SAR generation."
        )
        return {"sar_reports": []}

    exposure_evidence = graph_data.get(
        "exposure_evidence"
    )

    if exposure_evidence is None:
        exposure_evidence = pd.DataFrame(
            columns=[
                "graph_date",
                "high_risk_sender",
                "exposed_receiver",
                "Amount Paid",
            ]
        )

    elif not isinstance(
        exposure_evidence,
        pd.DataFrame
    ):
        exposure_evidence = pd.DataFrame(
            exposure_evidence
        )

    if not exposure_evidence.empty:
        exposure_evidence = (
            exposure_evidence.copy()
        )

        exposure_evidence["graph_date"] = (
            pd.to_datetime(
                exposure_evidence["graph_date"],
                errors="coerce"
            )
            .dt.normalize()
        )

    print(
        f"--- GENERATING {len(alerts):,} "
        "ACCOUNT/DAY SAR CASE(S) ---"
    )

    alerts = sorted(
        alerts,
        key=lambda x: x.get(
            "final_risk_score",
            0.0
        ),
        reverse=True
    )

    sar_reports = []

    for alert in alerts:

        account = alert["Account"]

        graph_date = pd.to_datetime(
            alert["graph_date"],
            errors="coerce"
        )

        if pd.isna(graph_date):
            graph_date_str = "N/A"
            normalized_date = None
        else:
            normalized_date = graph_date.normalize()
            graph_date_str = (
                normalized_date
                .strftime("%Y-%m-%d")
            )

        # -----------------------------------------
        # 1. Build reason codes from actual case
        # -----------------------------------------
        reason_codes = []

        if bool(
            alert.get(
                "is_high_risk_seed",
                False
            )
        ):
            reason_codes.append(
                {
                    "code": "RC-ML-SEED",
                    "description":
                        "Account exceeded the "
                        "combined ML high-risk "
                        "seed threshold"
                }
            )

        if bool(
            alert.get(
                "received_from_high_risk",
                False
            )
        ):
            reason_codes.append(
                {
                    "code": "RC-GRAPH-1HOP",
                    "description":
                        "Account received funds "
                        "directly from an original "
                        "high-risk seed"
                }
            )

        if not reason_codes:
            reason_codes.append(
                {
                    "code": "RC-RISK-THRESHOLD",
                    "description":
                        "Account crossed the "
                        "operational investigation "
                        "risk threshold"
                }
            )

        # -----------------------------------------
        # 2. Collect real graph evidence
        # -----------------------------------------
        transactions = []

        if (
            normalized_date is not None
            and not exposure_evidence.empty
        ):

            same_day = (
                exposure_evidence[
                    "graph_date"
                ]
                == normalized_date
            )

            # Incoming exposure:
            # high-risk A -> current account B
            incoming = exposure_evidence[
                same_day
                & (
                    exposure_evidence[
                        "exposed_receiver"
                    ]
                    == account
                )
            ]

            for _, row in incoming.iterrows():

                amount = row.get(
                    "Amount Paid",
                    0.0
                )

                amount = (
                    0.0
                    if pd.isna(amount)
                    else float(amount)
                )

                transactions.append(
                    {
                        "timestamp":
                            graph_date_str,
                        "from_account":
                            row[
                                "high_risk_sender"
                            ],
                        "to_account":
                            row[
                                "exposed_receiver"
                            ],
                        "amount_sar":
                            amount,
                        "direction":
                            "inbound graph exposure",
                    }
                )

            # Outgoing evidence:
            # current account is original seed A
            outgoing = exposure_evidence[
                same_day
                & (
                    exposure_evidence[
                        "high_risk_sender"
                    ]
                    == account
                )
            ]

            for _, row in outgoing.iterrows():

                amount = row.get(
                    "Amount Paid",
                    0.0
                )

                amount = (
                    0.0
                    if pd.isna(amount)
                    else float(amount)
                )

                transactions.append(
                    {
                        "timestamp":
                            graph_date_str,
                        "from_account":
                            row[
                                "high_risk_sender"
                            ],
                        "to_account":
                            row[
                                "exposed_receiver"
                            ],
                        "amount_sar":
                            amount,
                        "direction":
                            "outbound seed exposure",
                    }
                )

        # -----------------------------------------
        # 3. Build structured account/day case
        # -----------------------------------------
        structured_case = {
            "case_id": (
                f"CASE-"
                f"{uuid.uuid4().hex[:8].upper()}"
            ),

            "customer": {
                "customer_id": account,
                "name_masked":
                    "Redacted Entity",
                "segment":
                    "Transaction Monitoring Batch"
            },

            "risk_score": round(
                float(
                    alert[
                        "final_risk_score"
                    ]
                ),
                6
            ),

            "risk_tier": (
                alert.get(
                    "risk_label",
                    "High Risk"
                )
                .upper()
                .replace(" ", "_")
            ),

            "reason_codes":
                reason_codes,

            "model_scores": {
                "combined_base_risk":
                    round(
                        float(
                            alert.get(
                                "base_risk",
                                0.0
                            )
                        ),
                        6
                    )
            },

            "graph_features": {
                "graph_date":
                    graph_date_str,

                "graph_penalty":
                    round(
                        float(
                            alert.get(
                                "graph_penalty",
                                0.0
                            )
                        ),
                        6
                    ),

                "is_high_risk_seed":
                    bool(
                        alert.get(
                            "is_high_risk_seed",
                            False
                        )
                    ),

                "received_from_high_risk":
                    bool(
                        alert.get(
                            "received_from_high_risk",
                            False
                        )
                    ),

                "investigation_flag":
                    bool(
                        alert.get(
                            "investigation_flag",
                            False
                        )
                    ),
            },

            "transactions":
                transactions,
        }

        try:
            generated_text = generate_report(
                structured_case
            )

        except Exception as e:
            print(
                f"⚠️ SAR generation failed "
                f"for {account}: {e}"
            )

            generated_text = (
                "SYSTEM ERROR: "
                "Failed to generate SAR. "
                f"{e}"
            )

        sar_reports.append(
            generated_text
        )

    print(
        f"✅ Generated "
        f"{len(sar_reports):,} "
        "account/day SAR drafts."
    )

    return {
        "sar_reports": sar_reports,
        "final_insights": alerts,
    }


# 7. Report Node
def report_node(state: dict) -> dict:
    print("--- FINALIZING REPORT ---")
    existing_insights = state.get("final_insights", [])    
    return {
        "final_insights": existing_insights
    }
