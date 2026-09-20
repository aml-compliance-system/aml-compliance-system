import pandas as pd
import numpy as np
import duckdb
import xgboost as xgb
import pandas as pd
import os



def preprocessing_node(state: dict) -> dict:
    print("--- RUNNING PREPROCESSING (ETL) ---")
    
    csv_path = state.get("raw_csv_path", "input/unlabeled_transactions.csv")
    
    con = duckdb.connect()


    query = f"""
   WITH raw_batch AS (
        SELECT *,
            CAST("Timestamp" AS TIMESTAMP) AS ts,
            CAST("Amount Received" AS FLOAT) AS amt_rec,
            CAST("Amount Paid" AS FLOAT) AS amt_paid
        FROM read_csv_auto('{csv_path}')
    ),
    engineered AS (
        SELECT 
            *,
            -- Time & Categorical features
            CAST(date_part('hour', ts) AS FLOAT) AS hour_of_day,
            CAST("Receiving Currency" != "Payment Currency" AS FLOAT) AS currency_mismatch,
            CAST("From Bank" = "To Bank" AS FLOAT) AS same_bank,
            CAST(hash("Payment Format") % 10000000 AS FLOAT) AS payment_format_hash,
            
            -- In-Batch Graph & Rolling Features
            CAST(COUNT(*) OVER (PARTITION BY "Account") AS FLOAT) AS sender_outgoing_count_so_far_wk,
            CAST(COUNT(DISTINCT "Account_duplicated_0") OVER (PARTITION BY "Account") AS FLOAT) AS sender_unique_receivers_so_far_wk,
            CAST(COUNT(*) OVER (PARTITION BY "Account") AS FLOAT) AS sender_out_degree,
            CAST(COUNT(*) OVER (PARTITION BY "Account_duplicated_0") AS FLOAT) AS receiver_in_degree,
            CAST(COUNT(DISTINCT "Account_duplicated_0") OVER (PARTITION BY "Account") AS FLOAT) AS sender_unique_counterparties,
            CAST(COUNT(DISTINCT "Account") OVER (PARTITION BY "Account_duplicated_0") AS FLOAT) AS receiver_unique_counterparties,
            
            -- 24h and 72h rolling estimates inside batch
            CAST(COUNT(*) OVER (
                PARTITION BY "Account" 
                ORDER BY ts 
                RANGE BETWEEN INTERVAL '24' HOUR PRECEDING AND CURRENT ROW
            ) AS FLOAT) AS outgoing_count_24h,
            
            CAST(SUM(amt_paid) OVER (
                PARTITION BY "Account" 
                ORDER BY ts 
                RANGE BETWEEN INTERVAL '24' HOUR PRECEDING AND CURRENT ROW
            ) AS FLOAT) AS outgoing_sum_24h,

            CAST(COUNT(*) OVER (
                PARTITION BY "Account" 
                ORDER BY ts 
                RANGE BETWEEN INTERVAL '72' HOUR PRECEDING AND CURRENT ROW
            ) AS FLOAT) AS outgoing_count_72h,

            CAST(SUM(amt_paid) OVER (
                PARTITION BY "Account" 
                ORDER BY ts 
                RANGE BETWEEN INTERVAL '72' HOUR PRECEDING AND CURRENT ROW
            ) AS FLOAT) AS outgoing_sum_72h

        FROM raw_batch
    )
    SELECT * FROM engineered
    """
    
    df = con.execute(query).df()
    
    # 2. Historical features (If running against a local sample without past-week tables, 
    # fill remaining non-batch historical columns with column-wise defaults or medians)
    remaining_historical_cols = [
        "sender_in_degree", "receiver_out_degree", "sender_receiver_frequency", 
        "bidirectional_relationship", "sender_total_sent_prev_week", 
        "sender_avg_sent_prev_week", "sender_max_sent_prev_week", 
        "sender_outgoing_count_prev_week", "receiver_total_received_prev_week", 
        "receiver_avg_received_prev_week", "receiver_max_received_prev_week", 
        "receiver_incoming_count_prev_week", "amount_vs_prev_max_diff", 
        "amount_vs_prev_max_ratio", "transaction_frequency_diff", 
        "transaction_frequency_ratio", "hours_since_last_incoming", 
        "outgoing_to_incoming_ratio"
    ]
    
    for col in remaining_historical_cols:
        if col not in df.columns:
            df[col] = 0.0

    print(f"✅ Real-time ETL complete. Calculated batch rolling aggregates for {len(df)} transactions.")
    return {"cleaned_df": df}





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
    
    df = state["cleaned_df"]
    

    model_path = "models/final_xgboost_aml_model.json" 
    
    # Initialize the XGBoost Booster
    model = xgb.Booster()
    
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"Error loading model at {model_path}: {e}")
        print("Fallback: Outputting 0.0 probabilities to keep pipeline moving for local testing.")
        # Retrieve existing dict or create new one
        ml_probs = state.get("ml_probabilities", {})
        ml_probs["xgb_prob"] = [0.0] * len(df)
        return {"ml_probabilities": ml_probs}
        
    # 2. Extract features in the strict order required by XGBoost
    X = df[FEATURE_COLS].copy()
    
# 3. Create the DMatrix and run inference
    dmatrix = xgb.DMatrix(X)
    raw_probabilities = model.predict(dmatrix)
    
    # NORMALIZATION 
    probs_array = np.array(raw_probabilities)
    
    min_p = np.min(probs_array)
    max_p = np.max(probs_array)
    
    # Scale from 0.0 to 1.0 (where the highest risk in the batch becomes exactly 1.0)
    if max_p > min_p:
        normalized_probs = (probs_array - min_p) / (max_p - min_p)
    else:
        normalized_probs = np.zeros_like(probs_array) 
        
    # Optional: If you want XGBoost to contribute a MAXIMUM of 0.3 to your final score
    # xgb_risk_weights = normalized_probs * 0.3
    
    print(f"✅ XGBoost scored {len(normalized_probs)} transactions.")
    
    ml_probs = state.get("ml_probabilities", {})
    ml_probs["xgb_prob"] = normalized_probs.tolist()  
    return {"ml_probabilities": ml_probs}







# 3. Anomaly Model Node
def run_anomaly_model(state: dict) -> dict:
    print("--- RUNNING ANOMALY DETECTION ---")
    df = state["cleaned_df"]
    anomaly_scores = [0.0] * len(df)
    return {"ml_probabilities": {"anomaly_score": anomaly_scores}}


def run_behavioral_model(state: dict) -> dict:
    print("--- RUNNING BEHAVIORAL MODEL ---")
    df = state["cleaned_df"]
    behavioral_scores = [0.0] * len(df)
    return {"ml_probabilities": {"behavioral_score": behavioral_scores}}





# 4. Graph Context Node (cuGraph)
def graph_context_node(state: dict) -> dict:
    print("--- EXTRACTING TARGETED GRAPH SUBGRAPHS ---")
    df = state["cleaned_df"].copy()
    ml_probs = state.get("ml_probabilities", {})
    
    # 1. Read the ML scores that just finished running
    xgb_probs = ml_probs.get("xgb_prob", [0.0] * len(df))
    df['temp_ml_score'] = xgb_probs
    
    # 2. Identify the "Account As" (The Orchestrators)
    # Anyone with a base ML score over 0.75 is considered highly suspicious
    high_risk_senders = df[df['temp_ml_score'] >= 0.75]['Account'].unique()
    
    # 3. Find the "Account Bs" (The Silent Mules)
    # Who did the high-risk orchestrators send money to in this batch?
    mules_df = df[df['Account'].isin(high_risk_senders)]
    exposed_receivers = mules_df['Account_duplicated_0'].unique()
    
    # 4. Generate the Graph Penalty Array for the entire batch
    # If a row's receiver is in the exposed_receivers list, they get the +0.40 penalty
    graph_penalties = []
    for receiver in df['Account_duplicated_0']:
        if receiver in exposed_receivers:
            graph_penalties.append(0.40) 
        else:
            graph_penalties.append(0.0)
            
    print(f"Graph extracted {len(high_risk_senders)} high-risk nodes and penalized {sum(p > 0 for p in graph_penalties)} exposed connections.")
    
    return {"subgraph_data": {"penalties": graph_penalties}}

# 5. Risk Agent Node (Mathematical Weighting)
def risk_agent_node(state: dict) -> dict:
    print("--- RUNNING RISK AGENT MATH WEIGHTING ---")
    
    df = state["cleaned_df"].copy()
    ml_probs = state.get("ml_probabilities", {})
    
    # 1. Retrieve the probabilities (using zeroes if a model failed/skipped)
    xgb_probs = ml_probs.get("xgb_prob", [0.0] * len(df))
    anomaly_probs = ml_probs.get("anomaly_score", [0.0] * len(df)) 
    behavioral_probs = ml_probs.get("behavioral_score", [0.0] * len(df))
    
    # Retrieve Graph Penalties calculated sequentially after the ML models
    graph_penalties = state.get("subgraph_data", {}).get("penalties", [0.0] * len(df))
    
    # 2. Calculate Base ML Score (Out of 1.0)
    # Blend the 3 models (e.g., 40% XGBoost, 30% Anomaly, 30% Behavioral)
    base_ml_scores = [
        (x * 0.40) + (a * 0.30) + (b * 0.30) 
        for x, a, b in zip(xgb_probs, anomaly_probs, behavioral_probs)
    ]
    
    # 3. Add Graph Penalty & Cap at 1.0
    # If the Silent Mule has a 0.15 ML Base + 0.40 Graph Penalty = 0.55 Final Score.
    # If the Orchestrator has a 0.85 ML Base + 0.00 Graph Penalty = 0.85 Final Score.
    final_scores = [
        min(base + graph, 1.0) 
        for base, graph in zip(base_ml_scores, graph_penalties)
    ]
    
    df['final_risk_score'] = final_scores
    
    # 4. Assign Risk Labels based on your 0.55 Threshold
    def assign_label(score):
        if score >= 0.55:
            return "High Risk"
        elif score >= 0.30:
            return "Medium Risk"
        else:
            return "Low Risk"
            
    df['risk_label'] = df['final_risk_score'].apply(assign_label)
    
    # 5. Filter for High-Risk accounts to push to the LLM API
    high_risk_df = df[df['risk_label'] == "High Risk"].copy()
    
    print(f"🚨 Risk Agent assigned labels: {len(high_risk_df)} High Risk, "
          f"{len(df[df['risk_label'] == 'Medium Risk'])} Medium Risk, "
          f"{len(df[df['risk_label'] == 'Low Risk'])} Low Risk.")
    
    # Keep only the essential columns for the LLM
    alert_cols = ["Account", "Account_duplicated_0", "Amount Paid", "Payment Format", "final_risk_score", "risk_label"]
    risk_alerts = high_risk_df[alert_cols].to_dict(orient="records") if not high_risk_df.empty else []
    
    # Pass the fully labeled dataframe and the filtered alerts forward
    return {"risk_scores": df, "final_insights": risk_alerts}



# 6. LLM API Node
def llm_api_node(state: dict) -> dict:
    alerts = state.get("final_insights", [])
    sar_reports = []

    # If the Risk Agent found no high-risk transactions, skip the LLM entirely
    if not alerts:
        print("✅ No high-risk transactions detected. Skipping SAR generation.")
        return {"sar_reports": []}

    print(f"--- CALLING LLM TO GENERATE {len(alerts)} SAR(s) ---")
    
    for alert in alerts:
        # 1. Structure the prompt with the exact data points from the ETL and Model nodes
        prompt = f"""
        Generate a Suspicious Activity Report (SAR) narrative for the following transaction:
        - Originating Account: {alert['Account']}
        - Counterparty: {alert['Account_duplicated_0']}
        - Amount: ${alert['Amount Paid']:.2f}
        - Payment Format: {alert['Payment Format']}
        - Aggregate Risk Score: {alert['final_risk_score']:.2f}

        Please provide a concise, professional summary for a compliance investigator.
        """
        
        # 2. Execute the LLM Call 
        # (Placeholder: Replace this string with actual response = llm.invoke(prompt))
        generated_text = (
            f"CONFIDENTIAL SAR DRAFT: Account {alert['Account']} has been flagged for a highly suspicious "
            f"{alert['Payment Format']} transfer of ${alert['Amount Paid']:.2f} to {alert['Account_duplicated_0']}. "
            f"The autonomous risk engine assigned a critical severity score of {alert['final_risk_score']:.2f} due to "
            f"anomalous network topology and behavioral drift. Recommend immediate asset freeze and investigator review."
        )
        
        sar_reports.append(generated_text)

    print(f"✅ Successfully generated {len(sar_reports)} SAR drafts.")
    return {"sar_reports": sar_reports}


# 7. Report Node .
def report_node(state: dict) -> dict:
    print("--- FINALIZING REPORT ---")
    return {"final_insights": [{"status": "Completed", "alerts": 1}]}