import pandas as pd
import numpy as np

# 1. Preprocessing Node (ETL)
def preprocessing_node(state: dict) -> dict:
    print("--- RUNNING PREPROCESSING (ETL) ---")
    
    # For now, let's pretend we loaded a CSV
    # raw_df = pd.read_csv(state["raw_csv_path"])
    
    # Placeholder cleaned dataframe
    cleaned_df = pd.DataFrame({
        "Amount Received": [100.0, 5000.0],
        "Amount Paid": [100.0, 5000.0],
        "hour_of_day": [14, 3]
    })
    
    return {"cleaned_df": cleaned_df}

# 2. XGBoost Model Node
def run_xgboost_model(state: dict) -> dict:
    print("--- RUNNING XGBOOST MODEL ---")
    # Here is where you will load your saved xgboost_aml_model.json and predict
    probabilities = {"xgb_prob": [0.12, 0.95]}
    return {"ml_probabilities": probabilities}

# 3. Anomaly Model Node
def run_anomaly_model(state: dict) -> dict:
    print("--- RUNNING ANOMALY DETECTION ---")
    anomaly_scores = {"anomaly_score": [0.05, 0.88]}
    return {"ml_probabilities": anomaly_scores}

# 4. Graph Context Node (cuGraph)
def graph_context_node(state: dict) -> dict:
    print("--- EXTRACTING GRAPH SUBGRAPHS ---")
    subgraphs = {"connected_accounts": ["acc_1", "acc_2"]}
    return {"subgraph_data": subgraphs}

# 5. Risk Agent Node (Mathematical Weighting)
def risk_agent_node(state: dict) -> dict:
    print("--- RUNNING RISK AGENT MATH WEIGHTING ---")
    # This is where your 0.3 + 0.3 + 0.2 logic goes!
    scored_df = state["cleaned_df"].copy()
    scored_df["final_risk_score"] = 0.85 # Placeholder score
    return {"risk_scores": scored_df}

# 6. LLM API Node
def llm_api_node(state: dict) -> dict:
    print("--- CALLING LLM FOR SAR GENERATION ---")
    sar_text = "SAR Report: Account engaged in high-velocity structuring..."
    return {"sar_reports": [sar_text]}

# 7. Report Node
def report_node(state: dict) -> dict:
    print("--- FINALIZING REPORT ---")
    return {"final_insights": [{"status": "Completed", "alerts": 1}]}