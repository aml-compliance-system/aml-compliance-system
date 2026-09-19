from typing import TypedDict, List, Dict, Any
import pandas as pd

class AMLPipelineState(TypedDict):
    raw_csv_path: str                 # Input: path to raw daily CSV batch
    cleaned_df: pd.DataFrame          # Output of Preprocessing Node
    ml_probabilities: Dict[str, Any]  # Output of Parallel Models Node (XGBoost, Anomaly, etc.)
    subgraph_data: Dict[str, Any]     # Output of Graph Extraction Node
    risk_scores: pd.DataFrame         # Output of Risk Agent (math weighting)
    final_insights: List[Dict[str, Any]] # Output of LLM API Node
    sar_reports: List[str]            # Final Output: Generated SAR files