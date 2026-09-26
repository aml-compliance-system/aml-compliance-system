from typing import TypedDict, List, Dict, Any, Annotated
import pandas as pd


def merge_dicts(left: dict, right: dict) -> dict:
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}


class AMLPipelineState(TypedDict):
    # Input: path to the final feature-engineered parquet dataset
    final_parquet_path: str

    # Final dataset used by all models
    final_df: pd.DataFrame

    # Outputs from XGBoost, Autoencoder, and Behavioral models
    ml_probabilities: Annotated[Dict[str, Any], merge_dicts]

    # Output of Graph Extraction Node
    subgraph_data: Dict[str, Any]

    # Output of Risk Agent
    risk_scores: pd.DataFrame

    # Output of LLM API Node
    final_insights: List[Dict[str, Any]]

    # Final generated SAR reports
    sar_reports: List[str]

    evaluation_metrics: Dict[str, Any]
