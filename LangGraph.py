from langgraph.graph import StateGraph, END
from AMLPipelineState import AMLPipelineState
from  nodes import preprocessing_node,run_xgboost_model,run_anomaly_model,graph_context_node,risk_agent_node,llm_api_node,report_node

workflow = StateGraph(AMLPipelineState)

# 1. Add Nodes
workflow.add_node("preprocess", preprocessing_node)
workflow.add_node("xgb_model", run_xgboost_model)
workflow.add_node("anomaly_model", run_anomaly_model)
workflow.add_node("graph_extraction", graph_context_node)
workflow.add_node("risk_agent", risk_agent_node)
workflow.add_node("llm_api", llm_api_node)
workflow.add_node("report", report_node)

# 2. Define Edges & Parallel Flow
workflow.set_entry_point("preprocess")

# Fan-out: Preprocessing feeds into both models and graph context simultaneously
workflow.add_edge("preprocess", "xgb_model")
workflow.add_edge("preprocess", "anomaly_model")
workflow.add_edge("preprocess", "graph_extraction")

# Fan-in: Models and Graph feed into the Risk Agent
workflow.add_edge("xgb_model", "risk_agent")
workflow.add_edge("anomaly_model", "risk_agent")
workflow.add_edge("graph_extraction", "risk_agent")

# Sequential finish
workflow.add_edge("risk_agent", "llm_api")
workflow.add_edge("llm_api", "report")
workflow.add_edge("report", END)

app = workflow.compile()