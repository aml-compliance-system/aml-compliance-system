from langgraph.graph import StateGraph, END
from AMLPipelineState import AMLPipelineState
from  nodes import (preprocessing_node,run_xgboost_model,run_anomaly_model,
                    graph_context_node,risk_agent_node,
                    llm_api_node,report_node,run_behavioral_model)

workflow = StateGraph(AMLPipelineState)

# 1. Add Nodes
workflow.add_node("preprocess", preprocessing_node)
workflow.add_node("xgb_model", run_xgboost_model)
workflow.add_node("anomaly_model", run_anomaly_model)
workflow.add_node("behavioral_model", run_behavioral_model)

workflow.add_node("graph_extraction", graph_context_node)
workflow.add_node("risk_agent", risk_agent_node)
workflow.add_node("llm_api", llm_api_node)
workflow.add_node("report", report_node)


# 2. Define Edges & Parallel Flow
workflow.set_entry_point("preprocess")

# Fan-out: Preprocessing feeds into both models and graph context simultaneously
workflow.add_edge("preprocess", "xgb_model")
workflow.add_edge("preprocess", "anomaly_model")
workflow.add_edge("preprocess", "behavioral_model")


# Fan-in: Models and Graph feed into the Risk Agent
workflow.add_edge("xgb_model", "graph_extraction")
workflow.add_edge("anomaly_model", "graph_extraction")
workflow.add_edge("behavioral_model", "graph_extraction")


# Sequential finish
workflow.add_edge("graph_extraction", "risk_agent")
workflow.add_edge("risk_agent", "llm_api")
workflow.add_edge("llm_api", "report")
workflow.add_edge("report", END)

app = workflow.compile()





if __name__ == "__main__":
    print("🚀 Starting FINDETECT AML Pipeline...")
    
    # 1. Define the starting state
    initial_state = {
        "raw_csv_path": "input/unlabeled_transactions.csv"
    }
    
    # 2. Execute the workflow
    final_state = app.invoke(initial_state)
    
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    
    # 3. Print Risk Scores & Labels
    if "risk_scores" in final_state and "final_insights" in final_state:
        # Extracting the key columns to display cleanly
        df_results = final_state["risk_scores"][["Account", "Account_duplicated_0", "final_risk_score", "risk_label"]]
        
        # Print a sample of High Risk accounts if any exist, otherwise just print the top 5
        high_risk_sample = df_results[df_results['risk_label'] == 'High Risk']
        
        if not high_risk_sample.empty:
            print(f"\n🚨 Found {len(high_risk_sample)} High Risk Alerts. Showing sample:")
            print(high_risk_sample.head(5).to_string(index=False))
        else:
            print("\n✅ No High Risk alerts found. Showing first 5 transactions in batch:")
            print(df_results.head(5).to_string(index=False))
    else:
        print("❌ Failed to retrieve risk scores.")
        
    # 4. Print LLM SAR Reports
    if "sar_reports" in final_state and final_state["sar_reports"]:
        reports = final_state["sar_reports"]
        print(f"\n📝 LLM Success: Generated {len(reports)} SAR drafts.")
        print("\n--- SAMPLE SAR REPORT ---")
        print(reports[0])
        print("-------------------------\n")
    else:
        print("\n✅ Pipeline finished safely. No SARs generated (no transactions crossed the 0.55 threshold).")