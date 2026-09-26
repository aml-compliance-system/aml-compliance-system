from langgraph.graph import StateGraph, END

# Import the correctly defined state with the merge_dicts reducer
from AMLPipelineState import AMLPipelineState

from nodes import (
    preprocessing_node,
    customer_deviation_node,    
    run_xgboost_model,
    run_anomaly_model,
    run_behavioral_model,
    graph_context_node,
    meta_learner_node,         
    risk_agent_node,
    llm_api_node,
    report_node,
    evaluation_node,
)

# ✅ CORRECT: Initialize with your custom state (DO NOT overwrite this with 'dict')
workflow = StateGraph(AMLPipelineState)

# ============================================================================
# 1. ADD NODES
# ============================================================================
workflow.add_node("preprocess", preprocessing_node)
workflow.add_node("customer_deviation", customer_deviation_node)   
workflow.add_node("xgb_model", run_xgboost_model)
workflow.add_node("anomaly_model", run_anomaly_model)
workflow.add_node("behavioral_model", run_behavioral_model)
workflow.add_node("graph_extraction", graph_context_node)
workflow.add_node("meta_learner", meta_learner_node)               
workflow.add_node("risk_agent", risk_agent_node)
workflow.add_node("llm_api", llm_api_node)
workflow.add_node("report", report_node)
workflow.add_node("evaluation", evaluation_node)

# ============================================================================
# 2. SET ENTRY POINT
# ============================================================================
workflow.set_entry_point("preprocess")

# ============================================================================
# 3. DEFINE EDGES (Parallel Fan-Out / Fan-In)
# ============================================================================

# Step 1: Preprocess → Customer Deviation (sequential, must run first)
workflow.add_edge("preprocess", "customer_deviation")              

# Step 2: Customer Deviation → Fan-out to all 3 models (parallel execution)
workflow.add_edge("customer_deviation", "xgb_model")               
workflow.add_edge("customer_deviation", "anomaly_model")           
workflow.add_edge("customer_deviation", "behavioral_model")        

# Step 3: Fan-in from all 3 models → Graph Extraction
workflow.add_edge("xgb_model", "graph_extraction")
workflow.add_edge("anomaly_model", "graph_extraction")
workflow.add_edge("behavioral_model", "graph_extraction")

# Step 4: Graph Extraction → Meta-Learner → Risk Agent
workflow.add_edge("graph_extraction", "meta_learner")              
workflow.add_edge("meta_learner", "risk_agent")                    

# Step 5: Sequential finish & evaluation
workflow.add_edge("risk_agent", "llm_api")
workflow.add_edge("llm_api", "report")
workflow.add_edge("report", "evaluation")
workflow.add_edge("evaluation", END)

# ============================================================================
# 4. COMPILE
# ============================================================================
app = workflow.compile()

print("✅ AML Pipeline compiled successfully with parallel execution and meta-learner!")



if __name__ == "__main__":
    print("🚀 Starting FINDETECT AML Pipeline...")
    
    # 1. Define the starting state
    initial_state = {
        "final_parquet_path": "input/final_transactions_10k.parquet"
    }
    
    # 2. Execute the workflow
    final_state = app.invoke(initial_state)
    
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
    
    # 3. Print Account/Day Risk Results
    if "risk_scores" in final_state:

        df_results = final_state["risk_scores"][
            [
                "graph_date",
                "Account",
                "base_risk",
                "graph_penalty",
                "final_risk_score",
                "is_high_risk_seed",
                "received_from_high_risk",
                "investigation_flag",
                "risk_label",
            ]
        ].copy()

        df_results = df_results.sort_values(
            "final_risk_score",
            ascending=False
        )

        high_risk_sample = df_results[
            df_results["risk_label"] == "High Risk"
        ]

        if not high_risk_sample.empty:

            print(
                f"\n🚨 Found "
                f"{len(high_risk_sample):,} "
                "High Risk account/day cases."
            )

            print("\n--- TOP HIGH-RISK CASES ---")

            print(
                high_risk_sample
                .head(10)
                .to_string(index=False)
            )

        else:

            print(
                "\n✅ No accounts crossed the "
                "investigation threshold."
            )

            print(
                df_results
                .head(10)
                .to_string(index=False)
            )

    else:
        print(
            "❌ Failed to retrieve risk scores."
        )

    # 4. Print SAR Report Summary
    reports = final_state.get(
        "sar_reports",
        []
    )

    if reports:

        print(
            f"\n📝 Generated "
            f"{len(reports):,} "
            "SAR draft(s)."
        )

        print(
            "\n--- SAMPLE SAR REPORT ---"
        )

        print(reports[0])

        print(
            "-------------------------\n"
        )

    else:

        print(
            "\\n✅ Pipeline finished safely. "
            "No account/day cases crossed "
            "the 0.50 investigation threshold."
        )
