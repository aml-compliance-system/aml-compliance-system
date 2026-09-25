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
workflow.add_edge(
    [
        "xgb_model",
        "anomaly_model",
        "behavioral_model"
    ],
    "graph_extraction"
)


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
                f"\\n🚨 Found "
                f"{len(high_risk_sample):,} "
                "High Risk account/day cases."
            )

            print("\\n--- TOP HIGH-RISK CASES ---")

            print(
                high_risk_sample
                .head(10)
                .to_string(index=False)
            )

        else:

            print(
                "\\n✅ No accounts crossed the "
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
            f"\\n📝 Generated "
            f"{len(reports):,} "
            "SAR draft(s)."
        )

        print(
            "\\n--- SAMPLE SAR REPORT ---"
        )

        print(reports[0])

        print(
            "-------------------------\\n"
        )

    else:

        print(
            "\\n✅ Pipeline finished safely. "
            "No account/day cases crossed "
            "the 0.50 investigation threshold."
        )
