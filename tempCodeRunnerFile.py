    if "risk_scores" in final_state and "final_insights" in final_state:
        risk=final_state["risk_scores"]
        print("Risk score:",risk[:5])
    else:
        print("failed")
    # Check if the LLM generated SARs
    if "sar_reports" in final_state and final_state["sar_reports"]:
        reports = final_state["sar_reports"]
        print(f"\n✅ LLM Success: Generated {len(reports)} SAR drafts.")
        print("\n--- SAMPLE SAR REPORT ---")
        print(reports[0])
        print("-------------------------\n")
    else:
        print("✅ Pipeline finished safely with no high-risk alerts to report.")