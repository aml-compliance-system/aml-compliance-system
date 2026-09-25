from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from openai import OpenAI
import ollama
from LangGraph import app as langgraph_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=120.0,
)

CASE_CACHE = {}

class PipelineRequest(BaseModel):
    csv_path: str = "input/unlabeled_transactions.csv"

@app.post("/pipeline/analyze")
def analyze(request: PipelineRequest):
    try:
        initial_state = {
            "raw_csv_path": request.csv_path
        }
        
        final_state = langgraph_app.invoke(initial_state)
        alerts = final_state.get("final_insights", [])
        reports = final_state.get("sar_reports", [])
        
        CASE_CACHE.clear()
        for alert, report in zip(alerts, reports):
            account_id = alert.get('Account')
            if account_id:
                CASE_CACHE[account_id] = {
                    "alert": alert,
                    "report": report
                }
        
        return {
            "status": "success",
            "high_risk_alerts_found": len(alerts),
            "case_data": alerts,
            "reports": reports
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    case_id: str
    message: str

import json
import pandas as pd

@app.post("/chat")
def chat_with_case(request: ChatRequest):
    try:
        case_info = CASE_CACHE.get(request.case_id)
        
        if not case_info:
            context_text = f"Target Account ID: {request.case_id} (No telemetry cache found.)"
        else:
            raw_alert = case_info.get('alert', {})
            essential_keys = [
                'Account', 'Account_duplicated_0', 'Amount Paid', 'Amount Received',
                'final_risk_score', 'risk_label', 'base_risk', 'graph_penalty',
                'is_high_risk_seed', 'received_from_high_risk'
            ]
            slim_alert = {k: raw_alert[k] for k in essential_keys if k in raw_alert}
            alert_json = json.dumps(slim_alert, indent=2, default=str)
            
            context_text = (
                f"=== CASE METRICS ===\n{alert_json}\n\n"
                f"=== INITIAL SAR REPORT ===\n{case_info.get('report', 'No report available.')}"
            )

        system_instruction = (
            "You are an expert Anti-Money Laundering (AML) Compliance Investigator. "
            "Be direct, concise, and professional. Base answers strictly on the provided case telemetry."
        )

        user_content = (
            f"[CASE CONTEXT]\n{context_text}\n\n"
            f"[INVESTIGATOR QUERY]\n{request.message}"
        )

        # Native Ollama call (skips OpenAI proxy overhead entirely)
        response = ollama.chat(
            model='qwen2.5',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_content}
            ],
            options={'temperature': 0.1, 'num_predict': 400}
        )

        ai_reply = response['message']['content']

        return {
            "message": "success",
            "case_id": request.case_id,
            "response": ai_reply
        }

    except Exception as e:
        print(f"❌ CHAT ENDPOINT ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/case/{case_id}")
def get_case_details(case_id: str):
    if case_id in CASE_CACHE:
        return CASE_CACHE[case_id]
    

    for acc_id, data in CASE_CACHE.items():
        if acc_id == case_id or data['alert'].get('Account') == case_id:
            return data
            
    raise HTTPException(status_code=404, detail="Case telemetry not found. Please re-run the pipeline from the dashboard.")