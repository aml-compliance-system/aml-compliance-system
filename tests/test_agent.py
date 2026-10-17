"""Unit tests for the tool-using investigator loop (no Ollama required)."""

from __future__ import annotations

import json

import pytest

import agent as agent_module
from agent import AMLInvestigatorAgent


class FakeClient:
    """Minimal stand-in for ``ollama.Client`` with scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def tool_call(name: str, **arguments):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        }
    }


def final(text: str):
    return {"message": {"role": "assistant", "content": text}}


def make_agent(store, responses, **kwargs):
    client = FakeClient(responses)
    return AMLInvestigatorAgent(store=store, client=client, model="fake", **kwargs), client


def test_agent_calls_model_tools_and_answers(store):
    ag, client = make_agent(
        store,
        [
            tool_call("get_transactions", account_id="A1", date="2022-01-01"),
            final("Score 0.90 on 2022-01-01; see evidence."),
        ],
    )
    result = ag.chat("A1", "Why was this account flagged?", case_date="2022-01-01")

    assert result.answer.startswith("Score 0.90")
    assert result.steps == 2
    assert result.degraded is False
    assert len(client.calls) == 2
    assert client.calls[0]["tools"] == agent_module.TOOL_SCHEMAS
    names = [call.name for call in result.tool_calls]
    assert names.count("get_transactions") == 2  # prefetched + model-requested
    assert "get_graph_exposure" in names
    # Model-requested calls carry source="model"; prefetch carries "auto".
    assert any(call.source == "model" for call in result.tool_calls)

    payload = result.to_dict()
    assert payload["response"] == result.answer
    json.dumps(payload, allow_nan=False)


def test_prefetch_uses_requested_case_date(store):
    ag, client = make_agent(store, [final("answer")])
    ag.chat("A1", "status?", case_date="2022-01-01")

    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][-1]["content"]
    assert "Case date: 2022-01-01" in system
    assert "EVIDENCE PACK (prefetched" in user
    assert "CASE DATE: 2022-01-01" in user
    assert "final_risk_score=0.900" in user
    assert "inbound_total=150.00" in user
    assert "outbound_total=300.00" in user
    assert "ANSWER CHECK: inbound_total=150.00 is money RECEIVED" in user
    assert "SCORE NOTE: final_risk_score = clip(base_risk + graph_penalty" in user
    assert "TRANSACTIONS (2022-01-01)" in user


def test_prefetch_defaults_to_highest_risk_day(store):
    ag, client = make_agent(store, [final("answer")])
    result = ag.chat("A1", "status?")

    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][-1]["content"]
    assert "Case date: 2022-01-01" in system
    assert "CASE DATE: 2022-01-01" in user
    assert "SCORE HISTORY (newest first)" in user
    score_calls = [
        call for call in result.tool_calls if call.name == "get_account_scores"
    ]
    assert len(score_calls) >= 2  # all-dates lookup + exact highest-risk day


def test_date_in_question_is_used(store):
    ag, client = make_agent(store, [final("answer")])
    ag.chat("A1", "What happened on 2022-01-02?")

    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][-1]["content"]
    assert "Case date: 2022-01-02" in system
    assert "CASE DATE: 2022-01-02" in user


def test_unknown_requested_date_warns_and_falls_back(store):
    ag, client = make_agent(store, [final("answer")])
    result = ag.chat("A1", "status?", case_date="2030-01-01")

    assert any("No risk score found" in warning for warning in result.warnings)
    system = client.calls[0]["messages"][0]["content"]
    assert "Case date: 2022-01-01" in system


def test_text_json_tool_protocol_fallback(store):
    ag, client = make_agent(
        store,
        [
            final('{"tool": "get_transactions", "arguments": {"account_id": "A1", "date": "2022-01-01"}}'),
            final("Final answer after JSON tool request."),
        ],
    )
    result = ag.chat("A1", "show flows", case_date="2022-01-01")

    assert result.answer == "Final answer after JSON tool request."
    assert any(
        call.name == "get_transactions" and call.source == "model"
        for call in result.tool_calls
    )


def test_native_tools_unsupported_falls_back_to_json_protocol(store):
    ag, client = make_agent(
        store,
        [
            Exception("registry.ollama.ai/library/model does not support tools"),
            final("Answer without native tools."),
        ],
    )
    result = ag.chat("A1", "status?", case_date="2022-01-01")

    assert result.answer == "Answer without native tools."
    assert ag._tools_unsupported is True
    assert any("JSON tool protocol" in warning for warning in result.warnings)
    # The retry does not send tools again.
    assert "tools" not in client.calls[1]


def test_timeout_degrades_to_evidence_memo(store):
    ag, client = make_agent(store, [TimeoutError("timed out")])
    result = ag.chat("A1", "why flagged?", case_date="2022-01-01")

    assert result.degraded is True
    assert "Offline mode" in result.answer
    assert "0.9" in result.answer
    assert "300.00" in result.answer  # outbound total from the prefetched evidence
    assert any("timeout" in warning.lower() for warning in result.warnings)
    names = {call.name for call in result.tool_calls}
    assert {"get_account_scores", "get_transactions", "get_graph_exposure"} <= names


def test_empty_model_answer_degrades(store):
    ag, client = make_agent(store, [final("")])
    result = ag.chat("A1", "status?", case_date="2022-01-01")

    assert result.degraded is True
    assert "Offline mode" in result.answer
    assert any("empty answer" in warning for warning in result.warnings)


def test_max_steps_forces_final_answer_without_tools(store):
    ag, client = make_agent(
        store,
        [
            tool_call("get_transactions", account_id="A1", date="2022-01-01"),
            final("Forced final answer."),
        ],
        max_steps=1,
    )
    result = ag.chat("A1", "status?", case_date="2022-01-01")

    assert result.answer == "Forced final answer."
    assert result.steps == 1
    assert "tools" not in client.calls[1]
    forced_prompt = client.calls[1]["messages"][-1]["content"]
    assert "Tool-call limit reached" in forced_prompt


def test_native_tools_can_be_disabled_for_speed(store):
    ag, client = make_agent(store, [final("fast answer")])
    ag.native_tools = False
    result = ag.chat("A1", "status?", case_date="2022-01-01")

    assert result.answer == "fast answer"
    assert "tools" not in client.calls[0]
    assert "TOOL CALLING PROTOCOL" in client.calls[0]["messages"][0]["content"]


def test_history_is_sanitised(store):
    history = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "tool", "content": "nope"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "x" * 10_000},
    ]
    ag, client = make_agent(store, [final("ok")])
    ag.chat("A1", "follow up", history=history, case_date="2022-01-01")

    sent = client.calls[0]["messages"]
    roles = [message["role"] for message in sent]
    assert roles[0] == "system" and roles[-1] == "user"
    assert "system" not in roles[1:]
    # Only the three valid prior turns survive (system/tool/empty are dropped).
    assert roles == ["system", "user", "assistant", "user", "user"]
    for message in sent[1:-1]:
        assert len(message["content"]) <= agent_module.MAX_HISTORY_CHARS


def test_review_uses_investigation_prompt(store):
    ag, client = make_agent(store, [final("memo")])
    result = ag.review("A1", case_date="2022-01-01")

    assert result.answer == "memo"
    user_message = client.calls[0]["messages"][-1]["content"]
    assert "investigation memo" in user_message


def test_run_tool_delegates_to_store(store):
    ag = AMLInvestigatorAgent(store=store, client=FakeClient([]), model="fake")
    result = ag.run_tool("get_account_scores", {"account_id": "A1", "date": "2022-01-01"})
    assert result["ok"] is True
    assert result["data"]["summary"]["final_risk_score"] == pytest.approx(0.9)


def test_agent_requires_case_and_message(store):
    ag = AMLInvestigatorAgent(store=store, client=FakeClient([]), model="fake")

    no_case = ag.chat("", "hello")
    assert no_case.degraded is True
    assert "No case selected" in no_case.answer

    no_message = ag.chat("A1", "   ")
    assert no_message.degraded is True
    assert "enter a question" in no_message.answer


def test_empty_store_offline_memo_handles_missing_scores():
    from tools import AMLDataStore

    ag, client = make_agent(AMLDataStore(), [TimeoutError("down")])
    result = ag.chat("A1", "status?")
    assert result.degraded is True
    assert "No risk score was found" in result.answer
