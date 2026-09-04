"""Tests for the agent loop.

Two groups, and the split matters:

* **Free unit tests** (the bulk) drive the loop with a fake Anthropic client.
  They make no API call and cost nothing, and they cover the mechanics that are
  awkward to provoke on demand against a live model: two tools in one turn, a
  tool returning an error, running out of turns, token accounting, logging.
* **Live tests**, marked ``@live``, make real API calls and cost real (small)
  money. They are skipped unless REE_ASSISTANT_LIVE_TESTS=1, so the default
  suite — the one CI runs — stays free and fast.

Run the live ones deliberately:

    REE_ASSISTANT_LIVE_TESTS=1 pytest tests/test_agent.py -m live -v
"""

import json
import logging
import os
from types import SimpleNamespace

import pytest

from src import agent, config, tools

live = pytest.mark.live

requires_live = pytest.mark.skipif(
    os.environ.get("REE_ASSISTANT_LIVE_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="live API test — set REE_ASSISTANT_LIVE_TESTS=1 to run (costs real money)",
)


# ---------------------------------------------------------------------------
# A fake Anthropic client
# ---------------------------------------------------------------------------


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, arguments, block_id="toolu_01"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=block_id)


def response(content, stop_reason, input_tokens=100, output_tokens=20):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeMessages:
    """Replays a scripted list of responses, recording each request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("the agent asked for more turns than the test scripted")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


@pytest.fixture
def stub_tools(monkeypatch):
    """Replace run_tool so no test here touches the sibling project or git."""
    calls = []
    results = {}

    def run_tool(name, tool_input=None):
        calls.append((name, tool_input))
        return results.get(name, {"ok": True, "tool": name})

    monkeypatch.setattr(tools, "run_tool", run_tool)
    return SimpleNamespace(calls=calls, results=results)


# ---------------------------------------------------------------------------
# The model is config.MODEL, never a literal
# ---------------------------------------------------------------------------


def test_the_request_uses_the_configured_model(stub_tools, monkeypatch):
    """Switching models must never require editing agent.py."""
    monkeypatch.setattr(config, "MODEL", "claude-some-future-model")
    client = FakeClient([response([text_block("hi")], "end_turn")])

    result = agent.ask("hello", client=client)

    assert client.messages.requests[0]["model"] == "claude-some-future-model"
    assert result.model == "claude-some-future-model"


def test_agent_module_hardcodes_no_model_string():
    import inspect
    import re

    source = inspect.getsource(agent)

    assert re.findall(r"""["']claude-[\w.-]+["']""", source) == []


def test_the_request_carries_the_tool_schemas_and_system_prompt(stub_tools):
    client = FakeClient([response([text_block("hi")], "end_turn")])

    agent.ask("hello", client=client)

    request = client.messages.requests[0]
    assert request["tools"] is tools.TOOL_SCHEMAS
    assert request["system"] == agent.SYSTEM_PROMPT
    assert request["max_tokens"] == agent.MAX_TOKENS
    assert request["messages"] == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# No tool needed
# ---------------------------------------------------------------------------


def test_a_question_needing_no_tool_is_answered_in_one_round_trip(stub_tools):
    client = FakeClient(
        [response([text_block("I answer questions about the solar forecast.")], "end_turn")]
    )

    result = agent.ask("what are you?", client=client)

    assert result.answer == "I answer questions about the solar forecast."
    assert result.tool_calls == []
    assert result.used_tools is False
    assert result.turns == 1
    assert result.stop_reason == "end_turn"
    assert stub_tools.calls == []


def test_text_blocks_alongside_a_tool_call_do_not_become_the_answer(stub_tools):
    """Preamble text in a tool_use turn is not the final answer."""
    client = FakeClient(
        [
            response(
                [text_block("Let me check."), tool_use_block("get_live_status", {})],
                "tool_use",
            ),
            response([text_block("The model MAE is 1204.9 MW.")], "end_turn"),
        ]
    )

    result = agent.ask("how is it doing?", client=client)

    assert result.answer == "The model MAE is 1204.9 MW."


# ---------------------------------------------------------------------------
# One tool call
# ---------------------------------------------------------------------------


def test_a_tool_call_is_dispatched_and_its_result_returned_to_the_model(stub_tools):
    stub_tools.results["get_live_status"] = {"model_mae": 1204.9, "is_stale": False}
    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {}, "toolu_A")], "tool_use"),
            response([text_block("MAE is 1204.9 MW, fresh.")], "end_turn"),
        ]
    )

    result = agent.ask("what is the live MAE?", client=client)

    assert stub_tools.calls == [("get_live_status", {})]
    assert result.answer == "MAE is 1204.9 MW, fresh."
    assert result.turns == 2

    # The tool result went back as a tool_result block keyed to the call id.
    second_request = client.messages.requests[1]
    assistant_turn, tool_turn = second_request["messages"][1], second_request["messages"][2]
    assert assistant_turn["role"] == "assistant"
    assert tool_turn["role"] == "user"
    (block,) = tool_turn["content"]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_A"
    assert block["is_error"] is False
    assert json.loads(block["content"]) == {"model_mae": 1204.9, "is_stale": False}


def test_tool_arguments_are_passed_through_to_run_tool(stub_tools):
    client = FakeClient(
        [
            response(
                [tool_use_block("detect_anomalies", {"dataset": "historical", "std_threshold": 1.5})],
                "tool_use",
            ),
            response([text_block("Seven days stand out.")], "end_turn"),
        ]
    )

    result = agent.ask("which days were bad?", client=client)

    assert stub_tools.calls == [
        ("detect_anomalies", {"dataset": "historical", "std_threshold": 1.5})
    ]
    assert result.tool_calls[0].arguments == {"dataset": "historical", "std_threshold": 1.5}


# ---------------------------------------------------------------------------
# Several tools in one turn
# ---------------------------------------------------------------------------


def test_two_tools_in_one_turn_are_both_run_and_answered_together(stub_tools):
    """A question spanning live status and the model comparison needs both."""
    stub_tools.results["compare_models"] = {"recent": {"model_mae_mw": 1194.8}}
    stub_tools.results["get_live_status"] = {"model_mae": 1204.9}
    client = FakeClient(
        [
            response(
                [
                    tool_use_block("compare_models", {}, "toolu_1"),
                    tool_use_block("get_live_status", {}, "toolu_2"),
                ],
                "tool_use",
            ),
            response([text_block("Live 1204.9 vs 1194.8 over the test window.")], "end_turn"),
        ]
    )

    result = agent.ask("does live match the backtest?", client=client)

    assert [name for name, _ in stub_tools.calls] == ["compare_models", "get_live_status"]
    assert result.tool_names == ["compare_models", "get_live_status"]
    assert result.turns == 2

    # Both results come back in a single user message — splitting them teaches
    # the model to stop requesting tools in parallel.
    tool_turn = client.messages.requests[1]["messages"][2]
    assert len(tool_turn["content"]) == 2
    assert [b["tool_use_id"] for b in tool_turn["content"]] == ["toolu_1", "toolu_2"]


def test_tool_calls_record_the_turn_they_happened_on(stub_tools):
    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {}, "t1")], "tool_use"),
            response([tool_use_block("compare_models", {}, "t2")], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )

    result = agent.ask("compare live to the backtest", client=client)

    assert [call.turn for call in result.tool_calls] == [1, 2]
    assert result.turns == 3


# ---------------------------------------------------------------------------
# Tool errors reach the model instead of crashing the loop
# ---------------------------------------------------------------------------


def test_a_tool_error_goes_back_to_the_model_as_an_error_result(stub_tools):
    stub_tools.results["get_live_status"] = {
        "error": "latest_metrics.json not found at /x — the workflow may not have run"
    }
    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {}, "toolu_E")], "tool_use"),
            response(
                [text_block("I could not read the live metrics file, so I cannot say.")],
                "end_turn",
            ),
        ]
    )

    result = agent.ask("what is live right now?", client=client)

    # The loop survived and the model got to respond to the failure.
    assert result.answer.startswith("I could not read")
    assert result.turns == 2

    (block,) = client.messages.requests[1]["messages"][2]["content"]
    assert block["is_error"] is True
    assert "latest_metrics.json not found" in block["content"]

    assert result.tool_calls[0].is_error is True


def test_a_tool_error_is_logged_as_a_warning(stub_tools, caplog):
    stub_tools.results["query_mlflow_runs"] = {"error": "mlflow.db not found at /nope"}
    client = FakeClient(
        [
            response([tool_use_block("query_mlflow_runs", {"limit": 3})], "tool_use"),
            response([text_block("The run database is missing.")], "end_turn"),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="src.agent"):
        agent.ask("what were the last runs?", client=client)

    assert "mlflow.db not found" in caplog.text
    assert "query_mlflow_runs" in caplog.text


def test_an_unknown_tool_name_is_reported_rather_than_raised(monkeypatch):
    """run_tool already handles this; the loop must not intercept it."""
    monkeypatch.setattr(tools.sync, "sync_energy_forecast", lambda *a, **k: {"changed": False})
    client = FakeClient(
        [
            response([tool_use_block("summon_more_solar", {}, "toolu_X")], "tool_use"),
            response([text_block("That tool does not exist.")], "end_turn"),
        ]
    )

    result = agent.ask("summon more solar", client=client)

    assert result.tool_calls[0].is_error is True
    assert "unknown tool" in result.tool_calls[0].result["error"]
    assert result.answer == "That tool does not exist."


# ---------------------------------------------------------------------------
# Turn limit
# ---------------------------------------------------------------------------


def test_the_loop_stops_at_max_turns(stub_tools):
    """A model that never stops asking for tools must not run up the bill."""
    client = FakeClient(
        [response([tool_use_block("get_live_status", {}, f"t{i}")], "tool_use") for i in range(10)]
    )

    result = agent.ask("loop forever", client=client, max_turns=3)

    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert len(result.tool_calls) == 3
    assert len(client.messages.requests) == 3
    assert "Stopped after 3 turns" in result.answer


# ---------------------------------------------------------------------------
# Observability: the record Step 4 will score
# ---------------------------------------------------------------------------


def test_token_usage_accumulates_across_turns(stub_tools):
    client = FakeClient(
        [
            response(
                [tool_use_block("get_live_status", {})], "tool_use", input_tokens=100, output_tokens=30
            ),
            response([text_block("done")], "end_turn", input_tokens=400, output_tokens=50),
        ]
    )

    result = agent.ask("how is it doing?", client=client)

    assert result.input_tokens == 500
    assert result.output_tokens == 80


def test_every_tool_call_is_written_to_the_log_file(stub_tools, tmp_path):
    stub_tools.results["get_live_status"] = {"model_mae": 1204.9}
    log_path = tmp_path / "nested" / "agent_calls.jsonl"
    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {"unused": 1})], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )

    agent.ask("how is it doing?", client=client, log_path=log_path)

    (line,) = log_path.read_text().splitlines()
    entry = json.loads(line)
    assert entry["name"] == "get_live_status"
    assert entry["arguments"] == {"unused": 1}
    assert entry["result"] == {"model_mae": 1204.9}
    assert entry["is_error"] is False
    assert entry["question"] == "how is it doing?"
    assert entry["model"] == config.MODEL
    assert entry["logged_at"].startswith("20")
    assert isinstance(entry["duration_seconds"], float)


def test_the_log_file_appends_across_questions(stub_tools, tmp_path):
    log_path = tmp_path / "agent_calls.jsonl"
    for question in ("first", "second"):
        client = FakeClient(
            [
                response([tool_use_block("get_live_status", {})], "tool_use"),
                response([text_block("done")], "end_turn"),
            ]
        )
        agent.ask(question, client=client, log_path=log_path)

    lines = log_path.read_text().splitlines()
    assert [json.loads(line)["question"] for line in lines] == ["first", "second"]


def test_no_log_file_is_written_when_none_is_requested(stub_tools, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {})], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )

    agent.ask("how is it doing?", client=client)

    assert list(tmp_path.iterdir()) == []


def test_the_answer_serialises_for_an_eval_harness(stub_tools):
    stub_tools.results["get_live_status"] = {"model_mae": 1204.9}
    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {})], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )

    payload = agent.ask("how is it doing?", client=client).to_dict()

    assert payload["tool_names"] == ["get_live_status"]
    assert payload["used_tools"] is True
    assert payload["tool_calls"][0]["result"] == {"model_mae": 1204.9}
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Live tests — real API calls, real (small) cost
# ---------------------------------------------------------------------------


@live
@requires_live
def test_live_a_status_question_calls_a_tool_and_answers_from_it():
    """One real question, end to end: model picks a tool, we run it, it answers."""
    result = agent.ask("What is the solar model's current MAE right now?")

    assert result.used_tools, "expected the model to call a tool for a live-status question"
    assert result.answer.strip()
    assert result.stop_reason == "end_turn"
    assert result.input_tokens > 0

    # Grounded in a real result: the MAE from latest_metrics.json, not invented.
    live_call = next(call for call in result.tool_calls if not call.is_error)
    assert isinstance(live_call.result, dict)


@live
@requires_live
def test_live_a_question_needing_no_tool_skips_the_tools():
    result = agent.ask("Reply with exactly the word: pong. Do not call any tool.")

    assert result.answer.strip()
    assert result.tool_calls == []
    assert result.turns == 1
