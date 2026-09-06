"""Tests for the Streamlit interface.

Free — Streamlit's own AppTest harness runs app.py headlessly and src.agent.ask
is replaced with a stub, so no API call is made. The load-time test is the one
that matters most: opening the page must cost nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import agent, config

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# Relative paths resolve against this file, so point at the repo root.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _answer(text="The model's MAE is 1204.9 MW.", tool_calls=(("get_live_status", {}, {"model_mae": 1204.9}),)):
    result = agent.AgentAnswer(
        question="q", answer=text, model=config.MODEL, turns=2,
        input_tokens=100, output_tokens=40, stop_reason="end_turn",
    )
    for name, arguments, payload in tool_calls:
        result.tool_calls.append(
            agent.ToolCall(
                name=name, arguments=arguments, result=payload,
                is_error="error" in payload, duration_seconds=0.2, turn=1,
            )
        )
    return result


def test_the_page_loads_without_calling_the_api(monkeypatch):
    """Opening the page must not cost anything."""

    def explode(*args, **kwargs):
        raise AssertionError("the app called the model on page load")

    monkeypatch.setattr(agent, "ask", explode)

    app = AppTest.from_file(APP, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "☀️ REE Assistant"
    assert len(app.text_input) == 1  # the question box, unsubmitted


def test_submitting_a_question_calls_the_agent_once_and_shows_the_answer(monkeypatch):
    calls = []

    def fake_ask(question, **kwargs):
        calls.append(question)
        return _answer()

    monkeypatch.setattr(agent, "ask", fake_ask)

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input[0].set_value("What is the MAE?")
    app.button[0].click().run()

    assert calls == ["What is the MAE?"]
    assert not app.exception
    assert any("1204.9" in block.value for block in app.markdown)


def test_a_clean_answer_reports_that_the_guardrails_found_nothing(monkeypatch):
    monkeypatch.setattr(agent, "ask", lambda question, **kwargs: _answer())

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input[0].set_value("What is the MAE?")
    app.button[0].click().run()

    assert any("no issues found" in box.value for box in app.success)
    # And it does not overclaim.
    assert any("not a guarantee of correctness" in box.value for box in app.success)


def test_a_fabricated_figure_is_shown_to_the_user_not_hidden(monkeypatch):
    """The whole point of screening: a bad answer is still displayed, but flagged."""
    monkeypatch.setattr(
        agent,
        "ask",
        lambda question, **kwargs: _answer(
            text="The MAE is 999.9 MW.",
            tool_calls=(("get_live_status", {}, {"model_mae": 1204.9}),),
        ),
    )

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input[0].set_value("What is the MAE?")
    app.button[0].click().run()

    flagged = [box.value for box in app.warning] + [box.value for box in app.error]
    assert any("999.9" in text for text in flagged), flagged
    # The answer itself is still rendered — flagged, not suppressed.
    assert any("999.9" in block.value for block in app.markdown)


def test_a_silent_tool_failure_is_flagged(monkeypatch):
    monkeypatch.setattr(
        agent,
        "ask",
        lambda question, **kwargs: _answer(
            text="Everything looks normal.",
            tool_calls=(("get_live_status", {}, {"error": "latest_metrics.json not found"}),),
        ),
    )

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input[0].set_value("What is the MAE?")
    app.button[0].click().run()

    errors = [box.value for box in app.error]
    assert any("does not say so" in text for text in errors), errors


def test_an_api_failure_is_surfaced_rather_than_swallowed(monkeypatch):
    def explode(*args, **kwargs):
        raise TypeError("Could not resolve authentication method")

    monkeypatch.setattr(agent, "ask", explode)

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input[0].set_value("What is the MAE?")
    app.button[0].click().run()

    assert not app.exception, "the app crashed instead of reporting the failure"
    assert any("ANTHROPIC_API_KEY" in box.value for box in app.error)


def test_an_empty_question_does_not_call_the_api(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("called the model with an empty question")

    monkeypatch.setattr(agent, "ask", explode)

    app = AppTest.from_file(APP, default_timeout=30).run()
    app.text_input[0].set_value("   ")
    app.button[0].click().run()

    assert not app.exception
    assert any("Type a question first" in box.value for box in app.warning)
