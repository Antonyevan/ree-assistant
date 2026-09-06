"""Step 5: guardrails against the two failure modes the eval exposed in itself.

The evaluation harness scores faithfulness with checks written per question, so
it catches contradictions it was told to look for. These do not need that
foresight, and they run in CI for free.

Two of the three are checked against the recorded evaluation run in
eval_results.json — 21 real answers with the tool payloads behind them. That
makes them ongoing rather than documentation of known cases: rerun the eval,
commit the new results, and the guardrails re-measure against fresh behaviour.
A worsening tendency fails the build without anyone paying for an API call.

Live variants of each are marked @live and skipped by default, for the parts
that genuinely need a model's judgement rather than a recorded transcript.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src import agent, guardrails, tools
from tests.eval_cases import EVAL_CASES

live = pytest.mark.live

requires_live = pytest.mark.skipif(
    os.environ.get("REE_ASSISTANT_LIVE_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="live API test — set REE_ASSISTANT_LIVE_TESTS=1 to run (costs real money)",
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.json"

# Questions with one clear data need. Used by the live redundancy guardrail.
SINGLE_NEED_QUESTIONS = [
    ("What is the model's MAE right now?", "get_live_status"),
    ("When was the model last trained?", "query_mlflow_runs"),
    ("How many unusual error days did the historical model have?", "detect_anomalies"),
]

# ambiguous-drift computes "a degradation of about 9.5%" from two figures it
# correctly reported (1204.9 against 1100.04). Deriving a number is not
# inventing one, but the detector cannot tell them apart without admitting
# arithmetic, which was measured to destroy its sensitivity — so the one known
# derivation is named here instead.
KNOWN_DERIVED_ARITHMETIC = {"ambiguous-drift"}

# mlflow-when and compare-windows each called one tool they did not need.
KNOWN_REDUNDANT_CASES = {"mlflow-when", "compare-windows"}


@pytest.fixture(scope="module")
def recorded_run():
    if not RESULTS_PATH.exists():
        pytest.skip("no recorded eval run to check against")
    return json.loads(RESULTS_PATH.read_text())


# ---------------------------------------------------------------------------
# Guardrail 1 — a number stated as data must come from a tool
# ---------------------------------------------------------------------------


def test_an_answer_with_no_tool_call_states_no_figures(recorded_run):
    """The unambiguous fabrication case: no tool ran, so nothing can be sourced.

    There is nothing to derive from and nothing to round, so any measurement in
    the answer was invented. This is the one grounding check with no possible
    false positive.
    """
    offenders = {}
    for case in recorded_run["cases"]:
        if case["tools_called"]:
            continue
        invented = guardrails.ungrounded_numbers(case["answer"], {})
        if invented:
            offenders[case["id"]] = invented

    assert not offenders, f"figures stated with no tool call behind them: {offenders}"


def test_recorded_answers_do_not_invent_figures(recorded_run):
    """Every figure in every answer traces back to a tool result."""
    offenders = {
        case["id"]: bad
        for case in recorded_run["cases"]
        if (bad := guardrails.ungrounded_numbers(case["answer"], case["tool_results"]))
    }

    unexpected = set(offenders) - KNOWN_DERIVED_ARITHMETIC
    assert not unexpected, (
        "answers state figures no tool returned: "
        + json.dumps({k: offenders[k] for k in unexpected})
    )


def test_a_fabricated_figure_is_caught():
    answer = "The model's MAE is currently 1204.9 MW against a baseline of 832.1 MW."

    assert guardrails.ungrounded_numbers(answer, {}) == [1204.9, 832.1]


def test_a_reported_figure_is_not_flagged():
    answer = "The model's MAE is 1204.9 MW."
    results = {"get_live_status": {"model_mae": 1204.9, "baseline_mae": 832.1}}

    assert guardrails.ungrounded_numbers(answer, results) == []


def test_a_figure_close_to_but_not_equal_to_the_data_is_caught():
    """The dangerous case: plausible, well-formatted, and wrong."""
    answer = "The model's MAE is 1250.0 MW."
    results = {"get_live_status": {"model_mae": 1204.9}}

    assert guardrails.ungrounded_numbers(answer, results) == [1250.0]


def test_rounding_and_thousands_separators_are_grounded():
    results = {"get_live_status": {"model_mae": 1204.9, "win_rate_pct": 29.5}}

    assert guardrails.ungrounded_numbers("MAE 1,204.9 MW, winning ~30% of days", results) == []
    assert guardrails.ungrounded_numbers("MAE of roughly 1205 MW", results) == []


def test_negative_values_are_grounded_by_magnitude():
    """-44.8% improvement is reported as "44.8% worse"."""
    results = {"get_live_status": {"improvement_pct": -44.8}}

    assert guardrails.ungrounded_numbers("It is 44.8% worse than the baseline.", results) == []


def test_prose_integers_and_dates_are_not_treated_as_data_claims():
    results = {"get_live_status": {"anomaly_count": 1}}
    answer = "On 2026-08-18, across the 2015-2018 period, it wins 1 day in 3."

    assert guardrails.ungrounded_numbers(answer, results) == []


def test_figures_nested_deep_in_a_payload_count_as_grounded():
    results = {"detect_anomalies": {"anomalies": [{"date": "2026-08-18", "model_error_mw": 2292.7}]}}

    assert guardrails.ungrounded_numbers("The worst day was 2292.7 MW.", results) == []


def test_time_relative_fields_tolerate_drift():
    """hours_since_* is recomputed per call, so a small difference is not invention."""
    results = {"get_live_status": {"hours_since_computed": 2.1}}

    assert guardrails.ungrounded_numbers("Computed 2.3 hours ago.", results) == []
    assert guardrails.ungrounded_numbers("Computed 40.0 hours ago.", results) == [40.0]


# ---------------------------------------------------------------------------
# Guardrail 2 — a failed or incomplete tool result must be reported as such
# ---------------------------------------------------------------------------


def test_reporting_figures_from_an_errored_tool_is_caught():
    """An error payload carries no data, so any figure alongside it is invented."""
    results = {"get_live_status": {"error": "latest_metrics.json not found"}}
    answer = "The model's MAE is 1204.9 MW."

    assert guardrails.ungrounded_numbers(answer, results) == [1204.9]
    assert guardrails.acknowledges_failure(answer) is False


def test_an_honest_failure_report_passes_both_checks():
    results = {"get_live_status": {"error": "latest_metrics.json not found"}}
    answer = "I could not read the live metrics file, so I cannot give you a current MAE."

    assert guardrails.ungrounded_numbers(answer, results) == []
    assert guardrails.acknowledges_failure(answer) is True


def test_a_missing_field_must_not_be_filled_in():
    """The subtler case: the tool succeeded but did not return what was asked."""
    results = {"get_live_status": {"win_rate_pct": 29.5}}  # no model_mae

    assert guardrails.ungrounded_numbers("The MAE is 1204.9 MW.", results) == [1204.9]
    assert guardrails.ungrounded_numbers("No MAE was returned; win rate is 29.5%.", results) == []


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("I could not retrieve the data.", True),
        ("The file was not found.", True),
        ("That metric is unavailable.", True),
        ("The tool returned an error.", True),
        ("The model's MAE is 1204.9 MW.", False),
        ("Everything looks fine.", False),
    ],
)
def test_acknowledges_failure(answer, expected):
    assert guardrails.acknowledges_failure(answer) is expected


def test_the_loop_marks_an_error_payload_so_the_model_can_see_it(monkeypatch):
    """The guardrail depends on errors reaching the model flagged, not silently."""
    monkeypatch.setenv("REE_ASSISTANT_SYNC", "0")
    monkeypatch.setattr(
        tools, "run_tool", lambda name, args=None: {"error": "latest_metrics.json not found"}
    )

    from tests.test_agent import FakeClient, response, text_block, tool_use_block

    client = FakeClient(
        [
            response([tool_use_block("get_live_status", {}, "t1")], "tool_use"),
            response([text_block("I could not read the file.")], "end_turn"),
        ]
    )
    result = agent.ask("what is the MAE?", client=client)

    assert result.tool_calls[0].is_error is True
    (block,) = client.messages.requests[1]["messages"][2]["content"]
    assert block["is_error"] is True


# ---------------------------------------------------------------------------
# Guardrail 3 — no more tools than the question needs
# ---------------------------------------------------------------------------


def _redundancy(recorded_run) -> dict[str, list[str]]:
    by_id = {case.id: case for case in EVAL_CASES}
    offenders = {}
    for row in recorded_run["cases"]:
        case = by_id.get(row["id"])
        if case is None:
            continue
        if guardrails.redundant_tool_count(row["tools_called"], case.acceptable_tools):
            offenders[row["id"]] = row["tools_called"]
    return offenders


def test_redundant_tool_calls_have_not_spread(recorded_run):
    """Ongoing check: this must not get worse than the two cases we know about.

    Calling the right tool plus one it did not need is not a wrong answer, but
    it costs latency and tokens on every question. Pinning the known set means a
    third case fails the build rather than passing unnoticed.
    """
    offenders = _redundancy(recorded_run)
    new = set(offenders) - KNOWN_REDUNDANT_CASES

    assert not new, f"new redundant tool calls: {({k: offenders[k] for k in new})}"


def test_the_known_redundant_cases_are_still_the_known_ones(recorded_run):
    """If they get fixed, this fails and the baseline should be tightened."""
    offenders = set(_redundancy(recorded_run))
    fixed = KNOWN_REDUNDANT_CASES - offenders

    assert not fixed, (
        f"{sorted(fixed)} no longer calls a redundant tool — "
        "remove it from KNOWN_REDUNDANT_CASES so the guardrail keeps its edge"
    )


def test_redundant_tool_count_counts_beyond_the_minimum():
    one_tool = (frozenset({"get_live_status"}),)

    assert guardrails.redundant_tool_count(["get_live_status"], one_tool) == 0
    assert guardrails.redundant_tool_count(["get_live_status", "compare_models"], one_tool) == 1
    # A question that legitimately needs two tools is not penalised for two.
    two_tools = (frozenset({"query_mlflow_runs", "compare_models"}),)
    assert guardrails.redundant_tool_count(
        ["query_mlflow_runs", "compare_models"], two_tools
    ) == 0


def test_repeated_calls_to_one_tool_are_not_counted_as_redundant_breadth():
    """Two calls to the same tool is a different problem from two tools."""
    one_tool = (frozenset({"detect_anomalies"}),)

    assert guardrails.redundant_tool_count(
        ["detect_anomalies", "detect_anomalies"], one_tool
    ) == 0


# ---------------------------------------------------------------------------
# Live variants — real model judgement, manually run
# ---------------------------------------------------------------------------


@live
@requires_live
@pytest.mark.parametrize("question,expected", SINGLE_NEED_QUESTIONS, ids=lambda v: str(v)[:30])
def test_live_a_single_need_question_calls_exactly_one_tool(question, expected):
    result = agent.ask(question)

    assert result.tool_names, "expected one tool call, got none"
    assert len(set(result.tool_names)) == 1, (
        f"called {result.tool_names} for a question with one data need"
    )
    assert result.tool_names[0] == expected


@live
@requires_live
def test_live_a_tool_error_is_reported_not_papered_over(monkeypatch):
    monkeypatch.setattr(
        tools,
        "run_tool",
        lambda name, args=None: {"error": f"{name}: latest_metrics.json not found at /tmp/x"},
    )

    result = agent.ask("What is the model's MAE right now?")

    assert guardrails.acknowledges_failure(result.answer), (
        f"tool failed but the answer does not say so: {result.answer!r}"
    )
    invented = guardrails.ungrounded_numbers(result.answer, {})
    assert not invented, f"answered with figures despite the tool failing: {invented}"


@live
@requires_live
def test_live_a_missing_field_is_not_filled_in(monkeypatch):
    """The tool succeeds but omits the field asked about."""
    monkeypatch.setattr(
        tools,
        "run_tool",
        lambda name, args=None: {"win_rate_pct": 29.5, "computed_at": "2026-09-04T09:44:00+00:00"},
    )

    result = agent.ask("What is the model's MAE right now?")

    invented = guardrails.ungrounded_numbers(result.answer, {"t": {"win_rate_pct": 29.5}})
    assert not invented, f"invented a figure the tool did not return: {invented}"
