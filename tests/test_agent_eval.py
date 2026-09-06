"""Step 4: the evaluation harness.

Each case in tests/eval_cases.py is a real question with verified ground truth,
asked against the real model through the real tools. Every case is therefore a
paid API call, so these carry the same double gate as the other live tests: the
`live` marker is deselected by default in pytest.ini, and REE_ASSISTANT_LIVE_TESTS
must be set.

    REE_ASSISTANT_LIVE_TESTS=1 pytest tests/test_agent_eval.py -m live -v

The scoring functions themselves are pure and are unit-tested for free at the
bottom of this file — the harness that decides what counts as a pass should not
itself be unverified.

Three scores per case, kept separate because they fail separately and the
distinction is the whole point: an agent can pick the right tool and misreport
it, or fabricate a number without calling anything at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src import agent, config
from tests.eval_cases import EVAL_CASES, EvalCase

live = pytest.mark.live

requires_live = pytest.mark.skipif(
    os.environ.get("REE_ASSISTANT_LIVE_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="live API test — set REE_ASSISTANT_LIVE_TESTS=1 to run (costs real money)",
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.json"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Score:
    passed: bool | None  # None = not applicable to this case
    reason: str


@dataclass
class CaseResult:
    id: str
    question: str
    tools_called: list[str]
    # The payloads the tools returned. Kept so a grader fix can be applied to a
    # recorded run instead of paying for a fresh one — which is exactly what
    # happened to the first run of this eval.
    tool_results: dict[str, Any]
    tool_choice: Score
    arguments: Score
    faithfulness: Score
    answer: str
    turns: int
    input_tokens: int
    output_tokens: int

    @property
    def fully_passed(self) -> bool:
        return all(
            score.passed is not False
            for score in (self.tool_choice, self.arguments, self.faithfulness)
        )


def score_tool_choice(case: EvalCase, called: list[str]) -> Score:
    """Did it reach for an acceptable tool — or correctly for none?"""
    called_set = frozenset(called)

    if case.acceptable_tools == (frozenset(),):
        if called_set:
            return Score(False, f"expected no tool call, got {sorted(called_set)}")
        return Score(True, "correctly called no tool")

    if called_set in case.acceptable_tools:
        return Score(True, f"called {sorted(called_set)}")

    trimmed = called_set - case.tolerated_extras
    if trimmed in case.acceptable_tools:
        return Score(True, f"called {sorted(called_set)} (extras tolerated)")

    expected = " or ".join(sorted(str(sorted(s)) for s in case.acceptable_tools))
    return Score(False, f"called {sorted(called_set)}, expected {expected}")


def score_arguments(case: EvalCase, answer: agent.AgentAnswer) -> Score:
    """Were the arguments the question implied actually passed?"""
    if not case.expected_args:
        return Score(None, "no arguments required")

    problems = []
    for tool, required in case.expected_args.items():
        call = next((c for c in answer.tool_calls if c.name == tool), None)
        if call is None:
            problems.append(f"{tool} was not called")
            continue
        for key, expected in required.items():
            actual = call.arguments.get(key, "<missing>")
            # 1 and 1.0 are the same argument as far as the tool is concerned.
            same = actual == expected or (
                isinstance(actual, (int, float))
                and isinstance(expected, (int, float))
                and float(actual) == float(expected)
            )
            if not same:
                problems.append(f"{tool}.{key}={actual!r}, expected {expected!r}")

    if problems:
        return Score(False, "; ".join(problems))
    return Score(True, "arguments correct")


def score_faithfulness(case: EvalCase, answer: agent.AgentAnswer) -> Score:
    """Does the answer match what the tools actually returned?"""
    results: dict[str, Any] = {}
    for call in answer.tool_calls:
        if not call.is_error and call.name not in results:
            results[call.name] = call.result

    passed, reason = case.faithfulness(answer.answer, results)
    return Score(passed, reason)


def score(case: EvalCase, answer: agent.AgentAnswer) -> CaseResult:
    return CaseResult(
        id=case.id,
        question=case.question,
        tools_called=answer.tool_names,
        tool_results={
            call.name: call.result for call in reversed(answer.tool_calls) if not call.is_error
        },
        tool_choice=score_tool_choice(case, answer.tool_names),
        arguments=score_arguments(case, answer),
        faithfulness=score_faithfulness(case, answer),
        answer=answer.answer,
        turns=answer.turns,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    """Per-dimension pass rates. Not-applicable arguments are excluded, not counted as wins."""

    def rate(values: list[bool]) -> dict[str, Any]:
        return {
            "passed": sum(values),
            "of": len(values),
            "pct": round(sum(values) / len(values) * 100, 1) if values else None,
        }

    return {
        "cases": len(results),
        "tool_choice": rate([r.tool_choice.passed for r in results if r.tool_choice.passed is not None]),
        "arguments": rate([r.arguments.passed for r in results if r.arguments.passed is not None]),
        "faithfulness": rate([r.faithfulness.passed for r in results if r.faithfulness.passed is not None]),
        "all_three": rate([r.fully_passed for r in results]),
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
    }


# ---------------------------------------------------------------------------
# The live run
# ---------------------------------------------------------------------------

_collected: list[CaseResult] = []


@pytest.fixture(scope="module", autouse=True)
def write_results_file():
    """Write eval_results.json after the run, so the rate can be cited later."""
    _collected.clear()
    yield
    if not _collected:
        return

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "model": config.MODEL,
                "summary": summarise(_collected),
                "cases": [asdict(r) for r in _collected],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


@live
@requires_live
@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.id)
def test_eval_case(case: EvalCase):
    result = score(case, agent.ask(case.question))
    _collected.append(result)

    failures = [
        f"{dimension}: {getattr(result, dimension).reason}"
        for dimension in ("tool_choice", "arguments", "faithfulness")
        if getattr(result, dimension).passed is False
    ]

    assert not failures, (
        f"\n  question: {case.question}"
        f"\n  tools:    {result.tools_called}"
        f"\n  answer:   {result.answer[:400]}"
        f"\n  failed:   " + "\n            ".join(failures)
    )


# ---------------------------------------------------------------------------
# Free tests: the eval set, and the scoring that judges it
# ---------------------------------------------------------------------------


def test_the_eval_set_is_large_enough_and_covers_every_tool():
    assert 15 <= len(EVAL_CASES) <= 25

    covered = {tool for case in EVAL_CASES for group in case.acceptable_tools for tool in group}
    assert covered == {
        "get_live_status",
        "query_mlflow_runs",
        "compare_models",
        "detect_anomalies",
    }


def test_the_eval_set_has_the_awkward_cases():
    ids = {case.id for case in EVAL_CASES}

    # A question needing no tool at all.
    assert any(case.acceptable_tools == (frozenset(),) for case in EVAL_CASES)
    # A question whose honest answer is unflattering.
    assert {"unflattering-beats-ree", "unflattering-improvement"} <= ids
    # A question two tools answer differently, both defensibly.
    assert {"ambiguous-which-mae", "ambiguous-drift"} <= ids
    # Questions that imply arguments.
    assert sum(1 for case in EVAL_CASES if case.expected_args) >= 3


def test_case_ids_are_unique():
    ids = [case.id for case in EVAL_CASES]

    assert len(ids) == len(set(ids))


def _answer(tool_calls=(), text="", **kwargs):
    result = agent.AgentAnswer(
        question="q", answer=text, model=config.MODEL, turns=1, **kwargs
    )
    for name, arguments, payload in tool_calls:
        result.tool_calls.append(
            agent.ToolCall(
                name=name,
                arguments=arguments,
                result=payload,
                is_error="error" in payload,
                duration_seconds=0.1,
                turn=1,
            )
        )
    return result


def _case(**overrides):
    defaults = dict(
        id="x",
        question="q",
        acceptable_tools=(frozenset({"get_live_status"}),),
        faithfulness=lambda answer, results: (True, "ok"),
        notes="",
    )
    return EvalCase(**{**defaults, **overrides})


def test_tool_choice_passes_on_an_acceptable_set():
    assert score_tool_choice(_case(), ["get_live_status"]).passed is True


def test_tool_choice_fails_on_the_wrong_tool():
    result = score_tool_choice(_case(), ["compare_models"])

    assert result.passed is False
    assert "compare_models" in result.reason


def test_tool_choice_fails_when_a_no_tool_question_calls_one():
    result = score_tool_choice(_case(acceptable_tools=(frozenset(),)), ["get_live_status"])

    assert result.passed is False
    assert "expected no tool call" in result.reason


def test_tool_choice_passes_when_a_no_tool_question_calls_nothing():
    assert score_tool_choice(_case(acceptable_tools=(frozenset(),)), []).passed is True


def test_tool_choice_tolerates_declared_extras_only():
    case = _case(tolerated_extras=frozenset({"compare_models"}))

    assert score_tool_choice(case, ["get_live_status", "compare_models"]).passed is True
    assert score_tool_choice(case, ["get_live_status", "detect_anomalies"]).passed is False


def test_arguments_are_not_applicable_when_none_are_expected():
    assert score_arguments(_case(), _answer()).passed is None


def test_arguments_pass_when_the_expected_value_was_sent():
    case = _case(
        acceptable_tools=(frozenset({"detect_anomalies"}),),
        expected_args={"detect_anomalies": {"dataset": "historical"}},
    )
    answer = _answer([("detect_anomalies", {"dataset": "historical"}, {"anomaly_count": 7})])

    assert score_arguments(case, answer).passed is True


def test_arguments_fail_on_a_wrong_or_missing_value():
    case = _case(
        acceptable_tools=(frozenset({"detect_anomalies"}),),
        expected_args={"detect_anomalies": {"dataset": "historical"}},
    )

    wrong = _answer([("detect_anomalies", {"dataset": "recent"}, {})])
    assert score_arguments(case, wrong).passed is False
    assert "'recent'" in score_arguments(case, wrong).reason

    missing = _answer([("detect_anomalies", {}, {})])
    assert score_arguments(case, missing).passed is False


def test_arguments_treat_one_and_one_point_zero_as_the_same():
    case = _case(
        acceptable_tools=(frozenset({"detect_anomalies"}),),
        expected_args={"detect_anomalies": {"std_threshold": 1.0}},
    )
    answer = _answer([("detect_anomalies", {"std_threshold": 1}, {})])

    assert score_arguments(case, answer).passed is True


def test_faithfulness_ignores_results_from_failed_tool_calls():
    """A tool that errored returned no data, so nothing can be faithful to it."""
    from tests.eval_cases import states

    case = _case(faithfulness=states("get_live_status", "model_mae"))
    answer = _answer([("get_live_status", {}, {"error": "file missing"})], text="1204.9")

    result = score_faithfulness(case, answer)
    assert result.passed is False
    assert "never called" in result.reason


def test_summarise_reports_each_dimension_separately():
    from tests.eval_cases import states

    good = score(
        _case(faithfulness=states("get_live_status", "model_mae")),
        _answer([("get_live_status", {}, {"model_mae": 1204.9})], text="MAE is 1204.9 MW"),
    )
    bad = score(
        _case(faithfulness=states("get_live_status", "model_mae")),
        _answer([("get_live_status", {}, {"model_mae": 1204.9})], text="MAE is 900 MW"),
    )

    summary = summarise([good, bad])
    assert summary["cases"] == 2
    assert summary["tool_choice"] == {"passed": 2, "of": 2, "pct": 100.0}
    assert summary["faithfulness"] == {"passed": 1, "of": 2, "pct": 50.0}
    assert summary["all_three"]["pct"] == 50.0
    assert summary["arguments"]["of"] == 0  # not applicable to either case


# ---------------------------------------------------------------------------
# The number matcher decides most faithfulness scores — it gets its own tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("MAE is 1204.9 MW", 1204.9, True),
        ("MAE is 1,204.9 MW", 1204.9, True),  # thousands separator
        ("roughly 1205 MW", 1204.9, True),  # rounded
        ("MAE is 900 MW", 1204.9, False),
        ("improvement is -44.8%", -44.8, True),
        ("44.8% worse than REE", -44.8, True),  # magnitude, sign checked separately
        ("there are 4 runs", 4, True),
        ("44.8% worse", 4, False),  # must not match digits inside another number
        ("recorded in 2026", 4, False),  # must not match inside a year
        ("43 of 181 days", 43, True),
        ("43 of 181 days", 181, True),
        ("29.5% of days", 29.5, True),
        # Regressions from the first eval run: an answer more precise than the
        # expected value was scored as omitting it.
        ("the baseline MAE logged was 123.61 (specifically 123.6121)", 123.6121, True),
        ("123.6121", 123.6121, True),
        ("123.61", 123.6121, True),
        ("124", 123.6121, True),
        ("123.9", 123.6121, False),
        # Dates must not donate digits to the matcher.
        ("computed on 2026-09-04 at 09:44", 4, False),
        ("about 30% of days", 29.5, True),
    ],
)
def test_mentions_number(text, value, expected):
    from tests.eval_cases import mentions_number

    assert mentions_number(text, value) is expected


def test_dig_walks_dicts_and_lists():
    from tests.eval_cases import dig

    payload = {"runs": [{"metrics": {"model_mae": 1100.04}}]}

    assert dig(payload, "runs.0.metrics.model_mae") == 1100.04


def test_forbids_unqualified_catches_an_unqualified_claim():
    from tests.eval_cases import forbids_unqualified

    check = forbids_unqualified("beats REE")

    assert check("The model beats REE comfortably.", {})[0] is False
    assert check("The model is worse than REE.", {})[0] is True


def test_forbids_unqualified_allows_a_true_but_awkward_sentence():
    """The exact sentence that failed the first run, verbatim."""
    from tests.eval_cases import forbids_unqualified

    check = forbids_unqualified("beats REE")
    answer = (
        "**No, this model does not beat REE's official day-ahead forecast.**\n"
        "- **Win rate:** 29.5% (the model beats REE on less than 1 in 3 days)"
    )

    passed, reason = check(answer, {})
    assert passed is True, reason


def test_staleness_expectation_follows_the_tool_result():
    from tests.eval_cases import staleness_is_reported

    stale = {"get_live_status": {"is_stale": True, "hours_since_computed": 2.1}}
    fresh = {"get_live_status": {"is_stale": False, "hours_since_computed": 0.3}}

    # Stale data must be flagged as such.
    assert staleness_is_reported("Figures are current.", stale)[0] is False
    assert staleness_is_reported("These are 2.1 hours old.", stale)[0] is True
    # Fresh data must not be described as stale.
    assert staleness_is_reported("The data is stale.", fresh)[0] is False
    assert staleness_is_reported("Computed 18 minutes ago.", fresh)[0] is True
