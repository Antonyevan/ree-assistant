"""The evaluation set: 20 questions with verified ground truth.

Every expectation here was established by running the underlying tool first and
reading what it actually returned — not by guessing what it ought to say. The
figures quoted in the notes were true when the case was written; where a value
moves (the live metrics refresh every 30 minutes, the recent training data
weekly) the scoring reads the value back out of the tool result produced during
the run rather than comparing against a frozen number. Only the historical side
is genuinely fixed: it is computed from a 2015-2018 CSV that never changes.

Three things are scored separately for each case, because they fail separately:

* **tool choice** — did it reach for the right tool, or correctly for none?
* **arguments** — if the question implied arguments, did it pass them?
* **faithfulness** — does the answer match what the tool returned, without
  inventing or contradicting it?

An agent can call the right tool and then misreport it, or fabricate a number
without calling anything. A single pass/fail hides which happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Reading values back out of a tool result
# ---------------------------------------------------------------------------


def dig(payload: Any, path: str) -> Any:
    """Resolve a dotted path like 'recent.model_mae_mw' or 'anomalies.0.date'."""
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


# Dates and clock times are checked by quotes(), not by the number matcher.
# Stripping them first stops "2026-09-04" from offering a stray 4 or 9.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?|\b\d{1,2}:\d{2}\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[tuple[float, int]]:
    """Every number the text states, as (value, decimal places written)."""
    found = []
    for match in _TIMESTAMP.sub(" ", text).split():
        for token in _NUMBER.finditer(match):
            raw = token.group().replace(",", "")
            decimals = len(raw.partition(".")[2])
            try:
                found.append((float(raw), decimals))
            except ValueError:  # pragma: no cover - regex guarantees a number
                pass
    return found


def mentions_number(text: str, value: float) -> bool:
    """True if the answer states this number at any sensible precision.

    Compares numerically rather than by string, because string matching gets
    this exactly backwards: it rejects an answer *more* precise than expected.
    Asked for a baseline MAE of 123.6121, the agent answered "123.61
    (specifically 123.6121)" and the old matcher scored it as omitted, because
    its candidate "123.6" was followed by another digit.

    A stated number counts if the target rounds to it at the precision the
    answer chose: "1205" matches 1204.9, "123.61" matches 123.6121, "900" does
    not match 1204.9. Magnitude is what is checked — an improvement of -44.8%
    is commonly written "44.8% worse" — so direction is a separate concern
    (see forbids_unqualified).
    """
    target = abs(float(value))
    for stated, decimals in numbers_in(text):
        tolerance = 0.5 * (10.0**-decimals)
        if abs(stated - target) <= tolerance + 1e-9:
            return True
    return False


# ---------------------------------------------------------------------------
# Faithfulness checks
# ---------------------------------------------------------------------------

# A check takes the answer text and the tool results (name -> payload) keyed by
# the first successful call to that tool, and returns (passed, reason).
Check = Callable[[str, dict[str, Any]], "tuple[bool, str]"]


def states(tool: str, *paths: str) -> Check:
    """The answer must state the values the tool actually returned."""

    def check(answer: str, results: dict[str, Any]) -> tuple[bool, str]:
        if tool not in results:
            return False, f"{tool} was never called, so its values cannot be reported"
        missing = []
        for path in paths:
            value = dig(results[tool], path)
            if not mentions_number(answer, float(value)):
                missing.append(f"{path}={value}")
        if missing:
            return False, f"answer omits {', '.join(missing)}"
        return True, "states the returned values"

    return check


def quotes(tool: str, *paths: str) -> Check:
    """The answer must contain the string values at these paths (dates, names)."""

    def check(answer: str, results: dict[str, Any]) -> tuple[bool, str]:
        if tool not in results:
            return False, f"{tool} was never called"
        missing = [
            f"{path}={dig(results[tool], path)}"
            for path in paths
            if str(dig(results[tool], path)) not in answer
        ]
        if missing:
            return False, f"answer omits {', '.join(missing)}"
        return True, "quotes the returned values"

    return check


# Words that turn a superiority phrase into a true, unflattering statement.
_QUALIFIERS = (
    "not", "n't", "never", "rarely", "only", "less than", "fewer than",
    "under", "below", "worse", "fails", "barely", "no,", "seldom",
)


def forbids_unqualified(*phrases: str) -> Check:
    """The answer must not assert these claims *without qualification*.

    A flat blacklist is wrong here, and scored a correct answer as a failure:
    asked whether the model beats REE, the agent opened with "No, this model
    does not beat REE's official day-ahead forecast" and later wrote "Win rate:
    29.5% (the model beats REE on less than 1 in 3 days)". The second line is
    true and unflattering, but contains the forbidden substring.

    So the unit of judgement is the sentence, and a sentence carrying a
    qualifier is reporting rather than claiming.
    """

    def check(answer: str, _results: dict[str, Any]) -> tuple[bool, str]:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
            lowered = sentence.lower()
            claimed = next((p for p in phrases if p.lower() in lowered), None)
            if claimed is None:
                continue
            if any(qualifier in lowered for qualifier in _QUALIFIERS):
                continue
            return False, (
                f"asserts {claimed!r} without qualification: {sentence.strip()[:120]!r}"
            )
        return True, "makes no unqualified claim the data contradicts"

    return check


def requires_any(*phrases: str) -> Check:
    """The answer must contain at least one of these (case-insensitive)."""

    def check(answer: str, _results: dict[str, Any]) -> tuple[bool, str]:
        lowered = answer.lower()
        if any(phrase.lower() in lowered for phrase in phrases):
            return True, "acknowledges the required point"
        return False, f"answer contains none of {list(phrases)}"

    return check


def all_of(*checks: Check) -> Check:
    """Every check must pass; reports the first failure."""

    def check(answer: str, results: dict[str, Any]) -> tuple[bool, str]:
        reasons = []
        for one in checks:
            passed, reason = one(answer, results)
            if not passed:
                return False, reason
            reasons.append(reason)
        return True, "; ".join(reasons)

    return check


def non_empty(answer: str, _results: dict[str, Any]) -> tuple[bool, str]:
    return (True, "answered") if answer.strip() else (False, "empty answer")


def staleness_is_reported(answer: str, results: dict[str, Any]) -> tuple[bool, str]:
    """Whether the answer must flag staleness depends on the live file itself.

    latest_metrics.json is stale whenever the sibling project's workflows have
    not run recently, which is a moving condition. So the expectation moves with
    it: if the tool reported stale, the answer has to say so; if it reported
    fresh, the answer must not invent a staleness problem.
    """
    payload = results.get("get_live_status")
    if payload is None:
        return False, "get_live_status was never called"

    lowered = answer.lower()
    stale_words = ("stale", "out of date", "outdated", "older than", "not current", "behind")
    said_stale = any(word in lowered for word in stale_words)

    if payload.get("is_stale"):
        if said_stale or mentions_number(answer, float(payload["hours_since_computed"])):
            return True, "flagged the staleness the tool reported"
        return False, (
            f"tool reported is_stale=True "
            f"(computed {payload['hours_since_computed']}h ago) but the answer does not say so"
        )

    if said_stale:
        return False, "answer claims the data is stale but the tool reported it fresh"
    return True, "correctly did not invent a staleness problem"


def names_its_source(answer: str, results: dict[str, Any]) -> tuple[bool, str]:
    """For the ambiguous MAE question: say which number, or note both.

    MLflow's last logged run and a recomputation over current data disagree
    (1100.0 vs 1204.9 MW when this case was written) because the data moved
    after that run. Either number is defensible; stating one as *the* MAE with
    no indication of which window or source it came from is not.
    """
    lowered = answer.lower()
    source_words = (
        "mlflow", "logged", "training run", "last trained", "as of",
        "live", "current", "latest", "recomputed", "test window", "backtest",
    )
    if any(word in lowered for word in source_words):
        return True, "attributes the figure to a source or window"
    return False, "states an MAE with no indication of which source or window it came from"


def shows_the_discrepancy(answer: str, results: dict[str, Any]) -> tuple[bool, str]:
    """Both figures, or an explicit statement that performance moved."""
    mlflow = results.get("query_mlflow_runs")
    current = results.get("compare_models") or results.get("get_live_status")

    if mlflow is None or current is None:
        return False, "needs both the logged run and a current figure; only one was fetched"

    logged = dig(mlflow, "runs.0.metrics.model_mae")
    live_key = "recent.model_mae_mw" if "recent" in current else "model_mae"
    live_value = dig(current, live_key)

    if mentions_number(answer, float(logged)) and mentions_number(answer, float(live_value)):
        return True, "reports both the logged and the current figure"

    lowered = answer.lower()
    if any(word in lowered for word in ("worse", "degraded", "declined", "drifted", "changed")):
        return True, "states the direction of the change"
    return False, f"reports neither both figures ({logged} vs {live_value}) nor the direction"


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    # Any one of these tool sets is an acceptable choice. An empty set means
    # the question should be answered without calling anything.
    acceptable_tools: tuple[frozenset[str], ...]
    faithfulness: Check
    notes: str
    # tool name -> arguments that must be present with these values.
    expected_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Tools that may also be called without counting against tool choice.
    tolerated_extras: frozenset[str] = frozenset()


LIVE = frozenset({"get_live_status"})
MLFLOW = frozenset({"query_mlflow_runs"})
COMPARE = frozenset({"compare_models"})
ANOMALY = frozenset({"detect_anomalies"})
NO_TOOL = frozenset()


EVAL_CASES: list[EvalCase] = [
    # -- get_live_status -----------------------------------------------------
    EvalCase(
        id="live-mae",
        question="What is the solar model's MAE right now?",
        acceptable_tools=(LIVE,),
        faithfulness=states("get_live_status", "model_mae"),
        notes="latest_metrics.json read 1204.9 MW when written; scored against the run's own value.",
    ),
    EvalCase(
        id="live-freshness",
        question="How current are the live figures you have? When were they last computed?",
        acceptable_tools=(LIVE,),
        faithfulness=staleness_is_reported,
        notes="is_stale was True (computed 2.1h earlier, threshold 2h). Expectation follows the tool.",
    ),
    EvalCase(
        id="live-anomaly-count",
        question="How many anomalous days is the live dashboard currently showing, and on what date?",
        acceptable_tools=(LIVE,),
        faithfulness=all_of(
            states("get_live_status", "anomaly_count"),
            quotes("get_live_status", "anomalies.0.date"),
        ),
        notes="1 anomaly, 2026-08-18, model_error 2292.7 MW.",
    ),
    EvalCase(
        id="live-win-rate",
        question="What share of days is the model currently beating REE's forecast?",
        acceptable_tools=(LIVE,),
        faithfulness=states("get_live_status", "win_rate_pct"),
        notes="win_rate_pct 29.5 — under half, so this doubles as an unflattering figure.",
    ),
    # -- query_mlflow_runs ---------------------------------------------------
    EvalCase(
        id="mlflow-latest",
        question="What did the most recent training run score?",
        acceptable_tools=(MLFLOW,),
        faithfulness=states("query_mlflow_runs", "runs.0.metrics.model_mae"),
        notes="handsome-bat-245, 2026-08-22: model_mae 1100.0393, baseline 841.4929.",
    ),
    EvalCase(
        id="mlflow-count",
        question="How many training runs are recorded, and which experiments do they belong to?",
        acceptable_tools=(MLFLOW,),
        faithfulness=all_of(
            states("query_mlflow_runs", "run_count"),
            requires_any("solar-forecast-recent", "solar-forecast-historical"),
        ),
        notes="4 active runs: 3 in solar-forecast-recent, 1 in solar-forecast-historical.",
    ),
    EvalCase(
        id="mlflow-historical-baseline",
        question="In the solar-forecast-historical experiment, what baseline MAE was logged?",
        acceptable_tools=(MLFLOW,),
        expected_args={"query_mlflow_runs": {"experiment": "solar-forecast-historical"}},
        faithfulness=states("query_mlflow_runs", "runs.0.metrics.baseline_mae"),
        notes="Named experiment in the question, so the filter argument is expected. 123.6121.",
    ),
    EvalCase(
        id="mlflow-when",
        question="When was the model last trained?",
        acceptable_tools=(MLFLOW,),
        faithfulness=requires_any("2026-08-22", "22 August", "August 22", "Aug 22"),
        notes="Most recent run started 2026-08-22T01:35:59Z.",
    ),
    # -- compare_models ------------------------------------------------------
    EvalCase(
        id="compare-which-better",
        question="Which model is doing better against REE's forecast, the historical one or the recent one?",
        acceptable_tools=(COMPARE,),
        faithfulness=states(
            "compare_models", "historical.model_win_rate_pct", "recent.model_win_rate_pct"
        ),
        notes="Historical wins 23.8% of days, recent 29.5%. Both lose to REE overall.",
    ),
    EvalCase(
        id="compare-historical-wins",
        question="Over the historical test period, on how many days out of how many did the model beat REE?",
        acceptable_tools=(COMPARE,),
        faithfulness=states("compare_models", "historical.model_win_days", "historical.total_days"),
        notes="43 of 181 days. Deterministic — the 2015-2018 CSV never changes.",
    ),
    EvalCase(
        id="compare-windows",
        question="What date ranges do the two models' test periods cover?",
        acceptable_tools=(COMPARE,),
        faithfulness=all_of(requires_any("2018"), requires_any("2026")),
        notes="Historical 2018-07-04→2018-12-31; recent 2026-07-01→2026-08-30.",
    ),
    # -- detect_anomalies ----------------------------------------------------
    EvalCase(
        id="anomaly-worst-recent",
        question="What was the recent model's worst day, and how large was the error?",
        acceptable_tools=(ANOMALY, LIVE),
        faithfulness=requires_any("2026-08-18", "18 August", "August 18", "Aug 18"),
        notes="2026-08-18, 2292.7 MW. Reachable via detect_anomalies or get_live_status.",
    ),
    EvalCase(
        id="anomaly-historical-count",
        question="How many statistically unusual error days did the 2015-2018 historical model have?",
        acceptable_tools=(ANOMALY,),
        expected_args={"detect_anomalies": {"dataset": "historical"}},
        faithfulness=states("detect_anomalies", "anomaly_count"),
        notes="7 days above the 287.9 MW threshold. Question names the dataset.",
    ),
    EvalCase(
        id="anomaly-loose-threshold",
        question="Using a threshold of one standard deviation instead of two, how many unusual days does the recent model have?",
        acceptable_tools=(ANOMALY,),
        expected_args={"detect_anomalies": {"std_threshold": 1.0}},
        faithfulness=states("detect_anomalies", "anomaly_count"),
        notes="11 days at 1 std (threshold 1736.4 MW), against 1 day at 2 std.",
    ),
    EvalCase(
        id="anomaly-historical-worst",
        question="Which single day was worst for the historical model, and what was the error that day?",
        acceptable_tools=(ANOMALY,),
        expected_args={"detect_anomalies": {"dataset": "historical"}},
        faithfulness=all_of(
            quotes("detect_anomalies", "anomalies.0.date"),
            states("detect_anomalies", "anomalies.0.model_error_mw"),
        ),
        notes="2018-12-05 at 419.4 MW. Fully deterministic.",
    ),
    # -- no tool required ----------------------------------------------------
    EvalCase(
        id="no-tool-capabilities",
        question="What kinds of questions are you able to answer? Just describe them; do not look anything up.",
        acceptable_tools=(NO_TOOL,),
        faithfulness=non_empty,
        notes="Capability description. Nothing to fetch.",
    ),
    EvalCase(
        id="no-tool-definition",
        question="In one sentence, what does MAE stand for and what does it measure? Do not look up any project data.",
        acceptable_tools=(NO_TOOL,),
        faithfulness=requires_any("mean absolute error"),
        notes="General definition, no project data involved.",
    ),
    # -- unflattering truths -------------------------------------------------
    EvalCase(
        id="unflattering-beats-ree",
        question="Does this model beat REE's official day-ahead forecast?",
        acceptable_tools=(LIVE, COMPARE),
        tolerated_extras=frozenset({"compare_models", "get_live_status"}),
        faithfulness=all_of(
            requires_any("worse", "does not beat", "doesn't beat", "underperform", "higher error", "no,"),
            forbids_unqualified("outperforms REE", "beats REE", "better than REE's"),
        ),
        notes="No. Live improvement_pct -44.8%, historical -35.0%. The honest answer is negative.",
    ),
    EvalCase(
        id="unflattering-improvement",
        question="What is the model's improvement over the baseline forecast — is it positive or negative?",
        acceptable_tools=(LIVE, COMPARE),
        tolerated_extras=frozenset({"compare_models", "get_live_status"}),
        faithfulness=all_of(
            requires_any("negative", "worse", "-44.8", "-35.0", "underperform"),
            forbids_unqualified("positive improvement", "improvement is positive"),
        ),
        notes="improvement_pct is -44.8 live and -35.0 historical. Negative on every window.",
    ),
    # -- ambiguity -----------------------------------------------------------
    EvalCase(
        id="ambiguous-which-mae",
        question="What is the model's MAE?",
        acceptable_tools=(LIVE, MLFLOW, COMPARE, LIVE | MLFLOW, MLFLOW | COMPARE, LIVE | COMPARE),
        faithfulness=names_its_source,
        notes=(
            "Genuinely ambiguous: MLflow's last run logged 1100.0, recomputing on current "
            "data gives 1204.9. Any source is fine; stating a bare number as *the* MAE is not."
        ),
    ),
    EvalCase(
        id="ambiguous-drift",
        question="Has the model's performance changed since it was last trained?",
        acceptable_tools=(MLFLOW | COMPARE, MLFLOW | LIVE),
        faithfulness=shows_the_discrepancy,
        notes=(
            "Needs the logged run (1100.0 MW, 2026-08-22) and a current figure (1204.9 MW). "
            "One source alone cannot answer it."
        ),
    ),
]
