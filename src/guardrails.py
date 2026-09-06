"""Checks that catch the ways this agent could be wrong while sounding right.

Step 5. These live in src/ rather than in the test file because they are
runtime-usable, not merely test assertions: the Step 6 interface can run
ungrounded_numbers() over an answer before showing it and flag anything the
tools did not actually support.

They target the two gaps the evaluation harness surfaced about itself:

* The eval scores faithfulness with checks written per question — it catches
  contradictions it was told to look for. A fluent answer wrong in a way nobody
  anticipated passes. ungrounded_numbers() inverts that: instead of asking
  "did the answer say what we expected", it asks "did the answer say anything
  the tools never returned", which needs no per-question foresight.
* An answer built on a failed or incomplete tool result must say so.
  acknowledges_failure() checks that it does.

Neither is a proof of correctness. A wrong claim made without numbers, or a
number that coincidentally matches an unrelated field, still gets through.
They close the specific holes we know about.
"""

from __future__ import annotations

import re
from typing import Any

# Dates, year ranges and clock times are identifiers, not measurements.
# Stripping them stops "2026-09-04", "2015-2018" and "09:44" from being read as
# the figures 2026, 2015 or 44. Years are matched last so a full date goes first.
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"  # 2026-09-04, with optional time
    r"|\b\d{1,2}:\d{2}\b"  # 09:44
    r"|\b(?:19|20)\d{2}\s*[-–—/]\s*(?:19|20)?\d{2}\b"  # 2015-2018, 2015-18
    r"|\b(?:19|20)\d{2}\b"  # a bare year
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A number carrying one of these is being stated as a measurement.
_UNIT_AFTER = re.compile(r"\s*(?:%|percent|mw|megawatts?)\b", re.IGNORECASE)

# Below this, a bare integer is far more likely to be prose ("1 in 3 days",
# "the top 5") than a figure read off a tool result.
_BARE_INTEGER_CEILING = 100

_FAILURE_WORDS = (
    "error", "could not", "couldn't", "cannot", "can't", "unable",
    "failed", "failure", "not found", "missing", "no data", "unavailable",
    "does not exist", "doesn't exist", "not available", "no result",
)


def numbers_in(text: str) -> list[tuple[float, int]]:
    """Every number the text states, as (value, decimal places written)."""
    found = []
    for token in _TIMESTAMP.sub(" ", text).split():
        for match in _NUMBER.finditer(token):
            raw = match.group().replace(",", "")
            try:
                found.append((float(raw), len(raw.partition(".")[2])))
            except ValueError:  # pragma: no cover - the regex guarantees a number
                pass
    return found


def mentions_number(text: str, value: float) -> bool:
    """True if the text states this number at any sensible precision.

    Numeric rather than string comparison, because string matching rejects an
    answer *more* precise than the expected value: asked for 123.6121, an agent
    answering "123.61 (specifically 123.6121)" was once scored as omitting it.
    A stated figure counts when the target rounds to it at the precision the
    answer chose, so 1205 matches 1204.9 and 900 does not. Magnitude only —
    direction is a separate question.
    """
    target = abs(float(value))
    for stated, decimals in numbers_in(text):
        if abs(stated - target) <= 0.5 * (10.0**-decimals) + 1e-9:
            return True
    return False


# hours_since_* fields are recomputed at call time, so a value recorded later
# drifts from what the agent was shown. Ground them loosely rather than reading
# the drift as invention.
_DRIFTING_FIELD = re.compile(r"hours_since", re.IGNORECASE)
_DRIFT_TOLERANCE_HOURS = 1.0


def drifting_values(payload: Any) -> list[float]:
    """Values from time-relative fields, which move between calls."""
    found: list[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _DRIFTING_FIELD.search(str(key)) and isinstance(value, (int, float)):
                found.append(float(value))
            else:
                found.extend(drifting_values(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(drifting_values(item))
    return found


def numbers_in_payload(payload: Any) -> list[float]:
    """Every number anywhere in a tool result, however deeply nested."""
    found: list[float] = []
    if isinstance(payload, bool):
        return found
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.extend(numbers_in_payload(value))
            # Numbers inside string fields count too: a date like "2026-08-18"
            # or a window "2018-07-04 → 2018-12-31" is data the answer may quote.
            if isinstance(value, str):
                found.extend(value_ for value_, _ in numbers_in(value))
        return found
    if isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(numbers_in_payload(item))
        return found
    if isinstance(payload, str):
        found.extend(value for value, _ in numbers_in(payload))
    return found


def _is_a_data_claim(value: float, decimals: int, following_text: str) -> bool:
    """Is this number being asserted as a measurement, or is it just prose?

    Rhetorical integers ("1 in 3 days", "the top 5") are not claims about the
    data. A decimal, a large magnitude, or an attached unit means it is.
    """
    if decimals > 0:
        return True
    if abs(value) >= _BARE_INTEGER_CEILING:
        return True
    return bool(_UNIT_AFTER.match(following_text))


def ungrounded_numbers(answer: str, tool_results: dict[str, Any]) -> list[float]:
    """Figures the answer states as data that no tool result contains.

    The core anti-fabrication check. Every measurement-shaped number in the
    answer must trace back to something a tool actually returned; anything that
    does not is either invented or derived in a way the reader cannot audit.

    An empty tool_results means no tool ran, so *every* data claim is
    ungrounded — which is exactly the "answered from memory" failure.
    """
    # Magnitudes only: an improvement of -44.8% is reported as "44.8% worse".
    grounded = {abs(value) for value in numbers_in_payload(tool_results)}

    # Deliberately no arithmetic derivation. Admitting differences between
    # returned figures was measured to halve sensitivity on a real payload
    # (66 numbers became 1,962 grounded values), and admitting ratios collapsed
    # it entirely (5,837 values, at which point fabricated figures like 950 and
    # 1500 both pass). A figure the agent computed rather than read is reported
    # here and judged by a human; that is the cheaper error.

    drifting = [abs(value) for value in drifting_values(tool_results)]
    stripped = _TIMESTAMP.sub(" ", answer)

    suspect = []
    for match in _NUMBER.finditer(stripped):
        raw = match.group().replace(",", "")
        try:
            stated = float(raw)
        except ValueError:  # pragma: no cover
            continue
        decimals = len(raw.partition(".")[2])

        if not _is_a_data_claim(stated, decimals, stripped[match.end():]):
            continue

        stated_magnitude = abs(stated)
        tolerance = 0.5 * (10.0**-decimals) + 1e-9
        if any(abs(stated_magnitude - value) <= tolerance for value in grounded):
            continue
        if any(abs(stated_magnitude - value) <= _DRIFT_TOLERANCE_HOURS for value in drifting):
            continue
        suspect.append(stated)

    return suspect


def acknowledges_failure(answer: str) -> bool:
    """Does the answer name a problem rather than presenting a gap as data?"""
    lowered = answer.lower()
    return any(word in lowered for word in _FAILURE_WORDS)


def minimum_tools_required(acceptable_tool_sets) -> int:
    """The fewest tools any acceptable answer to a question needs."""
    return min((len(group) for group in acceptable_tool_sets), default=0)


def redundant_tool_count(tools_called, acceptable_tool_sets) -> int:
    """How many calls beyond the minimum a question actually needed.

    Calling the right tool plus one it did not need is not a wrong answer, but
    it costs latency and tokens, and it is the tendency the eval caught twice.
    """
    return max(0, len(set(tools_called)) - minimum_tools_required(acceptable_tool_sets))
