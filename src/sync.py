"""Pull the sibling checkout before we read anything out of it.

The three data tools in src/tools.py read local files in ~/projects/energy-forecast:
mlflow.db, the ESIOS data dumps, latest_metrics.json. A local checkout that has
not been pulled recently serves confidently wrong answers — we hit exactly that,
with a six-day-old mlflow.db, before this existed.

This mirrors energy-forecast's own sync_repo.py: same command, same 30s timeout,
same contract that a failure degrades to "use whatever local data exists" rather
than raising. It is reimplemented here rather than imported for two reasons: the
import would itself depend on the checkout being present and healthy, which is
the very thing we cannot assume at sync time; and the tools layer needs a
structured result (did the pull actually move HEAD?) that sync_repo.sync()'s
bare bool cannot express.

This is the one place in this project that writes to the sibling repo. It is a
`git pull` and nothing else — no commits, no pushes, no file edits of our own.
Note that with --no-rebase a pull over diverged local commits produces a merge
commit in that repo, which is the behaviour sync_repo.py already has.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src import config

log = logging.getLogger(__name__)

# Matches sync_repo.py, so a hung network fails the same way in both projects.
PULL_TIMEOUT_SECONDS = 30

# The metrics workflow commits every 30 minutes, so pulling more often than this
# buys nothing. The TTL also bounds the cost of a tool-heavy agent turn: several
# tool calls in a row trigger at most one pull.
SYNC_TTL_SECONDS = 300

_last_attempt_at: float | None = None


def sync_enabled() -> bool:
    """Off switch for tests, CI and offline use: REE_ASSISTANT_SYNC=0."""
    return os.environ.get("REE_ASSISTANT_SYNC", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def reset_sync_state() -> None:
    """Forget the last attempt, so the next call pulls. For tests."""
    global _last_attempt_at
    _last_attempt_at = None


def _result(status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, "changed": False, **extra}


def _head_sha(repo: Path) -> str | None:
    """Current HEAD, used only to tell whether a pull actually moved anything."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - never let a bookkeeping call break a read
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sync_energy_forecast(force: bool = False) -> dict[str, Any]:
    """Run `git pull --no-rebase` in the energy-forecast checkout.

    Never raises. Returns a dict with:

    * ``status`` — ok | skipped | disabled | unavailable | failed
    * ``changed`` — True when the pull moved HEAD, so callers know to drop any
      cached computation derived from the old files
    * ``detail`` — a short human-readable reason, for logs

    A failed attempt starts the TTL just as a successful one does. Offline, that
    keeps a burst of tool calls from paying the 30-second timeout every time.
    """
    global _last_attempt_at

    if not sync_enabled():
        return _result("disabled", "REE_ASSISTANT_SYNC is off")

    repo = config.ENERGY_FORECAST_DIR
    if not (repo / ".git").is_dir():
        log.warning(
            "Cannot sync: %s is not a git checkout — using existing local data", repo
        )
        return _result("unavailable", f"{repo} is not a git checkout")

    now = time.monotonic()
    if not force and _last_attempt_at is not None:
        age = now - _last_attempt_at
        if age < SYNC_TTL_SECONDS:
            return _result("skipped", f"last attempt {age:.0f}s ago")

    _last_attempt_at = now
    before = _head_sha(repo)

    try:
        completed = subprocess.run(
            ["git", "pull", "--no-rebase"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=PULL_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - timeout, missing git, bad cwd, ...
        log.warning(
            "git pull in %s failed (%s) — using existing local data", repo, exc
        )
        return _result("failed", str(exc))

    if completed.returncode != 0:
        log.warning(
            "git pull in %s failed: %s — using existing local data",
            repo,
            completed.stderr.strip() or f"exit code {completed.returncode}",
        )
        return _result("failed", completed.stderr.strip() or "git pull returned nonzero")

    after = _head_sha(repo)
    changed = before is not None and after is not None and before != after

    if changed:
        log.info("Synced %s: %s -> %s", repo, before[:8], after[:8])
    else:
        log.debug("Synced %s: already up to date", repo)

    return {
        "status": "ok",
        "detail": "pulled" if changed else "already up to date",
        "changed": changed,
        "head_before": before,
        "head_after": after,
    }
