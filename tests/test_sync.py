"""Tests for the pre-read sync of the energy-forecast checkout.

No git process is ever launched here: every test replaces subprocess.run, so the
suite neither needs a network nor touches the sibling repo. What is being tested
is the contract, not git — above all that a failed pull degrades to "carry on
with local data" instead of taking the caller down with it.
"""

import logging
import subprocess

import pytest

from src import config, sync


@pytest.fixture(autouse=True)
def fresh_sync_state(monkeypatch):
    """Each test starts with no TTL in effect and syncing switched on."""
    monkeypatch.setenv("REE_ASSISTANT_SYNC", "1")
    sync.reset_sync_state()
    yield
    sync.reset_sync_state()


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A directory that looks enough like a git checkout to get past the guard."""
    repo = tmp_path / "energy-forecast"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(config, "ENERGY_FORECAST_DIR", repo)
    return repo


def _fake_run(calls, *, returncode=0, stderr="", head="abc1234", exc=None):
    """Stand-in for subprocess.run that records calls and fakes git's answers."""

    def run(cmd, **kwargs):
        calls.append({"cmd": cmd, "cwd": kwargs.get("cwd"), "timeout": kwargs.get("timeout")})
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{head}\n", stderr="")
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    return run


# ---------------------------------------------------------------------------
# Graceful degradation — the point of the whole module
# ---------------------------------------------------------------------------


def test_sync_degrades_gracefully_when_git_pull_fails(fake_repo, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run(calls, returncode=1, stderr="fatal: could not read from remote repository"),
    )

    with caplog.at_level(logging.WARNING, logger="src.sync"):
        result = sync.sync_energy_forecast()

    # No exception reached the caller.
    assert result["status"] == "failed"
    assert result["changed"] is False
    assert "could not read from remote" in result["detail"]

    # And it said so, loudly enough to find in a log.
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "could not read from remote" in caplog.text
    assert "using existing local data" in caplog.text


def test_sync_degrades_gracefully_when_git_is_missing(fake_repo, monkeypatch, caplog):
    monkeypatch.setattr(
        subprocess, "run", _fake_run([], exc=FileNotFoundError("No such file: 'git'"))
    )

    with caplog.at_level(logging.WARNING, logger="src.sync"):
        result = sync.sync_energy_forecast()

    assert result["status"] == "failed"
    assert "git" in result["detail"]
    assert "using existing local data" in caplog.text


def test_sync_degrades_gracefully_when_git_pull_hangs(fake_repo, monkeypatch, caplog):
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run([], exc=subprocess.TimeoutExpired(cmd="git pull", timeout=30)),
    )

    with caplog.at_level(logging.WARNING, logger="src.sync"):
        result = sync.sync_energy_forecast()

    assert result["status"] == "failed"
    assert "using existing local data" in caplog.text


def test_sync_reports_unavailable_when_the_sibling_is_not_a_checkout(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(config, "ENERGY_FORECAST_DIR", tmp_path / "not-a-repo")

    with caplog.at_level(logging.WARNING, logger="src.sync"):
        result = sync.sync_energy_forecast()

    assert result["status"] == "unavailable"
    assert result["changed"] is False
    assert "not a git checkout" in caplog.text


# ---------------------------------------------------------------------------
# The happy path and the command actually issued
# ---------------------------------------------------------------------------


def test_sync_pulls_with_no_rebase_in_the_sibling_directory(fake_repo, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    sync.sync_energy_forecast()

    pulls = [call for call in calls if call["cmd"][:2] == ["git", "pull"]]
    assert len(pulls) == 1
    # Mirrors sync_repo.py exactly: same flag, same directory, same timeout.
    assert pulls[0]["cmd"] == ["git", "pull", "--no-rebase"]
    assert pulls[0]["cwd"] == fake_repo
    assert pulls[0]["timeout"] == sync.PULL_TIMEOUT_SECONDS


def test_sync_reports_no_change_when_head_stands_still(fake_repo, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run([], head="same111"))

    result = sync.sync_energy_forecast()

    assert result["status"] == "ok"
    assert result["changed"] is False
    assert result["detail"] == "already up to date"


def test_sync_reports_a_change_when_the_pull_moves_head(fake_repo, monkeypatch):
    heads = iter(["oldsha11", "newsha22"])  # rev-parse runs once before, once after

    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=next(heads) + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)

    result = sync.sync_energy_forecast()

    assert result["status"] == "ok"
    assert result["changed"] is True
    assert result["head_before"] == "oldsha11"
    assert result["head_after"] == "newsha22"


# ---------------------------------------------------------------------------
# Rate limiting and the off switch
# ---------------------------------------------------------------------------


def test_sync_is_skipped_inside_the_ttl(fake_repo, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    first = sync.sync_energy_forecast()
    second = sync.sync_energy_forecast()

    assert first["status"] == "ok"
    assert second["status"] == "skipped"
    # A burst of tool calls in one agent turn costs exactly one pull.
    assert len([c for c in calls if c["cmd"][:2] == ["git", "pull"]]) == 1


def test_sync_force_overrides_the_ttl(fake_repo, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))

    sync.sync_energy_forecast()
    result = sync.sync_energy_forecast(force=True)

    assert result["status"] == "ok"
    assert len([c for c in calls if c["cmd"][:2] == ["git", "pull"]]) == 2


def test_a_failed_sync_also_starts_the_ttl(fake_repo, monkeypatch):
    """Offline, we must not pay the 30s timeout on every tool call."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls, returncode=1, stderr="offline"))

    assert sync.sync_energy_forecast()["status"] == "failed"
    assert sync.sync_energy_forecast()["status"] == "skipped"
    assert len([c for c in calls if c["cmd"][:2] == ["git", "pull"]]) == 1


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_sync_can_be_switched_off(fake_repo, monkeypatch, value):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    monkeypatch.setenv("REE_ASSISTANT_SYNC", value)

    result = sync.sync_energy_forecast()

    assert result["status"] == "disabled"
    assert calls == []


def test_sync_is_on_by_default(monkeypatch):
    monkeypatch.delenv("REE_ASSISTANT_SYNC", raising=False)

    assert sync.sync_enabled() is True
