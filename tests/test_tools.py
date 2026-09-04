"""Step 2 tests: each tool verified in isolation.

No LLM, no Anthropic client, no network — the tools layer is deliberately
callable and checkable on its own, so a broken tool never has to be diagnosed
through a model's tool call.

The tests fall into two groups:

* Tests that build their own fixtures (a synthetic MLflow database, a synthetic
  daily-error table). These run everywhere, CI included.
* Tests marked ``needs_energy_forecast``, which exercise the tools against the
  real sibling project. They skip cleanly in CI, where energy-forecast is not
  checked out — see src/config.energy_forecast_available.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import config, tools

@pytest.fixture(autouse=True)
def no_git_pull(monkeypatch):
    """Never sync during these tests.

    Two reasons: a pull would reach the network, and the read-only assertions
    below would then be measuring git's writes rather than the tools'. Sync
    behaviour is covered on its own in tests/test_sync.py; the tests here that
    care about the run_tool hook stub it explicitly.
    """
    monkeypatch.setenv("REE_ASSISTANT_SYNC", "0")


needs_energy_forecast = pytest.mark.skipif(
    not config.energy_forecast_available(),
    reason="sibling energy-forecast project not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_summary():
    """Six ordinary days plus one clear error spike on 2026-01-05."""
    return pd.DataFrame(
        {
            "model_error": [100.0, 110.0, 90.0, 105.0, 900.0, 95.0, 100.0],
            "baseline_error": [120.0, 100.0, 130.0, 115.0, 200.0, 90.0, 125.0],
        },
        index=pd.Index(
            pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                ]
            ).date,
            name="date",
        ),
    )


@pytest.fixture
def mlflow_db(tmp_path):
    """A miniature mlflow.db with the same schema the real one uses.

    Lets the MLflow tool be tested for real — including the experiment filter
    and the metric/param joins — without the sibling project present.
    """
    path = tmp_path / "mlflow.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE experiments (
            experiment_id INTEGER PRIMARY KEY,
            name VARCHAR(256) NOT NULL,
            lifecycle_stage VARCHAR(32)
        );
        CREATE TABLE runs (
            run_uuid VARCHAR(32) PRIMARY KEY,
            name VARCHAR(250),
            status VARCHAR(9),
            start_time BIGINT,
            end_time BIGINT,
            lifecycle_stage VARCHAR(20),
            experiment_id INTEGER
        );
        CREATE TABLE latest_metrics (
            key VARCHAR(250) NOT NULL,
            value FLOAT NOT NULL,
            timestamp BIGINT,
            step BIGINT NOT NULL,
            is_nan BOOLEAN NOT NULL,
            run_uuid VARCHAR(32) NOT NULL
        );
        CREATE TABLE params (
            key VARCHAR(250) NOT NULL,
            value VARCHAR(8000) NOT NULL,
            run_uuid VARCHAR(32) NOT NULL
        );
        CREATE TABLE tags (
            key VARCHAR(250) NOT NULL,
            value VARCHAR(8000),
            run_uuid VARCHAR(32) NOT NULL
        );

        INSERT INTO experiments VALUES
            (1, 'solar-forecast-historical', 'active'),
            (2, 'solar-forecast-recent', 'active'),
            (3, 'abandoned-experiment', 'deleted');

        -- newest first: run-recent (2), then run-hist (1); run-deleted is not active
        INSERT INTO runs VALUES
            ('run-recent', 'colorful-foal-301', 'FINISHED', 1787226792000, 1787226795000, 'active', 2),
            ('run-hist',   '',                 'FINISHED', 1787017289000, 1787017294000, 'active', 1),
            ('run-deleted','gone-run-001',     'FINISHED', 1787000000000, 1787000001000, 'deleted', 2);

        INSERT INTO latest_metrics VALUES
            ('baseline_mae',    841.492886, 0, 0, 0, 'run-recent'),
            ('model_mae',      1100.039299, 0, 0, 0, 'run-recent'),
            ('improvement_pct',  -30.724729, 0, 0, 0, 'run-recent'),
            ('baseline_mae',    123.612140, 0, 0, 0, 'run-hist'),
            ('model_mae',       166.905412, 0, 0, 0, 'run-hist'),
            ('nan_metric',        0.0,      0, 0, 1, 'run-hist');

        INSERT INTO params VALUES
            ('model_type', 'GradientBoostingRegressor', 'run-recent'),
            ('test_days',  '60',                        'run-recent'),
            ('model_type', 'GradientBoostingRegressor', 'run-hist');

        -- run-hist has no runs.name, only the tag, like older MLflow runs
        INSERT INTO tags VALUES ('mlflow.runName', 'industrious-roo-243', 'run-hist');
        """
    )
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_daily_error_summary_averages_per_day():
    scored = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-01 11:00",
                    "2026-01-02 10:00",
                ],
                utc=True,
            ),
            "generation solar": [1000.0, 1000.0, 500.0],
            "forecast solar day ahead": [900.0, 1200.0, 450.0],
            "model_pred": [1050.0, 950.0, 600.0],
        }
    )

    daily = tools.daily_error_summary(scored)

    assert list(daily.columns) == ["model_error", "baseline_error"]
    assert len(daily) == 2
    # Day 1: model off by 50 then 50; REE off by 100 then 200.
    assert daily.iloc[0]["model_error"] == pytest.approx(50.0)
    assert daily.iloc[0]["baseline_error"] == pytest.approx(150.0)
    assert daily.iloc[1]["model_error"] == pytest.approx(100.0)


def test_daily_error_summary_does_not_mutate_input():
    scored = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01 10:00"], utc=True),
            "generation solar": [1000.0],
            "forecast solar day ahead": [900.0],
            "model_pred": [1050.0],
        }
    )
    before = list(scored.columns)

    tools.daily_error_summary(scored)

    assert list(scored.columns) == before


def test_win_rate_counts_days_the_model_beats_ree(daily_summary):
    result = tools.win_rate(daily_summary)

    # Model wins on 4 of the 7 days (loses 01-02, the 01-05 spike, and 01-06).
    assert result == {
        "model_win_days": 4,
        "total_days": 7,
        "model_win_rate_pct": pytest.approx(57.1),
    }


def test_win_rate_handles_an_empty_summary():
    empty = pd.DataFrame(columns=["model_error", "baseline_error"])

    assert tools.win_rate(empty) == {
        "model_win_days": 0,
        "total_days": 0,
        "model_win_rate_pct": None,
    }


def test_improvement_pct_sign_and_guard():
    # Positive when the model's error is smaller than REE's.
    assert tools._improvement_pct(baseline_mae=100.0, model_mae=80.0) == pytest.approx(20.0)
    assert tools._improvement_pct(baseline_mae=100.0, model_mae=130.0) == pytest.approx(-30.0)
    assert tools._improvement_pct(baseline_mae=0.0, model_mae=10.0) is None


def test_iso_converts_mlflow_epoch_milliseconds():
    assert tools._iso(1787226792279) == "2026-08-20T11:53:12.279000+00:00"
    assert tools._iso(None) is None


# ---------------------------------------------------------------------------
# Tool 1: detect_anomalies
# ---------------------------------------------------------------------------


def test_detect_anomalies_rejects_a_bad_dataset():
    with pytest.raises(tools.ToolError, match="unknown dataset"):
        tools.detect_anomalies(dataset="yesterday")


def test_detect_anomalies_rejects_a_non_positive_threshold():
    with pytest.raises(tools.ToolError, match="std_threshold"):
        tools.detect_anomalies(std_threshold=0)


@needs_energy_forecast
def test_detect_anomalies_matches_the_wrapped_function(daily_summary):
    """Our payload must report exactly what the sibling function decided."""
    _, anomaly_detection = tools._energy_forecast_modules()

    anomalies, threshold = anomaly_detection.detect_anomalies(daily_summary, std_threshold=2)

    # mean 214.3 + 2 * std 301.6 -> only the 900 MW spike clears the bar.
    assert threshold == pytest.approx(
        daily_summary["model_error"].mean() + 2 * daily_summary["model_error"].std()
    )
    assert list(anomalies.index) == [pd.Timestamp("2026-01-05").date()]


@needs_energy_forecast
def test_detect_anomalies_returns_a_serialisable_payload():
    result = tools.detect_anomalies(dataset="recent")

    assert result["dataset"] == "recent"
    assert result["days_analysed"] > 0
    assert result["std_threshold"] == 2.0
    assert result["threshold_mw"] > result["mean_daily_error_mw"]
    assert result["anomaly_count"] == len(result["anomalies"])
    assert result["anomaly_count"] <= result["days_analysed"]
    for row in result["anomalies"]:
        assert set(row) == {"date", "model_error_mw", "baseline_error_mw"}
        assert row["model_error_mw"] > result["threshold_mw"]

    json.dumps(result)  # must survive the trip back to the model


@needs_energy_forecast
def test_detect_anomalies_lower_threshold_flags_at_least_as_many_days():
    strict = tools.detect_anomalies(dataset="historical", std_threshold=2.0)
    loose = tools.detect_anomalies(dataset="historical", std_threshold=1.0)

    assert loose["threshold_mw"] < strict["threshold_mw"]
    assert loose["anomaly_count"] >= strict["anomaly_count"]


@needs_energy_forecast
def test_detect_anomalies_writes_no_anomaly_log():
    """record_anomalies() must never be reached — it appends to disk."""
    log_path = config.ENERGY_FORECAST_DIR / "anomaly_log.json"
    before = log_path.read_bytes() if log_path.exists() else None

    tools.detect_anomalies(dataset="recent")

    after = log_path.read_bytes() if log_path.exists() else None
    assert after == before


# ---------------------------------------------------------------------------
# Tool 2: query_mlflow_runs
# ---------------------------------------------------------------------------


def test_query_mlflow_runs_reads_runs_metrics_and_params(mlflow_db):
    result = tools.query_mlflow_runs(db_path=mlflow_db)

    assert result["run_count"] == 2  # the deleted run is excluded
    assert result["experiment_filter"] is None
    # Deleted experiments are not offered as filter options.
    assert result["experiments_available"] == [
        "solar-forecast-historical",
        "solar-forecast-recent",
    ]

    newest, older = result["runs"]
    assert newest["run_id"] == "run-recent"  # most recent first
    assert older["run_id"] == "run-hist"

    assert newest["run_name"] == "colorful-foal-301"
    assert newest["experiment"] == "solar-forecast-recent"
    assert newest["status"] == "FINISHED"
    assert newest["duration_seconds"] == pytest.approx(3.0)
    assert newest["params"] == {
        "model_type": "GradientBoostingRegressor",
        "test_days": "60",
    }
    assert newest["metrics"] == {
        "baseline_mae": pytest.approx(841.4929),
        "improvement_pct": pytest.approx(-30.7247),
        "model_mae": pytest.approx(1100.0393),
    }

    json.dumps(result)


def test_query_mlflow_runs_falls_back_to_the_run_name_tag(mlflow_db):
    result = tools.query_mlflow_runs(experiment="solar-forecast-historical", db_path=mlflow_db)

    (run,) = result["runs"]
    assert run["run_name"] == "industrious-roo-243"
    # is_nan metrics are dropped rather than emitted as a bogus 0.0.
    assert "nan_metric" not in run["metrics"]


def test_query_mlflow_runs_filters_by_experiment(mlflow_db):
    result = tools.query_mlflow_runs(experiment="solar-forecast-recent", db_path=mlflow_db)

    assert result["experiment_filter"] == "solar-forecast-recent"
    assert [run["run_id"] for run in result["runs"]] == ["run-recent"]


def test_query_mlflow_runs_respects_the_limit(mlflow_db):
    result = tools.query_mlflow_runs(limit=1, db_path=mlflow_db)

    assert [run["run_id"] for run in result["runs"]] == ["run-recent"]


def test_query_mlflow_runs_names_the_valid_experiments_when_asked_for_a_bad_one(mlflow_db):
    with pytest.raises(tools.ToolError, match="solar-forecast-recent"):
        tools.query_mlflow_runs(experiment="no-such-experiment", db_path=mlflow_db)


def test_query_mlflow_runs_rejects_a_bad_limit(mlflow_db):
    with pytest.raises(tools.ToolError, match="limit"):
        tools.query_mlflow_runs(limit=0, db_path=mlflow_db)


def test_query_mlflow_runs_reports_a_missing_database(tmp_path):
    with pytest.raises(tools.ToolError, match="mlflow.db not found"):
        tools.query_mlflow_runs(db_path=tmp_path / "nope.db")


def test_query_mlflow_runs_cannot_write_to_the_database(mlflow_db):
    """The connection is opened mode=ro, so a write is impossible by construction."""
    conn = tools._read_only_connection(mlflow_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM runs")
    finally:
        conn.close()


def test_query_mlflow_runs_leaves_the_database_byte_identical(mlflow_db):
    before = mlflow_db.read_bytes()

    tools.query_mlflow_runs(db_path=mlflow_db)

    assert mlflow_db.read_bytes() == before


@needs_energy_forecast
def test_query_mlflow_runs_against_the_real_database():
    result = tools.query_mlflow_runs()

    assert result["database"] == str(config.MLFLOW_DB)
    assert "solar-forecast-recent" in result["experiments_available"]
    assert result["run_count"] >= 1
    for run in result["runs"]:
        assert run["run_id"]
        assert run["experiment"]
        assert run["start_time"].startswith("20")

    json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 3: compare_models
# ---------------------------------------------------------------------------


@needs_energy_forecast
def test_compare_models_reports_both_sides():
    result = tools.compare_models()

    for side, expected_days in (("historical", 180), ("recent", 60)):
        report = result[side]
        assert report["dataset"] == side
        assert report["model_type"] == "GradientBoostingRegressor"
        assert report["features"] == tools.FEATURES
        assert report["train_rows"] > 0
        assert report["test_rows"] > 0
        assert report["model_mae_mw"] > 0
        assert report["baseline_mae_mw"] > 0
        # time_based_split cuts the last N days, so the test window is ~N days.
        assert expected_days <= report["total_days"] <= expected_days + 2
        assert 0 <= report["model_win_days"] <= report["total_days"]

    assert result["comparison"]["improvement_pct_delta"] == pytest.approx(
        result["recent"]["improvement_pct"] - result["historical"]["improvement_pct"],
        abs=0.1,
    )
    assert "not comparable" in result["caveat"]

    json.dumps(result)


@needs_energy_forecast
def test_compare_models_improvement_pct_agrees_with_the_maes():
    result = tools.compare_models()

    for side in ("historical", "recent"):
        report = result[side]
        expected = (
            (report["baseline_mae_mw"] - report["model_mae_mw"])
            / report["baseline_mae_mw"]
            * 100
        )
        assert report["improvement_pct"] == pytest.approx(expected, abs=0.1)


@needs_energy_forecast
def test_compare_models_reproduces_the_logged_historical_mlflow_run():
    """The historical side must match what train_model.py recorded in MLflow.

    That run logged model_mae 166.9 / baseline_mae 123.6. If our reimplementation
    of its logic drifts, this is where it shows up.
    """
    runs = tools.query_mlflow_runs(experiment="solar-forecast-historical", limit=1)
    if not runs["runs"]:
        pytest.skip("no historical run logged in mlflow.db")

    logged = runs["runs"][0]["metrics"]
    historical = tools.compare_models()["historical"]

    assert historical["model_mae_mw"] == pytest.approx(logged["model_mae"], abs=0.5)
    assert historical["baseline_mae_mw"] == pytest.approx(logged["baseline_mae"], abs=0.5)


@needs_energy_forecast
def test_scored_test_sets_are_disjoint_from_training_rows():
    for dataset in tools.DATASETS:
        scored, meta = tools._scored_test_set(dataset)
        assert meta["test_start"] < meta["test_end"]
        assert len(scored) == meta["test_rows"]
        assert scored["model_pred"].notna().all()


@needs_energy_forecast
def test_compare_models_does_not_touch_the_energy_forecast_project():
    """Running the tools must not add, remove or modify any sibling file.

    Scoped to the tools themselves: the autouse fixture keeps sync off, since
    src/sync.py's git pull is the one sanctioned write to that checkout.
    """
    watched = sorted(
        p for p in config.ENERGY_FORECAST_DIR.rglob("*")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
    )
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched}

    tools.compare_models()
    tools.detect_anomalies()
    tools.query_mlflow_runs()

    after = {
        p: (p.stat().st_mtime_ns, p.stat().st_size)
        for p in watched
        if p.exists()
    }
    assert after == before


# ---------------------------------------------------------------------------
# The registry the agent loop will use
# ---------------------------------------------------------------------------


def test_every_schema_has_a_function_and_vice_versa():
    schema_names = [schema["name"] for schema in tools.TOOL_SCHEMAS]

    assert sorted(schema_names) == sorted(tools.TOOL_FUNCTIONS)
    assert len(schema_names) == len(set(schema_names)) == 4


def test_schemas_are_well_formed_anthropic_tool_definitions():
    for schema in tools.TOOL_SCHEMAS:
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["description"].strip()
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert isinstance(input_schema["properties"], dict)
        assert isinstance(input_schema["required"], list)
        for prop in input_schema["properties"].values():
            assert prop["type"] in {"string", "integer", "number", "boolean"}
            assert prop.get("description", "").strip()

    json.dumps(tools.TOOL_SCHEMAS)


def test_schema_properties_are_real_parameters_of_the_function():
    import inspect

    for schema in tools.TOOL_SCHEMAS:
        signature = inspect.signature(tools.TOOL_FUNCTIONS[schema["name"]])
        for name in schema["input_schema"]["properties"]:
            assert name in signature.parameters, f"{schema['name']}.{name}"
            # Every documented parameter is optional, so the model may omit it.
            assert signature.parameters[name].default is not inspect.Parameter.empty


def test_run_tool_reports_an_unknown_tool_instead_of_raising():
    result = tools.run_tool("summon_more_solar")

    assert "unknown tool" in result["error"]


def test_run_tool_reports_bad_arguments_instead_of_raising():
    assert "bad arguments" in tools.run_tool("compare_models", {"when": "now"})["error"]


def test_run_tool_converts_tool_errors_into_a_readable_payload():
    result = tools.run_tool("detect_anomalies", {"dataset": "tomorrow"})

    assert "unknown dataset" in result["error"]


def test_run_tool_accepts_a_missing_input_dict(mlflow_db, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_DB", mlflow_db)

    result = tools.run_tool("query_mlflow_runs")

    assert result["run_count"] == 2


# ---------------------------------------------------------------------------
# Tool 4: get_live_status
# ---------------------------------------------------------------------------

# A copy of what refresh_metrics.yml commits, with every documented field.
LIVE_METRICS = {
    "computed_at": "2026-09-02T12:51:04.297087+00:00",
    "live_data_fetched_at": "2026-09-02T11:10:09.378898+00:00",
    "hours_since_live_fetch": 1.7,
    "model_mae": 1204.9,
    "baseline_mae": 832.1,
    "improvement_pct": -44.8,
    "win_rate_pct": 29.5,
    "anomaly_count": 1,
    "anomalies": [{"date": "2026-08-18", "model_error": 2292.7}],
}


@pytest.fixture
def live_metrics_file(tmp_path):
    """latest_metrics.json as the workflow writes it, computed just now."""

    def write(**overrides):
        payload = {**LIVE_METRICS, **overrides}
        payload.setdefault(
            "computed_at", datetime.now(timezone.utc).isoformat()
        )
        path = tmp_path / "latest_metrics.json"
        path.write_text(json.dumps(payload))
        return path

    return write


def test_get_live_status_returns_the_file_contents(live_metrics_file):
    path = live_metrics_file(computed_at=datetime.now(timezone.utc).isoformat())

    result = tools.get_live_status(path=path)

    # Every field the workflow publishes comes back untouched.
    for key, value in LIVE_METRICS.items():
        if key == "computed_at":
            continue
        assert result[key] == value

    assert result["anomalies"] == [{"date": "2026-08-18", "model_error": 2292.7}]
    assert result["source_file"] == str(path)
    assert result["file_modified_utc"].startswith("20")

    json.dumps(result)


def test_get_live_status_reports_fresh_metrics_as_fresh(live_metrics_file):
    path = live_metrics_file(
        computed_at=datetime.now(timezone.utc).isoformat(),
        hours_since_live_fetch=0.4,
    )

    result = tools.get_live_status(path=path)

    assert result["is_stale"] is False
    assert result["hours_since_computed"] == pytest.approx(0.0, abs=0.1)
    assert "staleness_note" not in result


def test_get_live_status_flags_a_file_the_workflow_stopped_updating(live_metrics_file):
    """hours_since_live_fetch was fine when written — the file itself went stale."""
    stale = datetime.now(timezone.utc) - timedelta(hours=30)
    path = live_metrics_file(computed_at=stale.isoformat(), hours_since_live_fetch=0.3)

    result = tools.get_live_status(path=path)

    assert result["is_stale"] is True
    assert result["hours_since_computed"] == pytest.approx(30.0, abs=0.2)
    assert "computed_at" in result["staleness_note"]


def test_get_live_status_flags_stale_live_data(live_metrics_file):
    path = live_metrics_file(
        computed_at=datetime.now(timezone.utc).isoformat(),
        hours_since_live_fetch=9.0,
    )

    result = tools.get_live_status(path=path)

    assert result["is_stale"] is True


def test_get_live_status_passes_through_fields_added_later(live_metrics_file):
    """The workflow may grow new keys; this tool must not need editing for that."""
    path = live_metrics_file(
        computed_at=datetime.now(timezone.utc).isoformat(),
        forecast_horizon_hours=48,
    )

    assert tools.get_live_status(path=path)["forecast_horizon_hours"] == 48


def test_get_live_status_reports_a_missing_file(tmp_path):
    with pytest.raises(tools.ToolError) as exc_info:
        tools.get_live_status(path=tmp_path / "latest_metrics.json")

    message = str(exc_info.value)
    assert "latest_metrics.json not found" in message
    # The message has to say what to do about it, since the model relays it.
    assert "refresh_metrics.yml" in message
    assert "pulled" in message


def test_get_live_status_reports_a_corrupt_file(tmp_path):
    path = tmp_path / "latest_metrics.json"
    path.write_text("{ half a file")

    with pytest.raises(tools.ToolError, match="not valid JSON"):
        tools.get_live_status(path=path)


def test_get_live_status_rejects_a_json_file_that_is_not_an_object(tmp_path):
    path = tmp_path / "latest_metrics.json"
    path.write_text("[1, 2, 3]")

    with pytest.raises(tools.ToolError, match="should contain a JSON object"):
        tools.get_live_status(path=path)


def test_get_live_status_survives_an_unparseable_computed_at(live_metrics_file):
    path = live_metrics_file(computed_at="not a timestamp")

    result = tools.get_live_status(path=path)

    assert result["hours_since_computed"] is None
    # hours_since_live_fetch is still usable, so a verdict is still possible.
    assert result["is_stale"] is False


@needs_energy_forecast
def test_get_live_status_against_the_real_file():
    result = tools.get_live_status()

    assert result["source_file"].endswith("latest_metrics.json")
    for key in ("model_mae", "baseline_mae", "win_rate_pct", "anomaly_count"):
        assert isinstance(result[key], (int, float))
    assert isinstance(result["anomalies"], list)
    assert len(result["anomalies"]) == result["anomaly_count"]
    assert result["is_stale"] in (True, False)

    json.dumps(result)


# ---------------------------------------------------------------------------
# run_tool syncs before reading
# ---------------------------------------------------------------------------


def test_run_tool_syncs_before_dispatching(live_metrics_file, monkeypatch):
    calls = []
    monkeypatch.setattr(
        tools.sync,
        "sync_energy_forecast",
        lambda *a, **kw: calls.append("sync") or {"status": "ok", "changed": False},
    )
    path = live_metrics_file(computed_at=datetime.now(timezone.utc).isoformat())

    result = tools.run_tool("get_live_status", {"path": str(path)})

    assert calls == ["sync"]
    assert result["model_mae"] == LIVE_METRICS["model_mae"]


def test_run_tool_clears_caches_when_the_sync_pulled_new_commits(
    live_metrics_file, monkeypatch
):
    """A pull can replace the files a cached result was computed from."""
    cleared = []
    monkeypatch.setattr(
        tools.sync, "sync_energy_forecast", lambda *a, **kw: {"status": "ok", "changed": True}
    )
    monkeypatch.setattr(tools, "clear_caches", lambda: cleared.append("cleared"))
    path = live_metrics_file(computed_at=datetime.now(timezone.utc).isoformat())

    tools.run_tool("get_live_status", {"path": str(path)})

    assert cleared == ["cleared"]


def test_run_tool_leaves_caches_alone_when_nothing_changed(live_metrics_file, monkeypatch):
    cleared = []
    monkeypatch.setattr(
        tools.sync, "sync_energy_forecast", lambda *a, **kw: {"status": "ok", "changed": False}
    )
    monkeypatch.setattr(tools, "clear_caches", lambda: cleared.append("cleared"))
    path = live_metrics_file(computed_at=datetime.now(timezone.utc).isoformat())

    tools.run_tool("get_live_status", {"path": str(path)})

    assert cleared == []


def test_run_tool_still_answers_when_the_sync_fails(live_metrics_file, monkeypatch):
    """A failed pull must degrade to local data, not to a failed tool call."""
    monkeypatch.setattr(
        tools.sync,
        "sync_energy_forecast",
        lambda *a, **kw: {"status": "failed", "changed": False, "detail": "offline"},
    )
    path = live_metrics_file(computed_at=datetime.now(timezone.utc).isoformat())

    result = tools.run_tool("get_live_status", {"path": str(path)})

    assert "error" not in result
    assert result["win_rate_pct"] == LIVE_METRICS["win_rate_pct"]


def test_clear_caches_empties_the_scored_test_set_cache():
    tools.clear_caches()

    assert tools._scored_test_set.cache_info().currsize == 0
