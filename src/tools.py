"""The tools layer: read-only functions the assistant can call.

Step 2 of the build. Each public function here wraps one piece of the sibling
Spain solar forecasting project at ~/projects/energy-forecast and returns a
plain, JSON-serialisable dict — the shape an LLM tool result needs. There is no
Anthropic client in this module and no network access; the agent loop that calls
these lives in a later step, and the tools are fully usable (and tested) without
it.

Three tools:

* ``detect_anomalies``   — wraps ``anomaly_detection.detect_anomalies``
* ``query_mlflow_runs``  — reads the MLflow run history from ``mlflow.db``
* ``compare_models``     — historical vs recent model, using the logic in
                           ``train_model.py`` / ``train_model_recent.py`` /
                           ``dashboard.py``

Read-only, deliberately
-----------------------
Nothing here writes to the energy-forecast project. Three specific precautions,
since the obvious implementations would all violate that:

1. We import ``features`` and ``anomaly_detection`` (pure function libraries)
   but never ``train_model.py``, ``train_model_recent.py`` or ``dashboard.py``.
   Those run their work at import time — importing them would retrain models
   and write new runs into mlflow.db.
2. We never call ``anomaly_detection.record_anomalies``: it appends to
   ``anomaly_log.json``. Only the pure ``detect_anomalies`` is wrapped.
3. mlflow.db is read with sqlite3 in ``mode=ro`` rather than through the MLflow
   client. MLflow's SQLAlchemy store will happily run schema migrations against
   a database it opens, which is a write we must not make to another project's
   file. A read-only connection makes that structurally impossible.

Paths to energy-forecast files are resolved to absolute paths before being
handed to the sibling project's functions, so nothing depends on the current
working directory.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import config

# Mirrors the feature list in train_model.py, train_model_recent.py and
# dashboard.py. Kept as a constant here rather than imported, because the
# modules that define it execute training at import time.
FEATURES = [
    "hour",
    "day_of_week",
    "solar_lag_24h",
    "wind_lag_24h",
    "solar_rolling_3h",
]

# time_based_split defaults used by each side of the comparison.
HISTORICAL_TEST_DAYS = 180  # features.time_based_split default, as train_model.py uses
RECENT_TEST_DAYS = 60  # train_model_recent.py / dashboard.py

DATASETS = ("recent", "historical")

# Keeps a tool result small enough to hand back to a model without flooding it.
MAX_ANOMALY_ROWS = 50
MAX_RUNS = 100


class ToolError(RuntimeError):
    """A tool could not run — missing sibling project, bad argument, etc.

    Raised by the tools so tests can assert on it, and converted into an
    ``{"error": ...}`` payload by :func:`run_tool` for the agent loop.
    """


# ---------------------------------------------------------------------------
# Access to the sibling project
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _energy_forecast_modules():
    """Import the sibling project's pure helper modules.

    Appended (not prepended) to sys.path so energy-forecast's generically named
    modules can't shadow anything already importable in this project.
    """
    if not config.ENERGY_FORECAST_DIR.is_dir():
        raise ToolError(
            f"energy-forecast project not found at {config.ENERGY_FORECAST_DIR}. "
            "Set ENERGY_FORECAST_DIR to its location."
        )

    path = str(config.ENERGY_FORECAST_DIR)
    if path not in sys.path:
        sys.path.append(path)

    try:
        import anomaly_detection  # noqa: PLC0415 — deliberately deferred
        import features  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on sibling checkout
        raise ToolError(f"could not import energy-forecast modules: {exc}") from exc

    return features, anomaly_detection


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise ToolError(f"{what} not found at {path}")
    return path


# ---------------------------------------------------------------------------
# Pure helpers (no energy-forecast dependency — unit-testable anywhere)
# ---------------------------------------------------------------------------


def daily_error_summary(scored):
    """Per-day mean absolute error for the model and for REE's forecast.

    ``scored`` is a test-set DataFrame carrying ``time``, ``generation solar``,
    ``forecast solar day ahead`` and ``model_pred``. This is the aggregation
    dashboard.py performs before charting or looking for anomalies, and it is
    the input ``anomaly_detection.detect_anomalies`` expects.
    """
    import pandas as pd  # noqa: PLC0415 — keeps module import cheap

    scored = scored.copy()
    scored["model_error"] = (scored["generation solar"] - scored["model_pred"]).abs()
    scored["baseline_error"] = (
        scored["generation solar"] - scored["forecast solar day ahead"]
    ).abs()
    scored["date"] = pd.to_datetime(scored["time"]).dt.date
    return scored.groupby("date")[["model_error", "baseline_error"]].mean()


def win_rate(daily_summary) -> dict[str, Any]:
    """How often the model beat REE's day-ahead forecast, counted by day."""
    total_days = int(len(daily_summary))
    wins = int((daily_summary["model_error"] < daily_summary["baseline_error"]).sum())
    return {
        "model_win_days": wins,
        "total_days": total_days,
        "model_win_rate_pct": round(wins / total_days * 100, 1) if total_days else None,
    }


def _improvement_pct(baseline_mae: float, model_mae: float) -> float | None:
    """Positive means the model beats REE's forecast. Matches train_model.py."""
    if not baseline_mae:
        return None
    return round((baseline_mae - model_mae) / baseline_mae * 100, 1)


def _iso(ms: int | None) -> str | None:
    """MLflow stores epoch milliseconds; the model reads dates far better."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Scoring the two models (shared by compare_models and detect_anomalies)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _scored_test_set(dataset: str):
    """Rebuild a model's test set with predictions attached.

    Returns ``(scored_dataframe, metadata)``. Cached because both tools need it
    and the historical side trains a model (~2s). Follows dashboard.py's
    ``get_historical_model_and_test`` / ``get_recent_model_and_test``.
    """
    if dataset not in DATASETS:
        raise ToolError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")

    features, _ = _energy_forecast_modules()

    if dataset == "historical":
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

        csv = _require(
            config.ENERGY_FORECAST_DIR / "data" / "energy_dataset.csv",
            "historical dataset",
        )
        df_clean = features.load_and_engineer(str(csv))
        train, test = features.time_based_split(df_clean, test_days=HISTORICAL_TEST_DAYS)

        # train_model.py trains this from scratch each run and saves no artifact,
        # so there is nothing to load — we retrain in memory with the same seed.
        model = GradientBoostingRegressor(random_state=42)
        model.fit(train[FEATURES], train["generation solar"])
        model_source = "retrained in memory (train_model.py logs no reusable artifact)"
        trained_on = "2015-2018 Kaggle historical dataset"
    else:
        recent_json = _require(
            config.ENERGY_FORECAST_DIR / "data" / "recent_solar_data.json",
            "recent ESIOS dataset",
        )
        with open(recent_json) as f:
            recent_data = json.load(f)

        df_clean = features.build_live_features(recent_data)
        train, test = features.time_based_split(df_clean, test_days=RECENT_TEST_DAYS)

        if config.RECENT_MODEL_PKL.exists():
            import joblib  # noqa: PLC0415

            model = joblib.load(config.RECENT_MODEL_PKL)
            model_source = f"loaded from {config.RECENT_MODEL_PKL.name}"
        else:
            # Same fit train_model_recent.py performs, minus the joblib.dump.
            from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

            model = GradientBoostingRegressor(random_state=42)
            model.fit(train[FEATURES], train["generation solar"])
            model_source = "retrained in memory (recent_model.pkl absent)"
        trained_on = "last ~1 year of REE ESIOS data"

    if test.empty:
        raise ToolError(f"{dataset} test split is empty — check the source data")

    test = test.copy()
    test["model_pred"] = model.predict(test[FEATURES])

    meta = {
        "dataset": dataset,
        "trained_on": trained_on,
        "model_source": model_source,
        "model_type": type(model).__name__,
        "features": list(FEATURES),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_start": str(test["time"].min()),
        "test_end": str(test["time"].max()),
    }
    return test, meta


def _model_report(dataset: str) -> dict[str, Any]:
    """One side of the comparison: MAE, improvement over REE, daily win rate."""
    scored, meta = _scored_test_set(dataset)
    daily = daily_error_summary(scored)

    model_mae = float((scored["generation solar"] - scored["model_pred"]).abs().mean())
    baseline_mae = float(
        (scored["generation solar"] - scored["forecast solar day ahead"]).abs().mean()
    )

    return {
        **meta,
        "model_mae_mw": round(model_mae, 1),
        "baseline_mae_mw": round(baseline_mae, 1),
        "improvement_pct": _improvement_pct(baseline_mae, model_mae),
        **win_rate(daily),
    }


# ---------------------------------------------------------------------------
# Tool 1: anomaly detection
# ---------------------------------------------------------------------------


def detect_anomalies(dataset: str = "recent", std_threshold: float = 2.0) -> dict[str, Any]:
    """Flag days where the model's error was statistically unusual.

    Wraps ``anomaly_detection.detect_anomalies`` from the energy-forecast
    project: a day is anomalous when its mean model error exceeds
    ``mean + std_threshold * std`` over the model's test period. The companion
    ``record_anomalies`` is deliberately not called — it writes to disk.
    """
    if dataset not in DATASETS:
        raise ToolError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if std_threshold <= 0:
        raise ToolError("std_threshold must be greater than 0")

    _, anomaly_detection = _energy_forecast_modules()
    scored, meta = _scored_test_set(dataset)
    daily = daily_error_summary(scored)

    anomalies, threshold = anomaly_detection.detect_anomalies(
        daily, std_threshold=std_threshold
    )

    rows = [
        {
            "date": str(date),
            "model_error_mw": round(float(row["model_error"]), 1),
            "baseline_error_mw": round(float(row["baseline_error"]), 1),
        }
        for date, row in anomalies.sort_values("model_error", ascending=False).iterrows()
    ]

    return {
        "dataset": dataset,
        "test_start": meta["test_start"],
        "test_end": meta["test_end"],
        "days_analysed": int(len(daily)),
        "std_threshold": std_threshold,
        "threshold_mw": round(float(threshold), 1),
        "mean_daily_error_mw": round(float(daily["model_error"].mean()), 1),
        "std_daily_error_mw": round(float(daily["model_error"].std()), 1),
        "anomaly_count": len(rows),
        "anomalies": rows[:MAX_ANOMALY_ROWS],
        "truncated": len(rows) > MAX_ANOMALY_ROWS,
    }


# ---------------------------------------------------------------------------
# Tool 2: MLflow run history
# ---------------------------------------------------------------------------

_RUNS_SQL = """
    SELECT r.run_uuid, r.name, r.status, r.start_time, r.end_time, e.name
    FROM runs r
    JOIN experiments e ON e.experiment_id = r.experiment_id
    WHERE r.lifecycle_stage = 'active'
      AND (? IS NULL OR e.name = ?)
    ORDER BY r.start_time DESC
    LIMIT ?
"""


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    """Open mlflow.db such that a write is impossible, not merely avoided."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def query_mlflow_runs(
    experiment: str | None = None,
    limit: int = 10,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the training run history recorded in the project's mlflow.db.

    Most recent run first, each with its logged params and latest metric values
    (``baseline_mae``, ``model_mae``, ``improvement_pct``). Optionally filtered
    to one experiment, e.g. ``solar-forecast-recent``.
    """
    if limit < 1:
        raise ToolError("limit must be at least 1")
    limit = min(limit, MAX_RUNS)

    db = Path(db_path) if db_path is not None else config.MLFLOW_DB
    _require(db, "mlflow.db")

    with closing(_read_only_connection(db)) as conn:
        experiments = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM experiments WHERE lifecycle_stage = 'active' ORDER BY name"
            )
        ]
        if experiment is not None and experiment not in experiments:
            raise ToolError(
                f"no experiment named {experiment!r}; available: {', '.join(experiments)}"
            )

        rows = conn.execute(_RUNS_SQL, (experiment, experiment, limit)).fetchall()

        runs = []
        for run_id, name, status, start_ms, end_ms, exp_name in rows:
            metrics = {
                key: value
                for key, value in conn.execute(
                    "SELECT key, value FROM latest_metrics WHERE run_uuid = ? AND is_nan = 0",
                    (run_id,),
                )
            }
            params = dict(
                conn.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,))
            )
            if not name:
                # Older runs keep the display name only as a tag.
                tag = conn.execute(
                    "SELECT value FROM tags WHERE run_uuid = ? AND key = 'mlflow.runName'",
                    (run_id,),
                ).fetchone()
                name = tag[0] if tag else None

            runs.append(
                {
                    "run_id": run_id,
                    "run_name": name,
                    "experiment": exp_name,
                    "status": status,
                    "start_time": _iso(start_ms),
                    "end_time": _iso(end_ms),
                    "duration_seconds": (
                        round((end_ms - start_ms) / 1000, 1)
                        if start_ms is not None and end_ms is not None
                        else None
                    ),
                    "params": params,
                    "metrics": {k: round(v, 4) for k, v in sorted(metrics.items())},
                }
            )

    return {
        "database": str(db),
        "experiment_filter": experiment,
        "experiments_available": experiments,
        "run_count": len(runs),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Tool 3: historical vs recent model comparison
# ---------------------------------------------------------------------------


def compare_models() -> dict[str, Any]:
    """Compare the historical (2015-2018) and recent (last ~1 year) models.

    Each side is scored against REE's own day-ahead forecast over its own test
    window, reproducing what train_model.py, train_model_recent.py and
    dashboard.py compute.
    """
    historical = _model_report("historical")
    recent = _model_report("recent")

    return {
        "historical": historical,
        "recent": recent,
        "comparison": {
            "improvement_pct_delta": (
                None
                if historical["improvement_pct"] is None or recent["improvement_pct"] is None
                else round(recent["improvement_pct"] - historical["improvement_pct"], 1)
            ),
            "win_rate_delta_pct": (
                None
                if historical["model_win_rate_pct"] is None
                or recent["model_win_rate_pct"] is None
                else round(recent["model_win_rate_pct"] - historical["model_win_rate_pct"], 1)
            ),
        },
        "caveat": (
            "Absolute MAE is not comparable across the two models: they are scored on "
            "different years and Spain's installed solar capacity grew substantially "
            "between them, so the recent model's errors are larger in MW simply because "
            "generation is larger. improvement_pct (versus REE's own forecast on the same "
            "window) and model_win_rate_pct are the comparable figures."
        ),
    }


# ---------------------------------------------------------------------------
# Registry — what the agent loop in the next step will hand to the model
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "detect_anomalies",
        "description": (
            "Find days where the solar forecasting model's error was statistically "
            "unusual (above mean + N standard deviations of daily error over the "
            "model's test period). Use for questions about bad days, error spikes, "
            "or when the model went wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "enum": list(DATASETS),
                    "description": (
                        "Which model to inspect: 'recent' (last ~1 year of ESIOS data, "
                        "the live dashboard model) or 'historical' (2015-2018 data). "
                        "Defaults to 'recent'."
                    ),
                },
                "std_threshold": {
                    "type": "number",
                    "description": (
                        "Standard deviations above the mean daily error before a day "
                        "counts as anomalous. Defaults to 2.0; lower it to surface more days."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "query_mlflow_runs",
        "description": (
            "Read the MLflow training run history (experiments, params and metrics such "
            "as baseline_mae, model_mae and improvement_pct) for the solar forecasting "
            "project. Use for questions about past training runs, what was tried, or how "
            "metrics changed over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "string",
                    "description": (
                        "Restrict to one experiment name, e.g. 'solar-forecast-recent' or "
                        "'solar-forecast-historical'. Omit for all experiments."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum runs to return, most recent first. Defaults to 10.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_models",
        "description": (
            "Compare the historical model (trained on 2015-2018 data) against the recent "
            "model (trained on the last ~1 year of ESIOS data): MAE, improvement over "
            "REE's official day-ahead forecast, and the share of days each model beats it. "
            "Use for questions about which model is better or whether performance has drifted."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_FUNCTIONS = {
    "detect_anomalies": detect_anomalies,
    "query_mlflow_runs": query_mlflow_runs,
    "compare_models": compare_models,
}


def run_tool(name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a tool call by name, turning failures into a readable payload.

    The agent loop hands whatever the model produced straight to this function,
    so a bad tool name or bad argument has to come back as a result the model
    can read and retry from, not as an exception that kills the conversation.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"unknown tool {name!r}; available: {', '.join(TOOL_FUNCTIONS)}"}

    try:
        return func(**(tool_input or {}))
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:  # wrong/unexpected arguments from the model
        return {"error": f"bad arguments for {name}: {exc}"}
