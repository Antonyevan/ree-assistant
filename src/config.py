"""Configuration for ree-assistant.

This project is a standalone agentic LLM assistant that answers questions by
calling read-only functions from a *separate* project: the Spain solar
forecasting system at ~/projects/energy-forecast.

Nothing here writes to or modifies that project. We only need to know where it
lives on disk so the tools layer can import its modules and read its files.
"""

from __future__ import annotations

import os
from pathlib import Path

# Model used for development. Haiku is cheap, fast, and sufficient to prove the
# agentic tool-calling pattern. Override with REE_ASSISTANT_MODEL if needed.
MODEL = os.environ.get("REE_ASSISTANT_MODEL", "claude-haiku-4-5")

# Location of the sibling energy-forecast project. Default assumes the two repos
# are checked out side by side (~/projects/energy-forecast and
# ~/projects/ree-assistant). Override with ENERGY_FORECAST_DIR.
ENERGY_FORECAST_DIR = Path(
    os.environ.get(
        "ENERGY_FORECAST_DIR",
        Path(__file__).resolve().parent.parent.parent / "energy-forecast",
    )
).resolve()

# Specific files inside that project the tools depend on.
MLFLOW_DB = ENERGY_FORECAST_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
RECENT_MODEL_PKL = ENERGY_FORECAST_DIR / "recent_model.pkl"


def energy_forecast_available() -> bool:
    """True if the sibling project is present on disk with the files we need.

    Used by tests to skip cleanly in CI, where energy-forecast is not checked
    out alongside this repo.
    """
    return ENERGY_FORECAST_DIR.is_dir() and MLFLOW_DB.exists()
