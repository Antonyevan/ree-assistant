"""Step 1 sanity checks: the scaffold imports and config resolves.

No LLM call here (that would need a key and cost money on every CI run). The
live API check is scripts/smoke_test_api.py, run manually.
"""

import importlib


def test_anthropic_importable():
    anthropic = importlib.import_module("anthropic")
    assert hasattr(anthropic, "Anthropic")


def test_config_loads():
    from src import config

    assert config.MODEL  # a non-empty model id
    # Path objects resolve without touching the filesystem.
    assert str(config.ENERGY_FORECAST_DIR).endswith("energy-forecast")
    assert config.MLFLOW_TRACKING_URI.startswith("sqlite:///")


def test_energy_forecast_available_returns_bool():
    from src import config

    assert isinstance(config.energy_forecast_available(), bool)
