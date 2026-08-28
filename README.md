# ree-assistant

An agentic LLM assistant that answers natural-language questions about the
[Spain solar forecasting project](../energy-forecast) by calling real functions
from it — not by summarising a document about it.

It exists to demonstrate two things: **agentic tool-calling** (the model decides
which function to call, with what arguments, and grounds its answer in the real
result) and **rigorous LLM evaluation** (a labelled eval set with known-correct
answers, built before any agent output is trusted).

This is a standalone project in its own repo. It depends on `energy-forecast`
being checked out alongside it, but is meant to be understandable without
reading that repo.

---

## Status

Built in the order below. Each step is finished only when its tests pass.

| # | Step | State |
|---|------|-------|
| 1 | Repo scaffold + confirm API key works with one trivial call | **done** |
| 2 | Tools layer (`src/tools.py`) — 3 read-only wrappers, each tested in isolation | not started |
| 3 | Agent loop (`src/agent.py`) — model picks a tool, runs it, answers from the result; every tool call logged | not started |
| 4 | Evaluation harness — 15–20 questions with known-correct answers; score tool choice, arguments, and answer faithfulness; record the honest pass rate | not started |
| 5 | Guardrail tests (`tests/test_guardrails.py`) — catch fabricated tool results and schema violations; wire into CI | not started |
| 6 | Interface (Streamlit or CLI) + this README filled in with the real methodology and the actual pass rate | not started |

The eval harness (step 4) comes **before** the agent's output is trusted or
shown off. That ordering is deliberate: the same discipline as writing tests
before trusting a bug fix.

---

## The three tools (planned)

Each is a thin, read-only wrapper around code in `energy-forecast` that already
works and is already tested there. None of them write to or modify that project.

1. **`detect_anomalies()`** — wraps `energy-forecast/anomaly_detection.py`. Flags
   days where the forecasting model's error is statistically unusual
   (mean + 2 standard deviations).
2. **MLflow run history** — queries the existing MLflow store
   (`energy-forecast/mlflow.db`) for training-run parameters and metrics
   (`baseline_mae`, `model_mae`, `improvement_pct`) across historical retrains.
3. **Historical-vs-recent model comparison** — compares the 2015–2018 historical
   model's performance against the recent-data model's, using the existing
   logic and results in that project.

Scope is fixed at these three until they work and are evaluated. No more tools,
no RAG / vector DB, no public deployment — this runs locally via
`streamlit run app.py` (or a CLI) and nothing more.

---

## Setup

```bash
pip install -r requirements.txt
pip install pytest            # test-only dependency, kept out of requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."   # never hardcoded; a spend limit is set in the Anthropic console

# Confirm the key works (one trivial call, ~cents):
python scripts/smoke_test_api.py

# Run the test suite:
pytest tests/ -v
```

`ENERGY_FORECAST_DIR` defaults to `../energy-forecast`. Override it if the
sibling project lives elsewhere. `REE_ASSISTANT_MODEL` defaults to
`claude-haiku-4-5`.

Tests that need the sibling project's data or MLflow database skip themselves
when it isn't present (e.g. in CI), so `pytest` stays green everywhere. Tests
never make a live API call.

---

## Repository structure

| Path | Purpose |
|---|---|
| `src/config.py` | Model id, path to the sibling `energy-forecast` project, availability check |
| `scripts/smoke_test_api.py` | Step 1: trivial live API call to confirm the key works |
| `tests/test_setup.py` | Step 1: scaffold imports and config resolve |
| `.github/workflows/run_tests.yml` | Runs `pytest` on every push and PR to `main` |

More rows land here as steps 2–6 are built.

---

## What didn't work

_(To be filled in honestly as the project is built — tool call failure patterns,
eval pass rate if mediocre, anything that had to be worked around.)_
