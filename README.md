# 🤖 ree-assistant

An agentic LLM assistant that answers natural-language questions about the [Spain Solar Generation Forecast](https://github.com/Antonyevan/ree-generation-forecast) project by calling its real functions — not by summarizing a document about it.

---

## What this is

Most "AI assistant" demos are a chat window wrapped around a language model with no way to check whether its answers are actually correct. This project exists specifically to avoid that: it demonstrates **agentic tool-calling** (the model decides which function to call, with what arguments, and grounds its answer in the real result) and **rigorous evaluation** (a labeled test set with known-correct answers, built and checked *before* any agent output is trusted or shown off).

It's a standalone project in its own repository, built on top of the solar forecasting project rather than inside it — reusing real, already-tested code as its toolset, not duplicating logic.

---

## Build status

Built in a fixed order. Each step only counts as done once its tests pass.

| # | Step | Status |
|---|---|---|
| 1 | Repo scaffold, API key confirmed working | ✅ Done |
| 2 | Tools layer — four read-only wrappers, each tested in isolation | ✅ Done |
| 3 | Agent loop — model selects a tool, runs it, answers from the real result | ⏳ In progress |
| 4 | Evaluation harness — known-correct answers, scored before anything is trusted | Not started |
| 5 | Guardrail tests — catch fabricated results and schema violations, enforced by CI | Not started |
| 6 | Interface (Streamlit/CLI) and final documentation | Not started |

The evaluation harness (step 4) is built **before** any agent output is trusted — the same discipline as writing a regression test before trusting a bug fix.

---

## The four tools

Each tool is a thin, read-only wrapper around code in the solar forecasting project that already works and is already tested there. None of them modify that project's files, models, or data.

| Tool | Wraps | Answers questions like |
|---|---|---|
| `detect_anomalies` | `anomaly_detection.py` | "Which days did the model go badly wrong?" |
| `query_mlflow_runs` | `mlflow.db` | "What did the last training runs score?" |
| `compare_models` | logic from `train_model.py` / `dashboard.py` | "Is the recent model better than the historical one?" |
| `get_live_status` | `latest_metrics.json` | "What is the dashboard showing right now?" |

`get_live_status` is different from the other three: rather than recomputing from local test data, it reads the numbers the solar project's own scheduled workflow already computed and committed — faster, and the authoritative answer for "right now."

---

## Project Journey

1. **Scaffolded the project.** Config, tests, CI, and a smoke test confirming the Anthropic API key works — verified before any tool logic was written.
2. **Built the tools layer.** Three functions wrapping real, existing code from the solar project. Verified `compare_models` reproduces an actual logged MLflow run exactly (166.9 MW model / 123.6 MW baseline), not just plausible-looking numbers.
3. **Enforced read-only access deliberately.** The obvious implementation — importing `train_model.py` directly — would have triggered real training and written new rows into the solar project's MLflow database, just from being imported. Avoided by reimplementing the comparison logic separately, and by opening `mlflow.db` in SQLite read-only mode so a write is structurally impossible, not just avoided by convention.
4. **Found a real staleness bug while building.** The tools read local files from the sibling project — files that had already gone six days stale earlier in the same session. `get_live_status` hit this directly: on first run, it correctly reported the live dashboard's data as 37 hours old, because the local checkout hadn't been synced.
5. **Fixed it with a sync step, not a workaround.** `src/sync.py` runs `git pull` in the sibling project before any tool executes, with sensible limits: failures are never fatal (the assistant still answers using whatever data exists), pulls are rate-limited to once per 5 minutes to avoid slowing down repeated offline calls, and a successful pull clears cached computations so stale results can't linger in memory after fresh files arrive.
6. **Documented the one intentional exception to "read-only."** The sync step's `git pull` is the single write this project makes to the sibling repository — stated plainly here and in the code, not hidden, since it's a genuine, deliberate departure from the project's own stated principle.

---

## Setup

```bash
pip install -r requirements.txt
pip install pytest  # test-only, kept out of requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."  # never hardcoded; a spend limit is set in the Anthropic console

python scripts/smoke_test_api.py  # confirms the key works, ~cents
pytest tests/ -v
```

`ENERGY_FORECAST_DIR` defaults to `../energy-forecast`. `REE_ASSISTANT_MODEL` defaults to `claude-haiku-4-5`. `REE_ASSISTANT_SYNC=0` disables the pre-read sync, for offline work.

Tests requiring the sibling project's data skip automatically when it isn't present (e.g. in CI), so the suite stays green everywhere. No test makes a live API call.

---

## Repository structure

| Path | Purpose |
|---|---|
| `src/config.py` | Model id, sibling project path, availability check |
| `src/tools.py` | The four tools, their schemas, and the dispatcher |
| `src/sync.py` | Pulls the sibling checkout before a tool reads it; never fatal |
| `scripts/smoke_test_api.py` | One trivial live API call, confirming the key works |
| `tests/test_setup.py` | Scaffold and config resolution |
| `tests/test_tools.py` | Each tool verified in isolation — no LLM, no network |
| `tests/test_sync.py` | Sync behavior, including graceful failure |
| `.github/workflows/run_tests.yml` | Runs pytest on every push and PR |

---

## Key Technical Decisions

- **Read-only by construction, not by convention** — the SQLite connection to `mlflow.db` is opened in a mode that makes a write structurally impossible, not merely discouraged.
- **No speculative scope** — exactly four tools, no vector database, no public deployment. Runs locally. Additional scope is added only once the current scope is built, tested, and evaluated.
- **The one write is documented, not hidden** — the sync step's `git pull` is a deliberate, stated exception to the project's read-only principle, not an inconsistency to gloss over.
- **The evaluation harness is built before the agent's output is ever trusted** — mirroring the exact discipline that governs the solar project's own regression tests.

---

## What didn't work

*This section will be completed honestly once the project is further along — any tool call failure patterns, the actual evaluation pass rate if it turns out mediocre, and anything that had to be reworked. Left genuinely blank for now rather than filled with a placeholder claim.*