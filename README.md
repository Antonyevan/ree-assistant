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
| 3 | Agent loop — model selects a tool, runs it, answers from the real result | ✅ Done |
| 4 | Evaluation harness — known-correct answers, scored before anything is trusted | ✅ Done |
| 5 | Guardrail tests — catch fabricated results and schema violations, enforced by CI | ✅ Done |
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

## The agent loop

`src/agent.py` takes a question, hands it to Claude along with `TOOL_SCHEMAS`,
runs whatever tool the model asks for through `run_tool()`, and feeds the real
result back so the final answer is grounded in this project's data.

```bash
python -m src.agent "What is the model's MAE right now?"
```

`ask()` returns an `AgentAnswer`: the answer text plus the full record of how it
was reached — every tool call's name, arguments, result, whether it errored,
how long it took, and which turn it happened on, along with token usage. That
record is the point. Step 4 scores tool choice and arguments, not just the
prose, and it cannot do that from a string.

Four things the loop gets right, each covered by a test:

* **No tool needed** — a question that doesn't require data is answered in one
  round trip, with an empty `tool_calls` list.
* **Several tools in one turn** — a question spanning live status and the
  backtest gets both, and all results go back in a *single* user message.
  Splitting them teaches the model to stop asking for tools in parallel.
* **Tool errors reach the model** — a `{"error": ...}` payload from `run_tool`
  goes back as a `tool_result` with `is_error: true`, so Claude can explain
  what failed instead of the loop crashing or inventing an answer.
* **A turn ceiling** — a model that keeps asking for tools stops at `MAX_TURNS`
  and says so, rather than running up the bill.

The model is `config.MODEL` at every call site; `agent.py` contains no model
string, and a test asserts that by grepping its own source. Switching models is
a config change or `REE_ASSISTANT_MODEL=...`, never an edit here. For the same
reason the request carries no `thinking` or `effort` parameters — those are
model-gated, and the default model does not accept them.

### Free tests and live tests

The unit tests drive the loop with a fake Anthropic client: no network, no cost,
and they cover the cases that are awkward to provoke on demand against a real
model. Two `@live` tests make real API calls and are skipped twice over — by the
`-m "not live"` default in `pytest.ini` and by an env var — so the default suite
and CI stay free:

```bash
pytest tests/                                                  # free, the default
REE_ASSISTANT_LIVE_TESTS=1 pytest tests/ -m live -v            # real calls, real cents
```

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
| `src/agent.py` | The tool-calling loop, the run record, and a small CLI |
| `src/guardrails.py` | Number grounding, failure acknowledgement, redundancy counting |
| `scripts/smoke_test_api.py` | One trivial live API call, confirming the key works |
| `tests/test_setup.py` | Scaffold and config resolution |
| `tests/test_tools.py` | Each tool verified in isolation — no LLM, no network |
| `tests/test_sync.py` | Sync behavior, including graceful failure |
| `tests/test_agent.py` | Loop mechanics against a fake client, plus two opt-in live tests |
| `tests/eval_cases.py` | The 21 evaluation questions and their verified ground truth |
| `tests/test_agent_eval.py` | Scores tool choice, arguments and faithfulness on each question |
| `tests/test_guardrails.py` | Fabrication, failure-reporting and redundancy checks, run in CI |
| `pytest.ini` | Registers the `live` marker and deselects it by default |
| `.github/workflows/run_tests.yml` | Runs pytest on every push and PR |

---

## Key Technical Decisions

- **Read-only by construction, not by convention** — the SQLite connection to `mlflow.db` is opened in a mode that makes a write structurally impossible, not merely discouraged.
- **No speculative scope** — exactly four tools, no vector database, no public deployment. Runs locally. Additional scope is added only once the current scope is built, tested, and evaluated.
- **The one write is documented, not hidden** — the sync step's `git pull` is a deliberate, stated exception to the project's read-only principle, not an inconsistency to gloss over.
- **The evaluation harness is built before the agent's output is ever trusted** — mirroring the exact discipline that governs the solar project's own regression tests.

---

## Guardrails

The evaluation scores faithfulness with checks written per question, so it
catches contradictions it was told to expect. Step 5 adds three checks that need
no such foresight, and two of them run for free in CI against the recorded
evaluation run — 21 real answers with the tool payloads behind them. Rerun the
eval, commit the results, and the guardrails re-measure against fresh behaviour.

**No figure without a tool call.** `ungrounded_numbers()` inverts the eval's
question: instead of "did the answer say what we expected", it asks "did the
answer state anything the tools never returned". Prose integers and dates are
ignored; a number is treated as a claim if it carries a decimal, a unit, or a
magnitude over 100. On the recorded run it flags one case, and that one is
legitimate arithmetic (see below). Where no tool ran at all, there is nothing to
derive from, so the check has no possible false positive.

**No papering over a failure.** An error payload carries no data, so a figure
stated alongside one is invented — the same check catches it, and
`acknowledges_failure()` verifies the answer names the problem. The subtler case
is a tool that succeeds but omits the field asked about; a fabricated value
there is caught the same way.

**No more tools than the question needs.** The two redundant-call cases the eval
found are pinned by name. A third fails the build; fixing one of the two also
fails, prompting the baseline to be tightened rather than left slack.

### What the detector deliberately does not do

Admitting derived arithmetic was measured, not assumed. On a real payload of 66
numbers, allowing differences between returned figures grew the grounded set to
1,962 values and halved sensitivity; allowing ratios grew it to 5,837, at which
point fabricated figures like 950 and 1500 both passed. So the detector stays
strict and the one known derivation — `ambiguous-drift` computing "about 9.5%"
from two figures it correctly reported — is named in the test instead. A figure
the agent computed rather than read gets flagged for a human, which is the
cheaper error.

These are not proof of correctness. A wrong claim made without numbers, or a
figure that coincidentally matches an unrelated field, still gets through.

---

## What didn't work

### The evaluation result

21 questions, scored on three dimensions. One live run (Claude Haiku 4.5,
2026-09-04), 72,355 input and 4,354 output tokens:

| Dimension | Result |
|---|---|
| Tool choice | 19/21 (90.5%) |
| Arguments | 4/4 (100%) — only 4 questions imply arguments |
| Faithfulness | 21/21 (100%) |
| **All three** | **19/21 (90.5%)** |

The first scoring of that run said 17/21 (81.0%). Two of the four failures
turned out to be bugs in the grader, not the agent. Full detail in
`eval_results.json`, which keeps both summaries.

### Two grader bugs, found by reading the failures

**The number matcher rejected answers that were too precise.** It built string
candidates (`123.612`, `123.6`, `124`) and required one to appear with no digit
following. Asked for a logged baseline MAE of 123.6121, the agent answered
"123.61 (specifically 123.6121)" — correct to more places than expected — and
every candidate failed its lookahead, because each was followed by another
digit. Scored as omitting the number it had actually quoted exactly. It now
compares numerically: a stated figure counts if the target rounds to it at the
precision the answer chose, so 1205 matches 1204.9 and 900 does not. ISO dates
and clock times are stripped first, so a timestamp cannot donate a stray digit.

**A blacklist could not tell reporting from claiming.** The unflattering case
forbade the substring "beats REE". The agent opened with "**No, this model does
not beat REE's official day-ahead forecast**" and later wrote "Win rate: 29.5%
(the model beats REE on less than 1 in 3 days)". The second line is true,
unflattering, and contains the forbidden phrase. The check now works sentence by
sentence and ignores any sentence carrying a qualifier ("not", "only", "less
than", "worse"…), so it still catches "The model beats REE comfortably" while
passing an honest report. Note this was not simple negation-blindness — the
sentence that tripped it was a qualified true statement, not a negated one.

Both fixes have regression tests built from the exact answer text that exposed
them. Rescoring the recorded run changed exactly two scores, and no passing case
flipped to failing — the fixes are corrections, not a loosening of the bar. The
tool payloads were verified byte-identical before rescoring (same
`latest_metrics.json` `computed_at`), and no answer was regenerated.

The lesson worth keeping: an eval that fails a correct answer is not a stricter
eval, it is a broken one, and the failure mode is invisible unless you read the
answers rather than the score. Both bugs would have quietly understated the
agent and, worse, would have rewarded a *less* precise answer.

### Two genuine agent behaviours, left as failures

The remaining 2/21 are real, and both are the same minor tendency — calling one
redundant tool alongside the correct one:

* **`mlflow-when`** ("When was the model last trained?") called
  `get_live_status` before `query_mlflow_runs`. It reached the right answer
  from the right source; the extra call was wasted.
* **`compare-windows`** ("What date ranges do the two models' test periods
  cover?") called `query_mlflow_runs` alongside `compare_models`.

Both answered correctly — faithfulness passed on each. The eval scores them as
tool-choice failures anyway, because "called the right tool plus one it did not
need" is a real inefficiency worth measuring, and hiding it behind a tolerance
would defeat the purpose. It costs latency and tokens rather than correctness,
and it is the kind of thing a tolerance in the grader would have made invisible.

### Not yet verified

The two `@live` tests in `tests/test_agent.py` have never been run — the
environment they were written in had no API key. The agent loop is nonetheless
exercised against the real API 21 times by the evaluation above, so the loop
itself is proven; those two specific tests are not.