"""Streamlit interface: ask a question, see the answer and how it was reached.

Deliberately not a plain chat window. The point of this project is that answers
are grounded in real function calls against a real forecasting system, so the
interface shows the tool calls, their arguments, and the payloads that came
back — and it screens every answer through src/guardrails.py before displaying
it, surfacing anything flagged *next to* the answer rather than quietly
dropping it. An interface that hid a fabricated figure would undo the point of
the four steps that precede it.

No API call happens on load. The model is only called when someone submits a
question, so opening the page costs nothing.

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src import agent, config, guardrails, tools

EVAL_RESULTS = Path(__file__).resolve().parent / "eval_results.json"

EXAMPLE_QUESTIONS = [
    "What is the solar model's MAE right now?",
    "Does this model beat REE's official day-ahead forecast?",
    "Has the model's performance changed since it was last trained?",
    "Which day was the recent model's worst, and how large was the error?",
    "How many training runs are recorded, and in which experiments?",
]


st.set_page_config(page_title="REE Assistant", page_icon="☀️", layout="wide")


@st.cache_data
def load_eval_summary() -> dict | None:
    """Read the recorded evaluation result. Local file only — no API call."""
    if not EVAL_RESULTS.exists():
        return None
    try:
        return json.loads(EVAL_RESULTS.read_text()).get("summary")
    except (json.JSONDecodeError, OSError):
        return None


def render_findings(findings: list[guardrails.Finding]) -> None:
    """Show what the guardrails caught, beside the answer rather than instead of it."""
    if not findings:
        st.success(
            "Guardrails: no issues found — every figure in this answer appears in a "
            "tool result. That is not a guarantee of correctness.",
            icon="✅",
        )
        return

    for finding in findings:
        body = f"**{finding.title}**  \n{finding.detail}"
        if finding.level == "warning":
            st.error(body, icon="🚨")
        else:
            st.warning(body, icon="🔍")


def render_tool_calls(result: agent.AgentAnswer) -> None:
    """The agentic part, made visible: what was called, with what, and what came back."""
    if not result.tool_calls:
        st.info(
            "No tool was called — the model answered this one directly. For a question "
            "about project data, that would itself be a problem, and the guardrail above "
            "checks for exactly that.",
            icon="💬",
        )
        return

    st.caption(f"{len(result.tool_calls)} tool call(s), in order:")
    for index, call in enumerate(result.tool_calls, start=1):
        status = "⚠️ error" if call.is_error else "✓"
        arguments = json.dumps(call.arguments) if call.arguments else "no arguments"
        with st.expander(
            f"{index}. `{call.name}`({arguments}) — {status} · {call.duration_seconds:.2f}s "
            f"· turn {call.turn}",
            expanded=call.is_error,
        ):
            st.caption("What the tool returned:")
            st.json(call.result, expanded=False)


# ---------------------------------------------------------------------------
# Sidebar — all read from local files, nothing here calls the API
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.markdown(
        "Answers questions about a [Spain solar forecasting project]"
        "(https://github.com/Antonyevan/ree-generation-forecast) by calling read-only "
        "functions from it — not by summarising a document about it."
    )

    st.subheader("Configuration")
    st.markdown(f"**Model:** `{config.MODEL}`")
    available = config.energy_forecast_available()
    st.markdown(
        f"**Sibling project:** {'✅ found' if available else '❌ not found'}  \n"
        f"`{config.ENERGY_FORECAST_DIR}`"
    )
    if not available:
        st.warning(
            "The energy-forecast checkout is missing, so the tools have nothing to read. "
            "Questions will return errors — which the agent should report plainly.",
            icon="⚠️",
        )

    st.subheader("Tools")
    for schema in tools.TOOL_SCHEMAS:
        st.markdown(f"- `{schema['name']}`")

    summary = load_eval_summary()
    if summary:
        st.subheader("Measured performance")
        st.markdown(
            f"On {summary['cases']} questions with verified answers:\n\n"
            f"- Tool choice: **{summary['tool_choice']['passed']}/{summary['tool_choice']['of']}**\n"
            f"- Arguments: **{summary['arguments']['passed']}/{summary['arguments']['of']}**\n"
            f"- Faithfulness: **{summary['faithfulness']['passed']}/{summary['faithfulness']['of']}**\n"
            f"- All three: **{summary['all_three']['passed']}/{summary['all_three']['of']}** "
            f"({summary['all_three']['pct']}%)"
        )
        st.caption("See the README's 'What didn't work' for what these numbers hide.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("☀️ REE Assistant")
st.caption(
    "Ask about the Spain solar forecast model. Every answer is screened before it is "
    "shown, and the tool calls behind it are visible below it."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="What is the solar model's MAE right now?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", type="primary")

with st.expander("Example questions"):
    for example in EXAMPLE_QUESTIONS:
        st.markdown(f"- {example}")
    st.caption(
        "Copy one into the box above. They are not buttons on purpose — nothing here "
        "should call the API without you asking it to."
    )

# The only place the model is ever called.
if submitted and question.strip():
    with st.spinner("Calling the model and running tools…"):
        try:
            result = agent.ask(question.strip(), log_path=agent.DEFAULT_LOG_PATH)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(
                f"**The request failed before an answer came back.**\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "If this mentions authentication, set `ANTHROPIC_API_KEY` in the "
                "environment you launched Streamlit from.",
                icon="🚨",
            )
            result = None

    if result is not None:
        findings = guardrails.screen(result.answer, result.tool_calls)
        st.session_state.history.insert(0, (result, findings))
elif submitted:
    st.warning("Type a question first.", icon="✍️")

for position, (result, findings) in enumerate(st.session_state.history):
    st.divider()
    st.subheader(result.question)

    render_findings(findings)
    st.markdown(result.answer)

    with st.container():
        st.markdown("**How this answer was produced**")
        render_tool_calls(result)
        st.caption(
            f"{result.model} · {result.turns} turn(s) · "
            f"{result.input_tokens:,} input / {result.output_tokens:,} output tokens · "
            f"stop reason `{result.stop_reason}`"
        )

    if position == 0 and len(st.session_state.history) > 1:
        st.caption("Earlier questions from this session follow.")
