"""The agent loop: a question in, an answer grounded in real tool results out.

Step 3 of the build. Claude is given the question and the schemas from
src/tools.py, decides which tool (if any) answers it, and we run that tool for
real and hand the result back so the final answer is grounded in this project's
actual data rather than in the model's recollection of it.

Written as a manual loop rather than the SDK's beta ``tool_runner``. The runner
builds schemas from decorated Python functions; we already have hand-written
schemas whose descriptions steer tool choice, and a run_tool() dispatcher that
handles the sibling-repo sync, cache invalidation and error formatting. The
manual loop also gives us the structured record of every call — name, arguments,
result — that Step 4's evaluation needs in order to score tool choice and
arguments, not just the final prose.

Nothing here names a model. The model comes from config.MODEL, so switching it
(or setting REE_ASSISTANT_MODEL) never means editing this file.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from src import config, tools

log = logging.getLogger(__name__)

# Answers are a few hundred tokens of grounded summary; this is headroom, not a
# target. Kept well inside every model's output cap so switching config.MODEL
# cannot make the request invalid.
MAX_TOKENS = 8192

# A question needing two tools takes three turns (call, call, answer). Five
# leaves room without letting a confused model loop indefinitely on our bill.
MAX_TURNS = 5

# .gitignore already excludes logs/ — run traces are local, not committed.
DEFAULT_LOG_PATH = Path("logs/agent_calls.jsonl")

SYSTEM_PROMPT = """You answer questions about a Spain solar generation forecasting project by calling tools that read its real data.

Ground every factual claim in a tool result. Never state a metric, date, or count from memory or inference — if you have not seen it in a tool result this conversation, call a tool or say you do not have it.

Use get_live_status for anything about the current or latest state. The other tools recompute from local test data over their own windows, which may differ from what is live.

Report what the data says, including when it is unflattering: this model currently performs worse than the operator's own forecast on several measures, and that is the correct answer when asked.

If a tool returns an error, say plainly what failed and what it means for the question. Do not retry the same call with the same arguments, and do not paper over it with a guess.

When a result carries freshness fields, say how current the figures are. Be concise and specific, and prefer exact numbers over adjectives."""


@dataclass
class ToolCall:
    """One tool invocation, recorded for observability and for Step 4's eval."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    is_error: bool
    duration_seconds: float
    turn: int


@dataclass
class AgentAnswer:
    """Everything one question produced: the answer, and how it was reached."""

    question: str
    answer: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tool_names(self) -> list[str]:
        """Tools called, in order — the shape Step 4 scores tool choice on."""
        return [call.name for call in self.tool_calls]

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "tool_names": self.tool_names,
            "used_tools": self.used_tools,
        }


def _text_from(content: list[Any]) -> str:
    """Join the text blocks of a response, ignoring tool_use blocks."""
    return "\n".join(block.text for block in content if block.type == "text").strip()


def _record(answer: AgentAnswer, call: ToolCall, log_path: Path | None) -> None:
    """Log a tool call: to the logger always, to a JSONL file when asked."""
    answer.tool_calls.append(call)

    if call.is_error:
        log.warning(
            "tool %s(%s) returned an error in %.2fs: %s",
            call.name,
            json.dumps(call.arguments),
            call.duration_seconds,
            call.result.get("error"),
        )
    else:
        log.info(
            "tool %s(%s) ok in %.2fs",
            call.name,
            json.dumps(call.arguments),
            call.duration_seconds,
        )

    if log_path is None:
        return

    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "question": answer.question,
        "model": answer.model,
        **asdict(call),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def ask(
    question: str,
    *,
    client: anthropic.Anthropic | None = None,
    max_turns: int = MAX_TURNS,
    log_path: Path | None = None,
    system: str = SYSTEM_PROMPT,
) -> AgentAnswer:
    """Answer one question, calling tools as the model requests them.

    Returns an AgentAnswer carrying the final text and the full record of what
    was called to produce it. A question needing no tool is answered in one
    round trip with an empty tool_calls list.

    A tool that fails is reported back to the model as an error tool_result
    rather than raised: the model can then explain the failure or take another
    route, which is what a user actually needs. API-level failures are left to
    propagate — a caller cannot sensibly treat an auth or rate-limit error as
    an answer.
    """
    client = client or anthropic.Anthropic()
    answer = AgentAnswer(question=question, answer="", model=config.MODEL)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=config.MODEL,  # never hardcoded — see module docstring
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools.TOOL_SCHEMAS,
            messages=messages,
        )

        answer.turns = turn
        answer.stop_reason = response.stop_reason
        answer.input_tokens += response.usage.input_tokens
        answer.output_tokens += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            answer.answer = _text_from(response.content)
            return answer

        messages.append({"role": "assistant", "content": response.content})

        # All tool_use blocks from one turn are answered in a single user
        # message. Splitting them across messages teaches the model to stop
        # asking for tools in parallel.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            arguments = dict(block.input)
            started = time.monotonic()
            result = tools.run_tool(block.name, arguments)
            elapsed = time.monotonic() - started

            is_error = "error" in result
            _record(
                answer,
                ToolCall(
                    name=block.name,
                    arguments=arguments,
                    result=result,
                    is_error=is_error,
                    duration_seconds=round(elapsed, 3),
                    turn=turn,
                ),
                log_path,
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # Out of turns with the model still asking for tools. Return what we have
    # and say so, rather than silently presenting a partial answer as final.
    answer.stop_reason = "max_turns"
    answer.answer = (
        f"Stopped after {max_turns} turns with the model still requesting tools. "
        f"Tools called: {', '.join(answer.tool_names) or 'none'}."
    )
    log.warning("hit max_turns=%s for question: %s", max_turns, question)
    return answer


def main() -> int:
    """CLI: `python -m src.agent "your question"`."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) < 2:
        print('Usage: python -m src.agent "your question"', file=sys.stderr)
        return 1

    result = ask(" ".join(sys.argv[1:]), log_path=DEFAULT_LOG_PATH)

    print(result.answer)
    print(
        f"\n[{result.model} | tools: {', '.join(result.tool_names) or 'none'} | "
        f"{result.turns} turn(s) | {result.input_tokens} in / {result.output_tokens} out]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
