"""Trivial end-to-end check that the Anthropic API key works.

Build order step 1: confirm the key works with one trivial call before writing
any tool or agent logic. No tools, no energy-forecast dependency — just proves
we can reach the API and get a response back.

Usage:
    python scripts/smoke_test_api.py

Requires ANTHROPIC_API_KEY in the environment (never hardcoded).
"""

import os
import sys

import anthropic

from src.config import MODEL


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set in the environment.", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly the word: pong",
            }
        ],
    )

    text = "".join(block.text for block in response.content if block.type == "text").strip()

    print(f"model:        {response.model}")
    print(f"stop_reason:  {response.stop_reason}")
    print(f"input_tokens: {response.usage.input_tokens}")
    print(f"output_tokens:{response.usage.output_tokens}")
    print(f"reply:        {text!r}")

    if "pong" not in text.lower():
        print("Unexpected reply — API reachable but response not as expected.", file=sys.stderr)
        return 1

    print("\nOK: Anthropic API key works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
