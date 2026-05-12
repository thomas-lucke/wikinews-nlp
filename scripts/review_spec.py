"""Send the project spec to Claude for a hostile review with extended thinking."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()


REVIEWER_PROMPT = (
    "You are a senior engineer doing a hostile review of a technical spec before "
    "any code is written. Find every ambiguity, every assumption stated as fact, "
    "every unjustified decision, every edge case not handled, and every place where "
    "a coding agent given only this spec would have to guess. Do not summarise what "
    "the spec does. Report problems only, numbered."
)


def review_spec(spec_path: str) -> str:
    """Read the spec, send it to Claude, and return only the text blocks of the response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )

    spec_text = Path(spec_path).read_text(encoding="utf-8")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        system=REVIEWER_PROMPT,
        messages=[{"role": "user", "content": spec_text}],
    )

    return "\n".join(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    spec_path = sys.argv[1] if len(sys.argv) > 1 else "docs/SPEC_v3.md"
    print(review_spec(spec_path))
