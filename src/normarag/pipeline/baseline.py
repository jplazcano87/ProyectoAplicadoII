"""Pipeline A: LLM-only baseline (no retrieval).

The reference configuration every other config is compared against. It answers
purely from the model's parametric knowledge — no corpus access — but emits the
same uniform system-output contract (empty `contexts`).
"""

from __future__ import annotations

from normarag.generation.llm import LLMClient, OpenAIClient
from normarag.generation.prompt import (
    BASELINE_SYSTEM_PROMPT,
    build_baseline_prompt,
    parse_structured_output,
)
from normarag.pipeline.contract import Timer, system_output


def run_baseline(question: str, llm: LLMClient | None = None) -> dict:
    llm = llm or OpenAIClient()

    with Timer() as timer:
        prompt = build_baseline_prompt(question)
        result = llm.complete(prompt, system=BASELINE_SYSTEM_PROMPT)
        parsed = parse_structured_output(result.text)

    return system_output(
        answer=parsed["answer"],
        citations=parsed["citations"],
        abstained=parsed["abstained"],
        contexts=[],
        latency_ms=timer.latency_ms,
        cost_usd=result.cost_usd,
    )
