"""Uniform system-output contract shared by every configuration (A/B/C/D).

Every pipeline returns the same dict so they are directly comparable in the eval
harness:

    {
      "answer": str,
      "citations": [{"chapter": str, "numeral": str}],
      "abstained": bool,
      "contexts": [chunk dict, ...],   # retrieved chunks ([] for the baseline)
      "latency_ms": float,
      "cost_usd": float,
    }
"""

from __future__ import annotations

from time import perf_counter


class Timer:
    """Context manager measuring wall-clock latency in milliseconds."""

    def __enter__(self) -> Timer:
        self._start = perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.latency_ms = (perf_counter() - self._start) * 1000.0


def system_output(
    answer: str,
    citations: list[dict],
    abstained: bool,
    contexts: list[dict],
    latency_ms: float,
    cost_usd: float,
) -> dict:
    return {
        "answer": answer,
        "citations": citations,
        "abstained": abstained,
        "contexts": contexts,
        "latency_ms": round(latency_ms, 2),
        "cost_usd": round(cost_usd, 8),
    }
