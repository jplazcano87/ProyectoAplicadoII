"""Approximate API pricing and token estimation for cost tracking.

Prices are USD per 1,000,000 tokens (OpenAI public pricing). They are only
used to report an *approximate* cost per query in the eval harness; update the
table if pricing changes. Unknown models fall back to zero cost with a warning.
"""

from __future__ import annotations

import warnings

import tiktoken

# USD per 1M tokens: {model: {"input": ..., "output": ...}}.
PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

_warned: set[str] = set()
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _price(model: str) -> dict[str, float]:
    if model not in PRICES:
        if model not in _warned:
            warnings.warn(f"No pricing for model '{model}'; cost reported as 0.", stacklevel=2)
            _warned.add(model)
        return {"input": 0.0, "output": 0.0}
    return PRICES[model]


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    """Cost in USD for a call given prompt/completion token counts."""
    p = _price(model)
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Approximate token count for a string (cl100k_base, close enough for all
    OpenAI models used here)."""
    return len(_ENCODER.encode(text or ""))


def embedding_cost_usd(model: str, text: str) -> float:
    """Approximate cost of embedding a single text (no output tokens)."""
    return cost_usd(model, estimate_tokens(text))
