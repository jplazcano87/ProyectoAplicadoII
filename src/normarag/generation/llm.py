"""LLMClient interface with OpenAI and Anthropic implementations.

Generators sit behind the LLMClient interface so the model (OpenAI generator,
Anthropic judge, or a local model) can be swapped via env. `complete` returns an
LLMResult carrying the text plus token usage and an approximate cost, so the eval
harness can report cost per query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential

from normarag.generation.pricing import cost_usd


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> LLMResult:
        """Return the model's completion for a prompt (+ optional system)."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or settings.generator_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def complete(self, prompt: str, system: str | None = None) -> LLMResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd(self.model, prompt_tokens, completion_tokens),
        )


class AnthropicClient(LLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.model = model or settings.judge_model
        self._client = Anthropic(api_key=api_key or settings.anthropic_api_key or None)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def complete(self, prompt: str, system: str | None = None) -> LLMResult:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [
            getattr(block, "text", "")
            for block in resp.content
            if getattr(block, "type", None) == "text"
        ]
        prompt_tokens = getattr(resp.usage, "input_tokens", 0) or 0
        completion_tokens = getattr(resp.usage, "output_tokens", 0) or 0
        return LLMResult(
            text="".join(parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd(self.model, prompt_tokens, completion_tokens),
        )
