"""LLMClient interface with OpenAI and Anthropic implementations.

Generators sit behind the LLMClient interface so the model (OpenAI generator,
Anthropic judge, or a local model) can be swapped via env.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text completion for a prompt (+ optional system)."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or settings.generator_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.model = model or settings.judge_model
        self._client = Anthropic(api_key=api_key or settings.anthropic_api_key or None)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def complete(self, prompt: str, system: str | None = None) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return "".join(parts)
