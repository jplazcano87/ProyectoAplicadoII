"""LLMClient interface with OpenAI and Anthropic implementations."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def complete(self, prompt: str) -> str:
        raise NotImplementedError("M2")


class AnthropicClient(LLMClient):
    def complete(self, prompt: str) -> str:
        raise NotImplementedError("M2")
