"""Reranker interface and Cohere implementation."""

from abc import ABC, abstractmethod


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        raise NotImplementedError


class CohereReranker(Reranker):
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        raise NotImplementedError("M4")
