"""Reranker interface and Cohere implementation.

Rerankers sit behind the Reranker interface so a local cross-encoder can replace
Cohere later without touching pipeline logic. The reranker reorders the fused
candidates against the query and keeps the most relevant TOP_K_RERANK.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential

# Process-wide throttle so concurrent eval workers never exceed Cohere's trial
# rate limit (~10 rerank calls/min). Shared across threads via a lock.
_rerank_lock = threading.Lock()
_last_rerank_at = 0.0


def _throttle(min_interval: float) -> None:
    global _last_rerank_at
    with _rerank_lock:
        wait = _last_rerank_at + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_rerank_at = time.monotonic()


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        """Reorder candidates by relevance to query; return the top_k, each with
        an added `rerank_score`."""
        raise NotImplementedError


class CohereReranker(Reranker):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        text_field: str = "text",
    ) -> None:
        import cohere

        # Cohere's SDK auto-detects CO_API_KEY; accept either env var.
        key = api_key or settings.co_api_key or settings.cohere_api_key or None
        self.model = model or settings.rerank_model
        self.text_field = text_field
        self._client = cohere.Client(api_key=key)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(min=5, max=120))
    def _rerank_call(self, query: str, documents: list[str], top_n: int):
        _throttle(settings.cohere_min_interval_s)
        return self._client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        if not candidates:
            return []
        top_k = top_k or settings.top_k_rerank
        documents = [c.get(self.text_field, "") for c in candidates]
        response = self._rerank_call(query, documents, min(top_k, len(candidates)))

        reranked: list[dict] = []
        for result in response.results:
            chunk = dict(candidates[result.index])
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)
        return reranked
