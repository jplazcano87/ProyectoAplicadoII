"""EmbeddingProvider interface and OpenAI implementation.

Providers sit behind the EmbeddingProvider interface so the embedding backend
can be swapped via env without touching indexing/retrieval logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential

# OpenAI's embeddings endpoint accepts many inputs per call; batch to stay well
# within request-size limits while minimizing round-trips.
_BATCH_SIZE = 100


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per input (same order)."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)

    @property
    def dim(self) -> int:
        return self._dim

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=batch)
        return [item.embedding for item in resp.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
