"""EmbeddingProvider interface and OpenAI implementation.

Providers sit behind the EmbeddingProvider interface so the embedding backend
can be swapped via env without touching indexing/retrieval logic.

Dense embeddings are cached to disk (keyed by model + text hash) so re-indexing
— e.g. to add the sparse/BM25 vector for hybrid search — does not re-pay for
dense embeddings that were already computed.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

from config.settings import settings
from tenacity import retry, stop_after_attempt, wait_exponential

# OpenAI's embeddings endpoint accepts many inputs per call; batch to stay well
# within request-size limits while minimizing round-trips.
_BATCH_SIZE = 100
_CACHE_DIR = Path(".cache/embeddings")


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
        cache_dir: Path | None = _CACHE_DIR,
    ) -> None:
        from openai import OpenAI

        self.model = model or settings.embedding_model
        self._dim = dim or settings.embedding_dim
        self._client = OpenAI(api_key=api_key or settings.openai_api_key or None)
        self._cache_dir = cache_dir

    @property
    def dim(self) -> int:
        return self._dim

    def _cache_path(self, text: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
        return self._cache_dir / self.model / f"{digest}.json"

    def _read_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path and path.exists():
            return json.loads(path.read_text())  # type: ignore[no-any-return]
        return None

    def _write_cache(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vector))

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=batch)
        return [item.embedding for item in resp.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, serving hits from the disk cache and only calling the
        API for the misses (which are then cached)."""
        vectors: list[list[float] | None] = [self._read_cache(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]

        for start in range(0, len(missing), _BATCH_SIZE):
            idx_batch = missing[start : start + _BATCH_SIZE]
            fresh = self._embed_batch([texts[i] for i in idx_batch])
            for i, vec in zip(idx_batch, fresh, strict=True):
                vectors[i] = vec
                self._write_cache(texts[i], vec)

        return [v for v in vectors if v is not None]

    def embed_query(self, text: str) -> list[float]:
        cached = self._read_cache(text)
        if cached is not None:
            return cached
        vector = self._embed_batch([text])[0]
        self._write_cache(text, vector)
        return vector
