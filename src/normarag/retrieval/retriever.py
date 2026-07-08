"""Dense retrieval (M2). Hybrid dense+sparse fusion (RRF) is added in M4."""

from __future__ import annotations

from config.settings import settings

from normarag.index.store import QdrantStore


class Retriever:
    """Thin wrapper over the vector store's dense search."""

    def __init__(self, store: QdrantStore | None = None) -> None:
        self.store = store or QdrantStore()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.store.search(query, top_k=top_k or settings.top_k_dense)
