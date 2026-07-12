"""Retrieval: dense (config B) and hybrid dense+sparse with RRF (config C).

Reciprocal Rank Fusion (RRF) merges the dense and sparse rankings: the dense
vector captures semantics, while the sparse BM25 vector rescues exact matches of
normative terms, chapter numbers and proper names that dense embeddings blur.
Fusion is done client-side (not server-side) so it is explicit and unit-testable.
"""

from __future__ import annotations

from config.settings import settings

from normarag.index.store import QdrantStore


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int | None = None,
    key: str = "chunk_id",
) -> list[dict]:
    """Fuse several ranked lists of chunks into one via RRF.

    Each chunk's fused score is sum(1 / (k + rank)) over the lists it appears in
    (rank is 0-indexed). Chunks are deduplicated by `key`; the returned list is
    sorted by descending fused score and each item carries `rrf_score`.
    """
    k = k if k is not None else settings.rrf_k
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            ident = chunk.get(key)
            if ident is None:
                continue
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            # Keep the first-seen payload (they share the same content/metadata).
            fused.setdefault(ident, chunk)

    ordered = sorted(fused.values(), key=lambda c: scores[c[key]], reverse=True)
    for chunk in ordered:
        chunk["rrf_score"] = scores[chunk[key]]
    return ordered


class Retriever:
    """Dense or hybrid (dense + sparse RRF) retrieval over the vector store."""

    def __init__(self, store: QdrantStore | None = None) -> None:
        self.store = store or QdrantStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_hybrid: bool = False,
    ) -> list[dict]:
        """Return candidate chunks for a query.

        Dense-only (config B) when use_hybrid is False. When True (config C),
        fuse dense top_k_dense and sparse top_k_sparse candidates with RRF.
        """
        if not use_hybrid:
            return self.store.search_dense(query, top_k=top_k or settings.top_k_dense)

        dense = self.store.search_dense(query, top_k=settings.top_k_dense)
        sparse = self.store.search_sparse(query, top_k=settings.top_k_sparse)
        fused = reciprocal_rank_fusion([dense, sparse])
        if top_k is not None:
            return fused[:top_k]
        return fused
