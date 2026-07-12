"""Qdrant vector store: collection management, upsert, dense + sparse search.

The collection carries two named vectors per chunk:
  - "dense"  : OpenAI embedding (semantic similarity)
  - "sparse" : BM25 (fastembed) with server-side IDF (exact term / numeral / name
               matches — important for normative references)

Dense search alone powers config B; hybrid (dense + sparse, fused with RRF in
retrieval/retriever.py) powers config C. Dense embeddings are disk-cached, so
re-indexing to add the sparse vector does not re-pay for them.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from normarag.index.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

DENSE = "dense"
SPARSE = "sparse"

# Stable namespace so a chunk_id always maps to the same Qdrant point id,
# making re-indexing idempotent.
_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Metadata carried in each point's payload (everything a citation/answer needs).
_PAYLOAD_FIELDS = (
    "chunk_id",
    "chapter_code",
    "chapter_title",
    "title_section",
    "numeral",
    "text",
    "page_start",
    "page_end",
    "n_tokens",
    "source_url",
)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make ingest` first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.collection = collection or settings.qdrant_collection
        self.embedder = embedder or OpenAIEmbeddingProvider()
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            check_compatibility=settings.qdrant_check_compatibility,
        )
        self._sparse_model: Any = None

    # --- sparse (BM25) embedding via fastembed --------------------------------

    def _sparse(self) -> Any:
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            self._sparse_model = SparseTextEmbedding(settings.sparse_model)
        return self._sparse_model

    def _sparse_docs(self, texts: list[str]) -> list[SparseVector]:
        embs = self._sparse().embed(texts)
        return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embs]

    def _sparse_query(self, text: str) -> SparseVector:
        emb = next(iter(self._sparse().query_embed(text)))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    # --- collection management ------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
                },
                # IDF modifier makes Qdrant apply BM25 inverse-document-frequency
                # weighting server-side over the indexed corpus.
                sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
            )

    def upsert_chunks(self, chunks: list[dict], batch_size: int = 128) -> int:
        """Embed (dense + sparse) and upsert chunks. The embedding text prefixes
        the structural header (chapter/título/numeral) to improve retrieval."""
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [self._embedding_text(c) for c in batch]
            dense_vectors = self.embedder.embed(texts)
            sparse_vectors = self._sparse_docs(texts)
            points = [
                PointStruct(
                    id=_point_id(c["chunk_id"]),
                    vector={DENSE: dense, SPARSE: sparse},
                    payload={k: c.get(k) for k in _PAYLOAD_FIELDS},
                )
                for c, dense, sparse in zip(batch, dense_vectors, sparse_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    # --- search ---------------------------------------------------------------

    def _hits_to_dicts(self, hits: list) -> list[dict]:
        results: list[dict] = []
        for hit in hits:
            payload: dict[str, Any] = dict(hit.payload or {})
            payload["score"] = hit.score
            results.append(payload)
        return results

    def search_dense(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = top_k or settings.top_k_dense
        vector = self.embedder.embed_query(query)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE,
            limit=limit,
            with_payload=True,
        ).points
        return self._hits_to_dicts(hits)

    def search_sparse(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = top_k or settings.top_k_sparse
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self._sparse_query(query),
            using=SPARSE,
            limit=limit,
            with_payload=True,
        ).points
        return self._hits_to_dicts(hits)

    # Backward-compatible alias used by config B (dense-only).
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.search_dense(query, top_k)

    @staticmethod
    def _embedding_text(chunk: dict) -> str:
        header = " ".join(
            part
            for part in (
                chunk.get("chapter_code"),
                chunk.get("title_section"),
                chunk.get("numeral"),
                chunk.get("chapter_title"),
            )
            if part
        )
        return f"{header}\n{chunk.get('text', '')}".strip()


def build_index(recreate: bool = True) -> int:
    """Load chunks.jsonl, (re)create the collection, and upsert dense + sparse."""
    chunks = load_chunks()
    store = QdrantStore()
    store.ensure_collection(recreate=recreate)
    count = store.upsert_chunks(chunks)
    print(
        f"Indexed {count} chunks (dense+sparse) into '{store.collection}' at {settings.qdrant_url}"
    )
    return count


if __name__ == "__main__":
    build_index()
