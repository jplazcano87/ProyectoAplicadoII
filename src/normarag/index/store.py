"""Qdrant vector store: collection management, upsert, and dense search.

M2 uses dense-only search. Sparse/BM25 hybrid retrieval is added in M4; this
module is written so that path can be layered on without changing the schema.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from normarag.index.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider

CHUNKS_PATH = Path("data/processed/chunks.jsonl")

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
        self.client = client or QdrantClient(url=settings.qdrant_url)

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[dict], batch_size: int = 128) -> int:
        """Embed and upsert chunks. The embedding text prefixes the structural
        header (chapter/título/numeral) to improve retrieval, per CLAUDE.md §7."""
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [self._embedding_text(c) for c in batch]
            vectors = self.embedder.embed(texts)
            points = [
                PointStruct(
                    id=_point_id(c["chunk_id"]),
                    vector=vector,
                    payload={k: c.get(k) for k in _PAYLOAD_FIELDS},
                )
                for c, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Dense top-k search. Returns payloads with an added `score`."""
        limit = top_k or settings.top_k_dense
        vector = self.embedder.embed_query(query)
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=limit, with_payload=True
        ).points
        results: list[dict] = []
        for hit in hits:
            payload: dict[str, Any] = dict(hit.payload or {})
            payload["score"] = hit.score
            results.append(payload)
        return results

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
    """Load chunks.jsonl, (re)create the collection, and upsert everything."""
    chunks = load_chunks()
    store = QdrantStore()
    store.ensure_collection(recreate=recreate)
    count = store.upsert_chunks(chunks)
    print(f"Indexed {count} chunks into '{store.collection}' at {settings.qdrant_url}")
    return count


if __name__ == "__main__":
    build_index()
