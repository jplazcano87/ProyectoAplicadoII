"""Tests for indexing and dense retrieval (no network; fakes/mocks)."""

from __future__ import annotations

from types import SimpleNamespace

from normarag.index.embeddings import EmbeddingProvider
from normarag.index.store import QdrantStore, _point_id
from normarag.retrieval.retriever import Retriever


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedder: vector length encodes call, values are fixed."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[float(len(t))] * self._dim for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] * self._dim

    @property
    def dim(self) -> int:
        return self._dim


class FakeSparse:
    """Stand-in for fastembed BM25: returns fixed indices/values, no download."""

    def embed(self, texts: list[str]):
        return [SimpleNamespace(indices=_Arr([1, 2]), values=_Arr([0.5, 0.5])) for _ in texts]

    def query_embed(self, text: str):
        return iter([SimpleNamespace(indices=_Arr([1]), values=_Arr([1.0]))])


class _Arr(list):
    def tolist(self):  # mimic numpy .tolist()
        return list(self)


class FakeQdrantClient:
    """Records upserts and returns canned search hits."""

    def __init__(self, hits: list[SimpleNamespace] | None = None) -> None:
        self.upserted: list = []
        self.created_with: dict | None = None
        self._exists = False
        self._hits = hits or []

    def collection_exists(self, name: str) -> bool:
        return self._exists

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        self.created_with = {
            "name": collection_name,
            "size": vectors_config["dense"].size,
            "sparse": list(sparse_vectors_config or {}),
        }
        self._exists = True

    def delete_collection(self, name: str) -> None:
        self._exists = False

    def upsert(self, collection_name: str, points: list) -> None:
        self.upserted.extend(points)

    def query_points(self, collection_name, query, limit, with_payload, using=None):
        return SimpleNamespace(points=self._hits[:limit])


def _chunk(chunk_id: str, text: str) -> dict:
    code = chunk_id.split("::")[0]
    return {
        "chunk_id": chunk_id,
        "chapter_code": code,
        "chapter_title": "Título de prueba",
        "title_section": "II",
        "numeral": "2.1",
        "text": text,
        "page_start": 1,
        "page_end": 1,
        "n_tokens": 3,
        "source_url": "https://example.test/x.pdf",
    }


def test_point_id_is_stable_and_deterministic() -> None:
    assert _point_id("1-13::II::2.1::0") == _point_id("1-13::II::2.1::0")
    assert _point_id("1-13::II::2.1::0") != _point_id("1-13::II::2.1::1")


def test_embedding_text_prefixes_structural_header() -> None:
    text = QdrantStore._embedding_text(_chunk("1-13::II::2.1::0", "cuerpo del numeral"))
    assert text.startswith("1-13 II 2.1 Título de prueba")
    assert "cuerpo del numeral" in text


def test_ensure_collection_creates_dense_and_sparse() -> None:
    client = FakeQdrantClient()
    embedder = FakeEmbedder(dim=7)
    store = QdrantStore(client=client, collection="c", embedder=embedder)
    store.ensure_collection()
    assert client.created_with == {"name": "c", "size": 7, "sparse": ["sparse"]}


def test_upsert_chunks_embeds_and_upserts_all() -> None:
    client = FakeQdrantClient()
    embedder = FakeEmbedder()
    store = QdrantStore(client=client, collection="c", embedder=embedder)
    store._sparse_model = FakeSparse()  # avoid loading the real BM25 model
    chunks = [_chunk(f"1-1::I::1.{i}::0", f"texto {i}") for i in range(3)]

    n = store.upsert_chunks(chunks, batch_size=2)

    assert n == 3
    assert len(client.upserted) == 3
    # Point ids derive from chunk_id, payload carries the metadata.
    ids = {p.id for p in client.upserted}
    assert ids == {_point_id(c["chunk_id"]) for c in chunks}
    assert client.upserted[0].payload["chapter_code"] == "1-1"
    assert "score" not in client.upserted[0].payload


def test_search_returns_payloads_with_score() -> None:
    hits = [
        SimpleNamespace(payload={"chunk_id": "1-1::I::1::0", "text": "a"}, score=0.9),
        SimpleNamespace(payload={"chunk_id": "1-2::I::1::0", "text": "b"}, score=0.5),
    ]
    client = FakeQdrantClient(hits=hits)
    store = QdrantStore(client=client, collection="c", embedder=FakeEmbedder())

    results = store.search_dense("plazo de informe", top_k=5)

    assert [r["chunk_id"] for r in results] == ["1-1::I::1::0", "1-2::I::1::0"]
    assert results[0]["score"] == 0.9


def test_retriever_dense_delegates_to_store() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls: list = []

        def search_dense(self, query: str, top_k=None) -> list[dict]:
            self.calls.append((query, top_k))
            return [{"chunk_id": "x", "score": 1.0}]

    store = FakeStore()
    retriever = Retriever(store=store)  # type: ignore[arg-type]
    out = retriever.retrieve("¿cuál es el plazo?", top_k=3)

    assert out == [{"chunk_id": "x", "score": 1.0}]
    assert store.calls == [("¿cuál es el plazo?", 3)]


def test_retriever_hybrid_fuses_dense_and_sparse() -> None:
    class FakeStore:
        def search_dense(self, query: str, top_k=None) -> list[dict]:
            return [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]

        def search_sparse(self, query: str, top_k=None) -> list[dict]:
            return [{"chunk_id": "b", "score": 2.0}, {"chunk_id": "c", "score": 1.0}]

    retriever = Retriever(store=FakeStore())  # type: ignore[arg-type]
    out = retriever.retrieve("q", use_hybrid=True)
    ids = {c["chunk_id"] for c in out}
    assert ids == {"a", "b", "c"}  # fused + deduped
    assert out[0]["chunk_id"] == "b"  # appears in both -> highest RRF
