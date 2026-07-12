"""Tests for M4 config C: RRF fusion, reranker (mocked), and the toggles."""

from __future__ import annotations

from normarag.generation.llm import LLMResult
from normarag.generation.prompt import build_citation_prompt, parse_structured_output
from normarag.pipeline.rag_improved import run_rag_improved
from normarag.retrieval.rerank import Reranker
from normarag.retrieval.retriever import reciprocal_rank_fusion


def _chunk(cid: str, text: str = "x") -> dict:
    return {"chunk_id": cid, "chapter_code": cid.split("-")[0], "numeral": "1", "text": text}


# --- RRF fusion -------------------------------------------------------------


def test_rrf_fuses_two_known_rankings() -> None:
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    sparse = [_chunk("b"), _chunk("d"), _chunk("a")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [c["chunk_id"] for c in fused]
    # 'a' (ranks 0 and 2) and 'b' (ranks 1 and 0) appear in both -> top two.
    assert set(ids[:2]) == {"a", "b"}
    # Dedup: every id appears once.
    assert len(ids) == len(set(ids)) == 4
    # Scores are descending and attached.
    scores = [c["rrf_score"] for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_single_ranking_preserves_order() -> None:
    ranking = [_chunk("a"), _chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([ranking])
    assert [c["chunk_id"] for c in fused] == ["a", "b", "c"]


def test_rrf_rank_beats_frequency_with_small_k() -> None:
    # With small k, a top-ranked item can outweigh one merely present twice lower.
    dense = [_chunk("top"), _chunk("x"), _chunk("y"), _chunk("dup")]
    sparse = [_chunk("z"), _chunk("w"), _chunk("dup")]
    fused = reciprocal_rank_fusion([dense, sparse], k=1)
    assert fused[0]["chunk_id"] == "top"


# --- Reranker (mocked) ------------------------------------------------------


class FakeReranker(Reranker):
    """Reverses candidate order and tags a rerank_score, no network."""

    def __init__(self) -> None:
        self.calls: list = []

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        self.calls.append((query, len(candidates), top_k))
        ordered = list(reversed(candidates))[: (top_k or len(candidates))]
        for i, c in enumerate(ordered):
            c = dict(c)
            c["rerank_score"] = 1.0 - i * 0.1
            ordered[i] = c
        return ordered


class FakeRetriever:
    def __init__(self, contexts: list[dict]) -> None:
        self.contexts = contexts
        self.calls: list = []

    def retrieve(self, query: str, top_k=None, use_hybrid: bool = False) -> list[dict]:
        self.calls.append((query, top_k, use_hybrid))
        return self.contexts


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list = []

    def complete(self, prompt: str, system=None) -> LLMResult:
        self.prompts.append((prompt, system))
        return LLMResult(text=self.reply, prompt_tokens=5, completion_tokens=3, cost_usd=0.0001)


_REPLY = '{"answer": "ok", "citations": [{"chapter": "1-8", "numeral": "2"}], "abstained": false}'


def test_config_c_full_pipeline_reranks_and_uses_hybrid() -> None:
    retriever = FakeRetriever([_chunk("a"), _chunk("b"), _chunk("c")])
    reranker = FakeReranker()
    llm = FakeLLM(_REPLY)
    result = run_rag_improved(
        "¿pregunta?",
        retriever=retriever,  # type: ignore[arg-type]
        reranker=reranker,
        llm=llm,  # type: ignore[arg-type]
        use_hybrid=True,
        use_rerank=True,
        use_citation_prompt=True,
    )
    assert result["answer"] == "ok"
    assert result["citations"] == [{"chapter": "1-8", "numeral": "2"}]
    # Hybrid retrieval was requested and the reranker ran.
    assert retriever.calls[0][2] is True
    assert reranker.calls and result["contexts"][0].get("rerank_score") is not None


def test_toggle_rerank_off_skips_reranker() -> None:
    retriever = FakeRetriever([_chunk("a"), _chunk("b")])
    reranker = FakeReranker()
    llm = FakeLLM(_REPLY)
    result = run_rag_improved(
        "q",
        retriever=retriever,  # type: ignore[arg-type]
        reranker=reranker,
        llm=llm,  # type: ignore[arg-type]
        use_rerank=False,
    )
    assert reranker.calls == []  # reranker not invoked
    assert all("rerank_score" not in c for c in result["contexts"])


def test_toggle_hybrid_off_uses_dense_only() -> None:
    retriever = FakeRetriever([_chunk("a")])
    llm = FakeLLM(_REPLY)
    run_rag_improved(
        "q",
        retriever=retriever,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        use_hybrid=False,
        use_rerank=False,
    )
    assert retriever.calls[0][2] is False  # use_hybrid=False propagated


def test_toggle_citation_prompt_changes_system_prompt() -> None:
    retriever = FakeRetriever([_chunk("a")])
    llm_strict = FakeLLM(_REPLY)
    run_rag_improved(
        "q",
        retriever=retriever,
        llm=llm_strict,  # type: ignore[arg-type]
        use_rerank=False,
        use_citation_prompt=True,
    )
    llm_naive = FakeLLM(_REPLY)
    run_rag_improved(
        "q",
        retriever=retriever,
        llm=llm_naive,  # type: ignore[arg-type]
        use_rerank=False,
        use_citation_prompt=False,
    )
    # Different system prompts for strict vs naive.
    assert llm_strict.prompts[0][1] != llm_naive.prompts[0][1]
    assert "ESTRICTAS" in llm_strict.prompts[0][1]


# --- structured output parse (citation prompt) ------------------------------


def test_citation_prompt_contains_strict_contract() -> None:
    prompt = build_citation_prompt("¿q?", [_chunk("1-8", "texto")])
    assert "No puedo responder con la normativa disponible" in prompt
    assert '"abstained"' in prompt


def test_parse_abstention_from_citation_output() -> None:
    raw = (
        '{"answer": "No puedo responder con la normativa disponible", '
        '"citations": [], "abstained": true}'
    )
    out = parse_structured_output(raw)
    assert out["abstained"] is True
    assert out["citations"] == []


def test_parse_normalizes_capitulo_prefix_in_chapter() -> None:
    # Models sometimes echo "Capítulo 1-8" into the chapter field; must normalize.
    raw = (
        '{"answer": "x", "citations": [{"chapter": "Capítulo 1-8", "numeral": "1.2"}], '
        '"abstained": false}'
    )
    out = parse_structured_output(raw)
    assert out["citations"] == [{"chapter": "1-8", "numeral": "1.2"}]
