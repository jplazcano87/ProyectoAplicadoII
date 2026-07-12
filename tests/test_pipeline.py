"""Tests for prompt building, output parsing and the naive RAG pipeline."""

from __future__ import annotations

from normarag.generation.llm import LLMResult
from normarag.generation.prompt import (
    build_rag_prompt,
    format_context,
    parse_structured_output,
)
from normarag.pipeline.baseline import run_baseline
from normarag.pipeline.rag_naive import run_rag_naive


def _ctx() -> list[dict]:
    return [
        {
            "chunk_id": "1-8::::2::0",
            "chapter_code": "1-8",
            "chapter_title": "Canales de atención bancarios",
            "title_section": "",
            "numeral": "2",
            "text": "Los bancos que cierren una oficina deberán avisar con 90 días.",
            "score": 0.88,
            "source_url": "https://example.test/1-8.pdf",
        }
    ]


# --- Output parsing ---------------------------------------------------------


def test_parse_clean_json() -> None:
    raw = (
        '{"answer": "90 días de anticipación.", '
        '"citations": [{"chapter": "1-8", "numeral": "2"}], "abstained": false}'
    )
    out = parse_structured_output(raw)
    assert out["answer"] == "90 días de anticipación."
    assert out["citations"] == [{"chapter": "1-8", "numeral": "2"}]
    assert out["abstained"] is False


def test_parse_json_wrapped_in_prose_and_fences() -> None:
    raw = 'Claro:\n```json\n{"answer": "x", "citations": [], "abstained": true}\n```'
    out = parse_structured_output(raw)
    assert out["answer"] == "x"
    assert out["citations"] == []
    assert out["abstained"] is True


def test_parse_coerces_alt_citation_keys_and_types() -> None:
    raw = '{"answer": "y", "citations": [{"chapter_code": 1, "numeral": 2.1}], "abstained": 0}'
    out = parse_structured_output(raw)
    assert out["citations"] == [{"chapter": "1", "numeral": "2.1"}]
    assert out["abstained"] is False


def test_parse_non_json_falls_back_to_raw_text() -> None:
    out = parse_structured_output("no pude responder")
    assert out["answer"] == "no pude responder"
    assert out["citations"] == []
    assert out["abstained"] is False


# --- Prompt building --------------------------------------------------------


def test_format_context_includes_citable_headers() -> None:
    rendered = format_context(_ctx())
    assert "Capítulo 1-8" in rendered
    assert "Numeral 2" in rendered
    assert "90 días" in rendered


def test_build_rag_prompt_contains_question_and_contract() -> None:
    prompt = build_rag_prompt("¿Con cuánta anticipación se avisa un cierre?", _ctx())
    assert "PREGUNTA:" in prompt
    assert "¿Con cuánta anticipación se avisa un cierre?" in prompt
    assert '"abstained"' in prompt  # output contract present


# --- Pipeline orchestration -------------------------------------------------


class FakeRetriever:
    def __init__(self, contexts: list[dict]) -> None:
        self.contexts = contexts
        self.calls: list = []

    def retrieve(self, query: str, top_k=None) -> list[dict]:
        self.calls.append((query, top_k))
        return self.contexts


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list = []

    def complete(self, prompt: str, system=None) -> LLMResult:
        self.prompts.append((prompt, system))
        return LLMResult(text=self.reply, prompt_tokens=10, completion_tokens=5, cost_usd=0.0001)


def test_run_rag_naive_uniform_contract_with_contexts() -> None:
    retriever = FakeRetriever(_ctx())
    llm = FakeLLM(
        '{"answer": "Con 90 días.", "citations": [{"chapter": "1-8", "numeral": "2"}], '
        '"abstained": false}'
    )
    result = run_rag_naive(
        "¿Con cuánta anticipación?", retriever=retriever, llm=llm, top_k=4  # type: ignore[arg-type]
    )

    assert result["answer"] == "Con 90 días."
    assert result["citations"] == [{"chapter": "1-8", "numeral": "2"}]
    assert result["abstained"] is False
    # Uniform contract: contexts (full retrieved chunks) + latency + cost.
    assert len(result["contexts"]) == 1
    assert result["contexts"][0]["chapter_code"] == "1-8"
    assert result["contexts"][0]["score"] == 0.88
    assert result["latency_ms"] >= 0
    assert result["cost_usd"] > 0  # generation + query-embedding cost
    # Retrieval used the requested top_k; the LLM saw the system prompt.
    assert retriever.calls == [("¿Con cuánta anticipación?", 4)]
    assert llm.prompts[0][1] is not None


def test_run_rag_naive_propagates_abstention() -> None:
    retriever = FakeRetriever([])
    llm = FakeLLM('{"answer": "", "citations": [], "abstained": true}')
    result = run_rag_naive("pregunta sin evidencia", retriever=retriever, llm=llm)  # type: ignore[arg-type]
    assert result["abstained"] is True
    assert result["contexts"] == []


def test_run_baseline_has_no_contexts() -> None:
    llm = FakeLLM('{"answer": "Respuesta directa.", "citations": [], "abstained": false}')
    result = run_baseline("¿Pregunta general?", llm=llm)  # type: ignore[arg-type]
    assert result["answer"] == "Respuesta directa."
    assert result["contexts"] == []
    assert result["cost_usd"] > 0
    assert "latency_ms" in result
