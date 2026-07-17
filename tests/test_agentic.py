"""Tests for config D (agentic) and the C-v2 interventions (no network)."""

from __future__ import annotations

from normarag.generation.llm import LLMResult
from normarag.pipeline.agentic import Agent
from normarag.pipeline.rag_improved import run_rag_improved

KNOWN = {("1-8", "1.2"), ("1-8", "1.1")}


def _chunk(cid, numeral="1.2"):
    return {"chunk_id": cid, "chapter_code": "1-8", "numeral": numeral, "text": f"txt {cid}"}


class FakeRetriever:
    def __init__(self, pool):
        self.pool = pool
        self.calls = []

    def retrieve(self, query, use_hybrid=False):
        self.calls.append(query)
        return list(self.pool)


class FakeReranker:
    def rerank(self, query, candidates, top_k=None):
        return candidates[: (top_k or len(candidates))]


class FakeLLM:
    """Returns a decomposition or a synthesis reply based on the system prompt."""

    def __init__(self, sub_questions, synth_reply):
        self.sub_questions = sub_questions
        self.synth_reply = synth_reply
        self.calls = []

    def complete(self, prompt, system=None):
        self.calls.append((prompt, system))
        if system and "planificador" in system:  # decompose node
            subs = ", ".join(f'"{s}"' for s in self.sub_questions)
            return LLMResult(text=f'{{"sub_questions": [{subs}]}}', cost_usd=0.0001)
        return LLMResult(text=self.synth_reply, cost_usd=0.0002)


def _agent(llm, pool, reranker=None, use_decompose=True):
    return Agent(
        llm=llm,
        retriever=FakeRetriever(pool),
        reranker=reranker,
        known_keys=KNOWN,
        top_k_rerank=6,
        use_decompose=use_decompose,
    )


# --- agentic self-check (the novel element) --------------------------------


def test_agentic_verify_rejects_fabricated_citation() -> None:
    # Synthesis cites a valid (1-8,1.2) and a fabricated (1-8, "2(a)").
    synth = (
        '{"answer": "resp", "citations": '
        '[{"chapter": "1-8", "numeral": "1.2"}, {"chapter": "1-8", "numeral": "2(a)"}], '
        '"abstained": false}'
    )
    llm = FakeLLM(["sub1"], synth)
    result = _agent(llm, [_chunk("c1")]).run("¿pregunta?")

    # Self-check keeps only the real citation, records the rejected one.
    assert result["citations"] == [{"chapter": "1-8", "numeral": "1.2"}]
    assert result["rejected_citations"] == [{"chapter": "1-8", "numeral": "2(a)"}]


def test_agentic_decomposes_and_retrieves_per_subquestion() -> None:
    llm = FakeLLM(["sub uno", "sub dos"], '{"answer":"x","citations":[],"abstained":true}')
    retriever = FakeRetriever([_chunk("c1"), _chunk("c2")])
    agent = Agent(llm=llm, retriever=retriever, reranker=None, known_keys=KNOWN)
    result = agent.run("compleja")

    assert result["sub_questions"] == ["sub uno", "sub dos"]
    assert retriever.calls == ["sub uno", "sub dos"]  # retrieved per sub-question
    assert result["abstained"] is True


def test_agentic_merges_and_dedups_contexts() -> None:
    llm = FakeLLM(["a", "b"], '{"answer":"x","citations":[],"abstained":false}')
    # Both sub-questions return the same chunk c1 plus a distinct one.
    pool_a = [_chunk("c1"), _chunk("c2")]
    agent = Agent(
        llm=llm, retriever=FakeRetriever(pool_a), reranker=FakeReranker(), known_keys=KNOWN
    )
    result = agent.run("q")
    ids = [c["chunk_id"] for c in result["contexts"]]
    assert sorted(ids) == ["c1", "c2"]  # deduped by chunk_id
    assert "latency_ms" in result and result["cost_usd"] > 0


def test_agentic_contract_shape() -> None:
    llm = FakeLLM(
        ["s"], '{"answer":"a","citations":[{"chapter":"1-8","numeral":"1.1"}],"abstained":false}'
    )
    result = _agent(llm, [_chunk("c1")]).run("q")
    for key in ("answer", "citations", "abstained", "contexts", "latency_ms", "cost_usd"):
        assert key in result


# --- C-v2 interventions -----------------------------------------------------


class FakeLLMSimple:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt, system=None):
        self.prompts.append((prompt, system))
        return LLMResult(text=self.reply, cost_usd=0.0001)


class FakeRetriever2:
    def __init__(self, pool):
        self.pool = pool

    def retrieve(self, query, use_hybrid=False):
        return list(self.pool)


_REPLY = '{"answer": "ok", "citations": [{"chapter": "1-8", "numeral": "1.2"}], "abstained": false}'


def test_cv2_exclude_preamble_drops_numeral_zero() -> None:
    pool = [_chunk("pre", numeral="0"), _chunk("c1", numeral="1.2"), _chunk("c2", numeral="1.1")]
    retriever = FakeRetriever2(pool)
    llm = FakeLLMSimple(_REPLY)
    result = run_rag_improved(
        "q", retriever=retriever, llm=llm, use_rerank=False, exclude_preamble=True
    )
    numerals = [c["numeral"] for c in result["contexts"]]
    assert "0" not in numerals  # preamble excluded


def test_cv2_soft_abstention_uses_soft_prompt() -> None:
    retriever = FakeRetriever2([_chunk("c1")])
    llm = FakeLLMSimple(_REPLY)
    run_rag_improved(
        "q",
        retriever=retriever,
        llm=llm,
        use_rerank=False,
        use_citation_prompt=True,
        soft_abstention=True,
    )
    system = llm.prompts[0][1]
    assert "Abstente SOLO si" in system  # soft system prompt in use
