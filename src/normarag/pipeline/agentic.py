"""Pipeline D: agentic RAG with LangGraph (the M6 novel element).

Graph: decompose -> retrieve -> synthesize -> verify.
  decompose  : split a complex question into atomic sub-questions
  retrieve   : hybrid retrieve (+ optional rerank) per sub-question, merge/dedup
  synthesize : answer with the strict citation prompt over the merged context
  verify     : self-check — drop any cited (chapter, numeral) that does not exist
               in the corpus (rejects fabricated sub-numerals like "2(a)", "1.a"
               that showed up as citation_invalid in M5)

Emits the same uniform contract as A/B/C plus `sub_questions` and
`rejected_citations` for the UI / analysis. Independent of A/B/C.
"""

from __future__ import annotations

from typing import TypedDict

from config.settings import settings

from normarag.generation.llm import LLMClient, LLMResult, OpenAIClient
from normarag.generation.pricing import embedding_cost_usd
from normarag.generation.prompt import (
    CITATION_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    build_citation_prompt,
    build_decompose_prompt,
    parse_structured_output,
    parse_sub_questions,
)
from normarag.pipeline.contract import Timer, system_output
from normarag.retrieval.rerank import Reranker
from normarag.retrieval.retriever import Retriever


class AgentState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    contexts: list[dict]
    answer: str
    citations: list[dict]
    abstained: bool
    rejected_citations: list[dict]
    cost_usd: float


class Agent:
    """Compiles the LangGraph agent once; `run` executes it per question."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        reranker: Reranker | None,
        known_keys: set[tuple[str, str]],
        top_k_rerank: int | None = None,
        use_decompose: bool = True,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.known = known_keys
        self.top_k = top_k_rerank or settings.top_k_rerank
        self.use_decompose = use_decompose
        self.graph = self._build()

    # --- nodes ----------------------------------------------------------------

    def _decompose(self, state: AgentState) -> AgentState:
        question = state["question"]
        if not self.use_decompose:
            return {"sub_questions": [question]}
        res = self.llm.complete(build_decompose_prompt(question), system=DECOMPOSE_SYSTEM_PROMPT)
        subs = parse_sub_questions(res.text, fallback=question)
        return {"sub_questions": subs, "cost_usd": state.get("cost_usd", 0.0) + res.cost_usd}

    def _retrieve(self, state: AgentState) -> AgentState:
        merged: dict[str, dict] = {}
        for sub in state["sub_questions"]:
            candidates = self.retriever.retrieve(sub, use_hybrid=True)
            if self.reranker is not None:
                candidates = self.reranker.rerank(sub, candidates, top_k=self.top_k)
            else:
                candidates = candidates[: self.top_k]
            for c in candidates:
                merged.setdefault(c["chunk_id"], c)
        cost = state.get("cost_usd", 0.0) + sum(
            embedding_cost_usd(settings.embedding_model, s) for s in state["sub_questions"]
        )
        return {"contexts": list(merged.values()), "cost_usd": cost}

    def _synthesize(self, state: AgentState) -> AgentState:
        prompt = build_citation_prompt(state["question"], state["contexts"])
        res = self.llm.complete(prompt, system=CITATION_SYSTEM_PROMPT)
        parsed = parse_structured_output(res.text)
        return {
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "abstained": parsed["abstained"],
            "cost_usd": state.get("cost_usd", 0.0) + res.cost_usd,
        }

    def _verify(self, state: AgentState) -> AgentState:
        """Self-check: keep only citations that exist in the corpus."""
        valid, rejected = [], []
        for c in state.get("citations", []):
            key = (str(c.get("chapter", "")), str(c.get("numeral", "")))
            (valid if key in self.known else rejected).append(c)
        return {"citations": valid, "rejected_citations": rejected}

    # --- graph ----------------------------------------------------------------

    def _build(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(AgentState)
        g.add_node("decompose", self._decompose)
        g.add_node("retrieve", self._retrieve)
        g.add_node("synthesize", self._synthesize)
        g.add_node("verify", self._verify)
        g.set_entry_point("decompose")
        g.add_edge("decompose", "retrieve")
        g.add_edge("retrieve", "synthesize")
        g.add_edge("synthesize", "verify")
        g.add_edge("verify", END)
        return g.compile()

    def run(self, question: str) -> dict:
        with Timer() as timer:
            final: AgentState = self.graph.invoke({"question": question, "cost_usd": 0.0})
        cost = final.get("cost_usd", 0.0) + embedding_cost_usd(settings.embedding_model, question)
        out = system_output(
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            abstained=bool(final.get("abstained", False)),
            contexts=final.get("contexts", []),
            latency_ms=timer.latency_ms,
            cost_usd=cost,
        )
        out["sub_questions"] = final.get("sub_questions", [question])
        out["rejected_citations"] = final.get("rejected_citations", [])
        return out


def run_agentic(
    question: str,
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    known_keys: set[tuple[str, str]] | None = None,
    *,
    use_decompose: bool = True,
    top_k_rerank: int | None = None,
) -> dict:
    if known_keys is None:
        from normarag.eval.metrics import known_citation_keys
        from normarag.index.store import load_chunks

        known_keys = known_citation_keys(load_chunks())
    if reranker is None:
        from normarag.retrieval.rerank import CohereReranker

        reranker = CohereReranker()
    agent = Agent(
        llm=llm or OpenAIClient(),
        retriever=retriever or Retriever(),
        reranker=reranker,
        known_keys=known_keys,
        top_k_rerank=top_k_rerank,
        use_decompose=use_decompose,
    )
    return agent.run(question)


__all__ = ["Agent", "AgentState", "LLMResult", "run_agentic"]
