"""Pipeline C: improved RAG (hybrid retrieval + rerank + citation prompt).

Orchestrates: embed question -> hybrid search (dense + sparse) -> RRF -> Cohere
rerank -> strict citation/abstention prompt -> structured output.

Each improvement is an independent toggle so M5 can measure its incremental
contribution via ablations without new code:
  use_hybrid          : dense+sparse RRF  vs  dense only
  use_rerank          : Cohere rerank     vs  no rerank
  use_citation_prompt : strict prompt     vs  naive prompt

Returns the same uniform contract as configs A and B.
"""

from __future__ import annotations

from config.settings import settings

from normarag.generation.llm import LLMClient, OpenAIClient
from normarag.generation.pricing import embedding_cost_usd
from normarag.generation.prompt import (
    CITATION_SYSTEM_PROMPT,
    SOFT_CITATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_citation_prompt,
    build_rag_prompt,
    build_soft_citation_prompt,
    parse_structured_output,
)
from normarag.pipeline.contract import Timer, system_output
from normarag.retrieval.rerank import CohereReranker, Reranker
from normarag.retrieval.retriever import Retriever


def _resolve(value: bool | None, default: bool) -> bool:
    return default if value is None else value


def run_rag_improved(
    question: str,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
    *,
    use_hybrid: bool | None = None,
    use_rerank: bool | None = None,
    use_citation_prompt: bool | None = None,
    top_k_rerank: int | None = None,
    soft_abstention: bool | None = None,
    exclude_preamble: bool | None = None,
) -> dict:
    use_hybrid = _resolve(use_hybrid, settings.use_hybrid)
    use_rerank = _resolve(use_rerank, settings.use_rerank)
    use_citation_prompt = _resolve(use_citation_prompt, settings.use_citation_prompt)
    # C-v2 interventions (M6), off by default so config C is unchanged.
    soft_abstention = _resolve(soft_abstention, settings.soft_abstention)
    exclude_preamble = _resolve(exclude_preamble, settings.exclude_preamble)
    # Number of chunks kept as context (after rerank, or from the fused list).
    # Overridable for the M5 top-k sweep; leaves everything else fixed.
    top_k_rerank = top_k_rerank if top_k_rerank is not None else settings.top_k_rerank

    retriever = retriever or Retriever()
    llm = llm or OpenAIClient()
    if use_rerank and reranker is None:
        reranker = CohereReranker()

    with Timer() as timer:
        # Retrieve a broad candidate pool, then narrow with the reranker.
        candidates = retriever.retrieve(question, use_hybrid=use_hybrid)
        # C-v2: drop preamble chunks (numeral "0") so the reranker can't surface
        # them instead of the specific numeral (a ranking_miss cause in M5).
        if exclude_preamble:
            candidates = [c for c in candidates if str(c.get("numeral")) != "0"] or candidates
        if use_rerank and reranker is not None:
            contexts = reranker.rerank(question, candidates, top_k=top_k_rerank)
        else:
            contexts = candidates[:top_k_rerank]

        if use_citation_prompt and soft_abstention:
            prompt = build_soft_citation_prompt(question, contexts)
            system = SOFT_CITATION_SYSTEM_PROMPT
        elif use_citation_prompt:
            prompt = build_citation_prompt(question, contexts)
            system = CITATION_SYSTEM_PROMPT
        else:
            prompt = build_rag_prompt(question, contexts)
            system = SYSTEM_PROMPT

        result = llm.complete(prompt, system=system)
        parsed = parse_structured_output(result.text)

    cost = result.cost_usd + embedding_cost_usd(settings.embedding_model, question)
    return system_output(
        answer=parsed["answer"],
        citations=parsed["citations"],
        abstained=parsed["abstained"],
        contexts=contexts,
        latency_ms=timer.latency_ms,
        cost_usd=cost,
    )
