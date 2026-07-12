"""Pipeline B: naive RAG (dense top-k retrieval + generation with citation).

retrieve -> build prompt -> generate -> parse structured output. Returns the
uniform system-output contract (see pipeline.contract) so it is directly
comparable with the other configurations in the eval harness.
"""

from __future__ import annotations

from config.settings import settings

from normarag.generation.llm import LLMClient, OpenAIClient
from normarag.generation.pricing import embedding_cost_usd
from normarag.generation.prompt import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    parse_structured_output,
)
from normarag.pipeline.contract import Timer, system_output
from normarag.retrieval.retriever import Retriever


def run_rag_naive(
    question: str,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    top_k: int | None = None,
) -> dict:
    retriever = retriever or Retriever()
    llm = llm or OpenAIClient()

    with Timer() as timer:
        contexts = retriever.retrieve(question, top_k=top_k or settings.top_k_dense)
        prompt = build_rag_prompt(question, contexts)
        result = llm.complete(prompt, system=SYSTEM_PROMPT)
        parsed = parse_structured_output(result.text)

    # Query-embedding cost is small but real; fold it into the reported cost.
    cost = result.cost_usd + embedding_cost_usd(settings.embedding_model, question)
    return system_output(
        answer=parsed["answer"],
        citations=parsed["citations"],
        abstained=parsed["abstained"],
        contexts=contexts,
        latency_ms=timer.latency_ms,
        cost_usd=cost,
    )
