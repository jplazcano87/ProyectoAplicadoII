"""Pipeline B: naive RAG (dense top-k retrieval + generation with citation).

retrieve -> build prompt -> generate -> parse structured output. The returned
dict follows the CLAUDE.md §9 contract plus the retrieved `sources` so the API
and UI can display where the answer came from.
"""

from __future__ import annotations

from config.settings import settings

from normarag.generation.llm import LLMClient, OpenAIClient
from normarag.generation.prompt import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    parse_structured_output,
)
from normarag.retrieval.retriever import Retriever


def _source_view(chunk: dict) -> dict:
    """Trim a retrieved chunk to the fields the UI needs to show a source."""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "chapter_code": chunk.get("chapter_code"),
        "chapter_title": chunk.get("chapter_title"),
        "title_section": chunk.get("title_section"),
        "numeral": chunk.get("numeral"),
        "score": chunk.get("score"),
        "source_url": chunk.get("source_url"),
        "text": chunk.get("text"),
    }


def run_rag_naive(
    question: str,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    top_k: int | None = None,
) -> dict:
    retriever = retriever or Retriever()
    llm = llm or OpenAIClient()

    contexts = retriever.retrieve(question, top_k=top_k or settings.top_k_dense)
    prompt = build_rag_prompt(question, contexts)
    raw = llm.complete(prompt, system=SYSTEM_PROMPT)
    result = parse_structured_output(raw)
    result["sources"] = [_source_view(c) for c in contexts]
    return result
