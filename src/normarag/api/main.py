"""FastAPI application exposing the RAG pipeline."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from normarag.pipeline.baseline import run_baseline
from normarag.pipeline.rag_improved import run_rag_improved
from normarag.pipeline.rag_naive import run_rag_naive

app = FastAPI(title="NormaRAG", version="0.1.0")

# A = baseline (LLM only), B = RAG naive (dense), C = RAG improved (hybrid+rerank).
CONFIGS = {"A": run_baseline, "B": run_rag_naive, "C": run_rag_improved}


class Citation(BaseModel):
    chapter: str
    numeral: str


class Source(BaseModel):
    chunk_id: str | None = None
    chapter_code: str | None = None
    chapter_title: str | None = None
    title_section: str | None = None
    numeral: str | None = None
    score: float | None = None
    rerank_score: float | None = None
    source_url: str | None = None
    text: str | None = None


class QueryRequest(BaseModel):
    question: str
    config: str = "C"  # A | B | C
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    config: str
    latency_ms: float | None = None
    cost_usd: float | None = None
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _source(chunk: dict) -> Source:
    return Source(
        chunk_id=chunk.get("chunk_id"),
        chapter_code=chunk.get("chapter_code"),
        chapter_title=chunk.get("chapter_title"),
        title_section=chunk.get("title_section"),
        numeral=chunk.get("numeral"),
        score=chunk.get("score"),
        rerank_score=chunk.get("rerank_score"),
        source_url=chunk.get("source_url"),
        text=chunk.get("text"),
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    run = CONFIGS.get(payload.config)
    if run is None:
        raise HTTPException(status_code=422, detail=f"unknown config '{payload.config}' (A|B|C)")

    # top_k is an exploration tool exposed only for retrieval configs.
    result = (
        run(payload.question)
        if payload.config == "A"
        else run(payload.question, top_k=payload.top_k)
    )
    return QueryResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        abstained=result["abstained"],
        config=payload.config,
        latency_ms=result.get("latency_ms"),
        cost_usd=result.get("cost_usd"),
        sources=[_source(c) for c in result["contexts"]],
    )
