"""FastAPI application exposing the RAG pipeline."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from normarag.pipeline.rag_naive import run_rag_naive

app = FastAPI(title="NormaRAG", version="0.1.0")


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
    source_url: str | None = None
    text: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    result = run_rag_naive(payload.question, top_k=payload.top_k)
    return QueryResponse(**result)
