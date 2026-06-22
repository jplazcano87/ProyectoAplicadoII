"""FastAPI application — POST /query endpoint."""

from fastapi import FastAPI

app = FastAPI(title="NormaRAG", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(payload: dict) -> dict:
    raise NotImplementedError("M2")
