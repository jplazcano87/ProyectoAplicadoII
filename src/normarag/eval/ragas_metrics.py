"""RAGAS metric wrappers (faithfulness, answer_relevancy, context_precision,
context_recall), judged by JUDGE_MODEL — distinct from the generator to avoid
self-evaluation bias (CLAUDE.md §9).

RAGAS pulls in a large, version-fragile dependency stack. To keep the harness
robust, everything here is imported lazily and guarded: if RAGAS (or the judge
key) is unavailable, `evaluate_ragas` returns None and the runner falls back to
the programmatic citation metrics only. Targets the RAGAS >=0.2 EvaluationDataset
API.
"""

from __future__ import annotations

import warnings

from config.settings import settings

RAGAS_COLUMNS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except Exception:
        return False


def evaluate_ragas(
    rows: list[dict],
    judge_model: str | None = None,
    embedding_model: str | None = None,
    api_key: str | None = None,
) -> list[dict] | None:
    """Compute RAGAS metrics for a batch of rows.

    Each row needs: `question` (str), `answer` (str), `contexts` (list[str]),
    `ground_truth` (str, the gold reference answer). Returns a list aligned with
    `rows` of {metric: score}, or None if RAGAS cannot run (import/config error).
    """
    api_key = api_key or settings.openai_api_key or None
    judge_model = judge_model or settings.judge_model
    embedding_model = embedding_model or settings.embedding_model

    if not rows:
        return []
    if not api_key:
        warnings.warn("No OPENAI_API_KEY; skipping RAGAS metrics.", stacklevel=2)
        return None

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:  # pragma: no cover - depends on optional stack
        warnings.warn(f"RAGAS unavailable ({exc}); skipping RAGAS metrics.", stacklevel=2)
        return None

    try:  # pragma: no cover - requires network + judge model
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"] or "",
                # RAGAS context metrics need at least one context string.
                retrieved_contexts=(r.get("contexts") or [""]),
                reference=r.get("ground_truth") or "",
            )
            for r in rows
        ]
        dataset = EvaluationDataset(samples=samples)
        judge = LangchainLLMWrapper(
            ChatOpenAI(model=judge_model, temperature=0, api_key=api_key)  # type: ignore[arg-type]
        )
        embedder = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=embedding_model, api_key=api_key)  # type: ignore[arg-type]
        )
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge,
            embeddings=embedder,
        )
        frame = result.to_pandas()
        out: list[dict] = []
        for _, row in frame.iterrows():
            out.append({col: _to_float(row.get(col)) for col in RAGAS_COLUMNS})
        return out
    except Exception as exc:  # pragma: no cover - runtime/network errors
        warnings.warn(f"RAGAS evaluation failed ({exc}); skipping RAGAS metrics.", stacklevel=2)
        return None


def _to_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN
