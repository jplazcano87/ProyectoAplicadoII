"""Run a configuration over the golden set and write per-question results.

For each example: generate the system output (cache-aware), score it with the
programmatic citation metrics, and — optionally — with RAGAS. Results go to
results/<config>.jsonl, one row per question with the system output plus every
metric, so report.py can build the comparison table.

CLI:
  python -m normarag.eval.runner --config baseline|rag_naive --dataset <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.settings import settings

from normarag.eval.golden import GoldenExample, load_golden
from normarag.eval.metrics import known_citation_keys, per_example_metrics
from normarag.eval.ragas_metrics import RAGAS_COLUMNS, evaluate_ragas
from normarag.index.store import CHUNKS_PATH, load_chunks

RESULTS_DIR = Path("results")
CACHE_DIR = Path(".cache/eval")

Generator = Callable[[str], dict]
CONFIGS = ("baseline", "rag_naive")


def build_generator(config_name: str) -> Generator:
    """Construct the real (OpenAI/Qdrant-backed) generator for a config."""
    from normarag.generation.llm import OpenAIClient

    llm = OpenAIClient()
    if config_name == "baseline":
        from normarag.pipeline.baseline import run_baseline

        return lambda q: run_baseline(q, llm=llm)
    if config_name == "rag_naive":
        from normarag.pipeline.rag_naive import run_rag_naive
        from normarag.retrieval.retriever import Retriever

        retriever = Retriever()
        return lambda q: run_rag_naive(q, retriever=retriever, llm=llm)
    raise ValueError(f"Unknown config '{config_name}'. Choices: {CONFIGS}")


def _cache_key(config_name: str, question: str) -> str:
    payload = json.dumps(
        {
            "config": config_name,
            "generator_model": settings.generator_model,
            "top_k": settings.top_k_dense,
            "question": question,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_one(config_name: str, question: str, generate: Generator, use_cache: bool) -> dict:
    cache_path = CACHE_DIR / config_name / f"{_cache_key(config_name, question)}.json"
    if use_cache and cache_path.exists():
        cached: dict = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached
    output = generate(question)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return output


def _trim_context(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "chapter_code": chunk.get("chapter_code"),
        "numeral": chunk.get("numeral"),
        "score": chunk.get("score"),
        "text": chunk.get("text"),
    }


def _build_row(
    config_name: str, example: GoldenExample, output: dict, known: set[tuple[str, str]]
) -> dict:
    metrics = per_example_metrics(output, example, known)
    return {
        "id": example.id,
        "config": config_name,
        "question": example.question,
        "answerable": example.answerable,
        "difficulty": example.difficulty,
        "tags": example.tags,
        "gold_answer": example.answer,
        "references": [r.model_dump() for r in example.references],
        "answer": output.get("answer", ""),
        "citations": output.get("citations", []),
        "abstained": bool(output.get("abstained", False)),
        "contexts": [_trim_context(c) for c in output.get("contexts", [])],
        "latency_ms": output.get("latency_ms"),
        "cost_usd": output.get("cost_usd"),
        **metrics,
    }


def _attach_ragas(rows: list[dict], examples: list[GoldenExample]) -> None:
    ragas_rows = [
        {
            "question": ex.question,
            "answer": row["answer"],
            "contexts": [c["text"] for c in row["contexts"] if c.get("text")],
            "ground_truth": ex.answer,
        }
        for row, ex in zip(rows, examples)
    ]
    scores = evaluate_ragas(ragas_rows)
    for row, score in zip(rows, scores or [{}] * len(rows)):
        for col in RAGAS_COLUMNS:
            row[col] = (score or {}).get(col)


def run_config(
    config_name: str,
    dataset_path: str | Path,
    *,
    generate: Generator | None = None,
    use_cache: bool = True,
    max_workers: int = 4,
    skip_ragas: bool = False,
    chunks_path: Path = CHUNKS_PATH,
    results_dir: Path = RESULTS_DIR,
) -> list[dict]:
    examples = load_golden(dataset_path)
    known = known_citation_keys(load_chunks(chunks_path)) if Path(chunks_path).exists() else set()
    generate = generate or build_generator(config_name)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outputs = list(
            pool.map(
                lambda ex: _generate_one(config_name, ex.question, generate, use_cache),
                examples,
            )
        )

    rows = [_build_row(config_name, ex, out, known) for ex, out in zip(examples, outputs)]
    if not skip_ragas:
        _attach_ragas(rows, examples)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{config_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_cost = sum(r["cost_usd"] or 0.0 for r in rows)
    print(
        f"[{config_name}] {len(rows)} questions -> {out_path} " f"(approx cost ${total_cost:.4f})"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an eval config over the golden set.")
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--dataset", default="data/golden/seed.jsonl")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the response cache.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics.")
    args = parser.parse_args()

    run_config(
        args.config,
        args.dataset,
        use_cache=not args.no_cache,
        max_workers=args.max_workers,
        skip_ragas=args.skip_ragas,
    )


if __name__ == "__main__":
    main()
