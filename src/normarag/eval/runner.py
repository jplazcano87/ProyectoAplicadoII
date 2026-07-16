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
CONFIGS = ("baseline", "rag_naive", "rag_improved")


def build_generator(
    config_name: str,
    *,
    use_hybrid: bool | None = None,
    use_rerank: bool | None = None,
    use_citation_prompt: bool | None = None,
    top_k_rerank: int | None = None,
) -> Generator:
    """Construct the real (OpenAI/Qdrant/Cohere-backed) generator for a config.

    For rag_improved the three toggles and top_k_rerank override the settings
    defaults, so M5 ablations and the top-k sweep run from the CLI without new
    logic.
    """
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
    if config_name == "rag_improved":
        from normarag.pipeline.rag_improved import run_rag_improved
        from normarag.retrieval.rerank import CohereReranker
        from normarag.retrieval.retriever import Retriever

        uh = settings.use_hybrid if use_hybrid is None else use_hybrid
        ur = settings.use_rerank if use_rerank is None else use_rerank
        uc = settings.use_citation_prompt if use_citation_prompt is None else use_citation_prompt
        retriever = Retriever()
        reranker = CohereReranker() if ur else None
        return lambda q: run_rag_improved(
            q,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            use_hybrid=uh,
            use_rerank=ur,
            use_citation_prompt=uc,
            top_k_rerank=top_k_rerank,
        )
    raise ValueError(f"Unknown config '{config_name}'. Choices: {CONFIGS}")


def _cache_key(config_name: str, question: str, variant: dict | None = None) -> str:
    payload = json.dumps(
        {
            "config": config_name,
            "generator_model": settings.generator_model,
            "top_k": settings.top_k_dense,
            "variant": variant or {},
            "question": question,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_one(
    config_name: str,
    question: str,
    generate: Generator,
    read_cache: bool,
    variant: dict | None = None,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Generate one answer. `read_cache` gates *reading* the cache; the result is
    always written back, so `--no-cache` refreshes stale entries (e.g. after a
    prompt/parser change) instead of leaving them behind.

    The cache is namespaced by config_name (not label) and keyed by variant, so
    repeated runs of the same config (the M5 3x variance runs) share the cached
    generation while writing to distinct result files."""
    cache_path = cache_dir / config_name / f"{_cache_key(config_name, question, variant)}.json"
    if read_cache and cache_path.exists():
        cached: dict = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached
    output = generate(question)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return output


def _trim_context(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "chapter_code": chunk.get("chapter_code"),
        "numeral": chunk.get("numeral"),
        "score": chunk.get("score"),
        "rrf_score": chunk.get("rrf_score"),
        "rerank_score": chunk.get("rerank_score"),
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
    label: str | None = None,
    variant: dict | None = None,
    use_cache: bool = True,
    max_workers: int = 4,
    skip_ragas: bool = False,
    chunks_path: Path = CHUNKS_PATH,
    results_dir: Path = RESULTS_DIR,
    cache_dir: Path = CACHE_DIR,
) -> list[dict]:
    """Run a config (optionally an ablation `variant`) over the golden set.

    `label` names the output file / cache namespace (defaults to config_name),
    so ablations of rag_improved can coexist as results/<label>.jsonl.
    `use_cache=False` regenerates and refreshes the cache (does not just bypass).
    """
    label = label or config_name
    examples = load_golden(dataset_path)
    known = known_citation_keys(load_chunks(chunks_path)) if Path(chunks_path).exists() else set()
    generate = generate or build_generator(config_name, **(variant or {}))

    print(
        f"[{label}] config={config_name} variant={variant or '{}'} | "
        f"{len(examples)} questions | cache={'on' if use_cache else 'off'} | "
        f"ragas={'off' if skip_ragas else 'on'} | workers={max_workers}",
        flush=True,
    )
    print(f"[{label}] generating answers...", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outputs = list(
            pool.map(
                lambda ex: _generate_one(
                    config_name, ex.question, generate, use_cache, variant, cache_dir
                ),
                examples,
            )
        )
    print(f"[{label}] generation done; scoring citation metrics...", flush=True)

    rows = [_build_row(label, ex, out, known) for ex, out in zip(examples, outputs)]
    if not skip_ragas:
        print(f"[{label}] starting RAGAS (this is the slow step)...", flush=True)
        _attach_ragas(rows, examples)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{label}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_cost = sum(r["cost_usd"] or 0.0 for r in rows)
    print(f"[{label}] DONE: {len(rows)} questions -> {out_path} (approx cost ${total_cost:.4f})\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an eval config over the golden set.")
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--dataset", default="data/golden/seed.jsonl")
    parser.add_argument("--label", default=None, help="Output/cache name (default: config).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the response cache.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics.")
    # Ablation toggles for rag_improved (unset -> settings default).
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--citation-prompt", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    variant: dict = {}
    if args.hybrid is not None:
        variant["use_hybrid"] = args.hybrid
    if args.rerank is not None:
        variant["use_rerank"] = args.rerank
    if args.citation_prompt is not None:
        variant["use_citation_prompt"] = args.citation_prompt

    run_config(
        args.config,
        args.dataset,
        label=args.label,
        variant=variant or None,
        use_cache=not args.no_cache,
        max_workers=args.max_workers,
        skip_ragas=args.skip_ragas,
    )


if __name__ == "__main__":
    main()
