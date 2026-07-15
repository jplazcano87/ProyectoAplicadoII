"""Run the full M5 experiment suite over the golden set, then build the report.

Everything reproducible with one command (`make experiments`). Nothing here
tunes the system — it only measures it:

  1. Main A/B/C, run N times (default 3) -> results/main/  (mean ± std)
  2. Ablations of C (one flag at a time) -> results/ablations/
  3. Top-k sweep of C (top_k_rerank in {3,6,10,20}) -> results/topk/
  4. Consolidated report -> results/comparativa.md (+ topk_curve.png)

Generation is cached by (config, variant, question), so the 3 variance runs
reuse the same answers and only re-run the RAGAS judge; ablations/top-k reuse
generation where the variant matches. Use --skip-ragas / --dataset / --runs to
control cost while validating the machinery.

Usage:
  python scripts/run_experiments.py --dataset data/golden/golden_dataset.jsonl
  python scripts/run_experiments.py --dataset data/golden/seed.jsonl --runs 1 --skip-ragas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

from normarag.eval.experiment_report import generate_experiment_report  # noqa: E402
from normarag.eval.runner import run_config  # noqa: E402

RESULTS_DIR = Path("results")
TOPK_VALUES = [3, 6, 10, 20]


def _golden_meta(dataset: Path) -> dict:
    raw = dataset.read_bytes()
    count = len([1 for line in raw.splitlines() if line.strip()])
    return {"path": str(dataset), "count": count, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_meta(dataset: Path, n_runs: int) -> None:
    meta = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_runs": n_runs,
        "golden": _golden_meta(dataset),
        "models": {
            "generator": settings.generator_model,
            "judge": settings.judge_model,
            "embedding": settings.embedding_model,
            "rerank": settings.rerank_model,
        },
        "top_k_dense": settings.top_k_dense,
        "top_k_sparse": settings.top_k_sparse,
        "top_k_rerank": settings.top_k_rerank,
        "rrf_k": settings.rrf_k,
    }
    (RESULTS_DIR / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_experiments(dataset: str, runs: int, skip_ragas: bool, workers: int) -> None:
    ds = Path(dataset)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_meta(ds, runs)

    common = {"use_cache": True, "skip_ragas": skip_ragas}

    # 1. Main A/B/C, N runs each (generation shared via cache; RAGAS re-runs).
    print(f"\n===== 1. MAIN A/B/C ({runs} runs) =====", flush=True)
    for config in ("baseline", "rag_naive", "rag_improved"):
        w = 1 if config == "rag_improved" else workers
        for i in range(1, runs + 1):
            run_config(
                config,
                ds,
                label=f"{config}__run{i}",
                results_dir=RESULTS_DIR / "main",
                max_workers=w,
                **common,
            )

    # 2. Ablations of C — one variable changed vs the full config.
    print("\n===== 2. ABLATIONS OF C =====", flush=True)
    ablations = {
        "C_full": {},
        "C_norerank": {"use_rerank": False},
        "C_dense": {"use_hybrid": False},
        "C_naiveprompt": {"use_citation_prompt": False},
    }
    for label, variant in ablations.items():
        run_config(
            "rag_improved",
            ds,
            label=label,
            variant=variant or None,
            results_dir=RESULTS_DIR / "ablations",
            max_workers=1,
            **common,
        )

    # 3. Top-k sweep (top_k_rerank), everything else fixed at the C defaults.
    print("\n===== 3. TOP-K SWEEP =====", flush=True)
    for k in TOPK_VALUES:
        run_config(
            "rag_improved",
            ds,
            label=f"C_k{k}",
            variant={"top_k_rerank": k},
            results_dir=RESULTS_DIR / "topk",
            max_workers=1,
            **common,
        )

    # 4. Consolidated report.
    print("\n===== 4. REPORT =====", flush=True)
    generate_experiment_report(RESULTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M5 experiment suite.")
    parser.add_argument("--dataset", default="data/golden/golden_dataset.jsonl")
    parser.add_argument("--runs", type=int, default=3, help="Variance runs for the A/B/C table.")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    run_experiments(args.dataset, args.runs, args.skip_ragas, args.max_workers)


if __name__ == "__main__":
    main()
