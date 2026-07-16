"""Re-judge existing results with RAGAS, resumable by batches and per-row cached.

Reuses the generated answers/contexts already in results/**/<label>.jsonl and
recomputes ONLY the RAGAS metrics with the configured judge (set
JUDGE_MODEL=gpt-4o to use a stronger, independent judge). Each row's RAGAS score
is cached by (judge_model, question, answer, contexts, reference), so a run
interrupted by a rate limit resumes where it left off — no generation is re-paid.

Usage:
  JUDGE_MODEL=gpt-4o python scripts/rejudge_ragas.py results/main/rag_naive__run1.jsonl
  JUDGE_MODEL=gpt-4o python scripts/rejudge_ragas.py results/main/*.jsonl --batch 10
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

from normarag.eval.ragas_metrics import RAGAS_COLUMNS, evaluate_ragas  # noqa: E402

CACHE_DIR = Path(".cache/ragas")


def _row_key(row: dict) -> str:
    payload = json.dumps(
        {
            "judge": settings.judge_model,
            "q": row.get("question"),
            "a": row.get("answer"),
            "ctx": [c.get("text") for c in row.get("contexts", [])],
            "ref": row.get("gold_answer"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / settings.judge_model / f"{key}.json"


def _ragas_input(row: dict) -> dict:
    return {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "contexts": [c["text"] for c in row.get("contexts", []) if c.get("text")],
        "ground_truth": row.get("gold_answer", ""),
    }


def rejudge_file(path: Path, batch_size: int) -> None:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

    # Which rows still need judging (not in the per-row cache)?
    todo: list[int] = []
    for i, row in enumerate(rows):
        cache = _cache_path(_row_key(row))
        if cache.exists():
            row.update(json.loads(cache.read_text()))
        else:
            todo.append(i)

    print(
        f"[{path.name}] {len(rows)} rows · {len(rows) - len(todo)} cached · "
        f"{len(todo)} to judge (judge={settings.judge_model}, batch={batch_size})",
        flush=True,
    )

    for start in range(0, len(todo), batch_size):
        idx_batch = todo[start : start + batch_size]
        scores = evaluate_ragas([_ragas_input(rows[i]) for i in idx_batch])
        if scores is None:
            print(f"  batch {start // batch_size + 1}: RAGAS unavailable, stopping.", flush=True)
            break
        for i, score in zip(idx_batch, scores, strict=True):
            keep = {c: (score or {}).get(c) for c in RAGAS_COLUMNS}
            rows[i].update(keep)
            cache = _cache_path(_row_key(rows[i]))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(keep), encoding="utf-8")
        print(f"  batch {start // batch_size + 1}: judged {len(idx_batch)} rows.", flush=True)

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"[{path.name}] done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge results with RAGAS (resumable).")
    parser.add_argument("paths", nargs="+", help="Result JSONL files (globs allowed).")
    parser.add_argument("--batch", type=int, default=10, help="Rows per batch (resume unit).")
    args = parser.parse_args()

    files = sorted({Path(p) for pattern in args.paths for p in glob.glob(pattern)})
    if not files:
        print("No matching result files.")
        sys.exit(1)
    for f in files:
        rejudge_file(f, args.batch)


if __name__ == "__main__":
    main()
