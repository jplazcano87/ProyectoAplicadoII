"""Validate a golden dataset JSONL file, reporting errors line by line.

Checks, per line:
  - valid JSON and required fields / types (pydantic schema)
  - unique ids across the file
  - answerable=false  -> references must be empty
  - answerable=true   -> references non-empty AND every (chapter, numeral)
    actually exists in data/processed/chunks.jsonl (catches typos in refs)

Usage:
  python scripts/validate_golden.py [path/to/dataset.jsonl]

Exits non-zero if any line is invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script lives outside the package; put the repo root on sys.path so the
# root `config` package and `normarag` resolve when run as `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from normarag.eval.golden import (  # noqa: E402
    GoldenExample,
    known_reference_keys,
    missing_references,
)
from normarag.index.store import CHUNKS_PATH, load_chunks  # noqa: E402

DEFAULT_DATASET = Path("data/golden/seed.jsonl")


def validate_file(dataset: Path, chunks_path: Path = CHUNKS_PATH) -> int:
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}")
        return 1

    known: set[tuple[str, str]] = set()
    if chunks_path.exists():
        known = known_reference_keys(load_chunks(chunks_path))
    else:
        print(
            f"WARNING: {chunks_path} not found; skipping reference-existence checks "
            "(run `make ingest` to enable them).\n"
        )

    seen_ids: dict[str, int] = {}
    errors = 0
    total = 0

    for lineno, raw in enumerate(dataset.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        total += 1
        line_errors: list[str] = []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[line {lineno}] INVALID JSON: {exc}")
            errors += 1
            continue

        # Unique ids — checked from the raw id so duplicates are caught even if
        # one of the copies also fails schema validation.
        ident = data.get("id")
        if isinstance(ident, str):
            if ident in seen_ids:
                line_errors.append(f"duplicate id (also on line {seen_ids[ident]})")
            else:
                seen_ids[ident] = lineno

        try:
            example = GoldenExample.model_validate(data)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "(root)"
                line_errors.append(f"{loc}: {err['msg']}")
            print(f"[line {lineno}] id={data.get('id', '?')} INVALID:")
            for e in line_errors:
                print(f"    - {e}")
            errors += 1
            continue

        # References must exist in the corpus (only when chunks are available).
        if known and example.answerable:
            missing = missing_references(example, known)
            if missing:
                pretty = ", ".join(f"{ch}::{num}" for ch, num in missing)
                line_errors.append(f"references not found in corpus: {pretty}")

        if line_errors:
            print(f"[line {lineno}] id={example.id} INVALID:")
            for e in line_errors:
                print(f"    - {e}")
            errors += 1
        else:
            print(f"[line {lineno}] id={example.id} OK")

    print()
    if errors:
        print(f"FAILED: {errors} invalid / {total} examples in {dataset}")
    else:
        print(f"OK: all {total} examples valid in {dataset}")
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a golden dataset JSONL file.")
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    args = parser.parse_args()
    sys.exit(validate_file(Path(args.dataset)))


if __name__ == "__main__":
    main()
