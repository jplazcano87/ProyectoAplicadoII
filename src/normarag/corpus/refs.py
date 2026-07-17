"""Emit the committed (chapter, numeral) reference keys from chunks.jsonl.

data/processed/corpus_refs.txt is a small, versioned list of every valid
`chapter::numeral` in the corpus. The golden validator falls back to it so CI
(which has no chunks.jsonl) can still check that golden references exist.
Regenerate it whenever the corpus is re-chunked (part of `make ingest`).
"""

from __future__ import annotations

from pathlib import Path

from normarag.index.store import CHUNKS_PATH, load_chunks

REFS_PATH = Path("data/processed/corpus_refs.txt")


def emit_refs(chunks_path: Path = CHUNKS_PATH, out: Path = REFS_PATH) -> int:
    chunks = load_chunks(chunks_path)
    keys = sorted({(c["chapter_code"], str(c["numeral"])) for c in chunks})
    out.write_text("".join(f"{ch}::{num}\n" for ch, num in keys), encoding="utf-8")
    print(f"Wrote {len(keys)} reference keys -> {out}")
    return len(keys)


if __name__ == "__main__":
    emit_refs()
