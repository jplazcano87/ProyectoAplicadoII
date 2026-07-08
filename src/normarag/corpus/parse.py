"""Extracts per-page text from RAN PDFs using PyMuPDF (offline, from manifest).

Reads the list of downloaded PDFs from data/raw/manifest.json — never touches
the network — and returns, for each chapter, its pages of extracted text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"


@dataclass
class Page:
    number: int  # 1-indexed page number
    text: str


@dataclass
class ParsedChapter:
    chapter_code: str
    chapter_title: str
    source_url: str
    pages: list[Page]


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run `python -m normarag.corpus.download` first."
        )
    return cast(list[dict], json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def parse_pdf(path: Path) -> list[Page]:
    """Extract text page by page from a single PDF."""
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append(Page(number=i, text=page.get_text("text")))
    return pages


def parse_corpus() -> list[ParsedChapter]:
    """Parse every PDF listed in the manifest."""
    manifest = _load_manifest()
    parsed: list[ParsedChapter] = []
    for entry in manifest:
        path = Path(entry["path"])
        if not path.exists():
            print(f"  ! missing PDF {path}; skipping")
            continue
        pages = parse_pdf(path)
        parsed.append(
            ParsedChapter(
                chapter_code=entry["chapter_code"],
                chapter_title=entry["title"],
                source_url=entry["pdf_url"],
                pages=pages,
            )
        )
        print(f"  parsed {entry['chapter_code']}: {len(pages)} pages")
    return parsed


if __name__ == "__main__":
    parse_corpus()
