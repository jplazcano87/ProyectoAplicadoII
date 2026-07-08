"""Structural chunking of RAN chapters: chapter -> título (roman) -> numeral.

The minimum chunk is a *numeral* (e.g. "2.1"), never a fixed-size window. Each
chunk carries its chapter/título context in metadata. Numerals longer than
MAX_TOKENS are split into overlapping sub-chunks (whole words, no mid-word cuts)
that share metadata and differ only in the trailing chunk_id index.

Detection is regex-based over the extracted text. RAN structure quirks handled:
  - Repeated page boilerplate ("RECOPILACION...", "Capítulo X-Y", "Hoja N").
  - Single-level numerals require a trailing period ("2.") so rating-scale list
    items like "2 CUMPLIMIENTO MATERIAL" are not mistaken for numerals.
  - Text after a numeral must start uppercase or end the line, so wrapped
    continuations like "66 quinquies, cuando..." are not mistaken for numerals.
  - Roman "título" headers require a mostly-uppercase heading (or bare on a
    line), so inline mentions / list items are not mistaken for títulos.

Output: data/processed/chunks.jsonl (one JSON object per line, schema below).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

from normarag.corpus.parse import ParsedChapter, parse_corpus

PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

MAX_TOKENS = 800
OVERLAP_RATIO = 0.15

_ENCODER = tiktoken.get_encoding("cl100k_base")

# --- Structural regexes -----------------------------------------------------

# Page boilerplate repeated on every page.
_BOILERPLATE_RE = re.compile(
    r"^\s*(RECOPILACION ACTUALIZADA DE NORMAS|Cap[ií]tulo\s+\d+-\d+|Hoja\s+\d+"
    r"|CAP[IÍ]TULO\s+\d+-\d+)\s*$",
    re.IGNORECASE,
)

# Roman-numeral título header: bare ("I.") or followed by a mostly-uppercase heading.
_TITULO_RE = re.compile(r"^([IVXLC]+)\.(?:\s+(?P<heading>.+?))?\s*$")

# Multi-level numeral ("3.1", "3.1.1"): internal dot present, unambiguous.
_NUMERAL_MULTI_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?(?:\s+(?P<rest>\S.*)|\s*)$")

# Single-level numeral ("2."): trailing period REQUIRED to avoid list items.
_NUMERAL_SINGLE_RE = re.compile(r"^(\d+)\.(?:\s+(?P<rest>\S.*)|\s*)$")


@dataclass
class Chunk:
    chunk_id: str
    chapter_code: str
    chapter_title: str
    title_section: str
    numeral: str
    text: str
    page_start: int
    page_end: int
    n_tokens: int
    source_url: str


def n_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _is_mostly_uppercase(text: str, threshold: float = 0.6) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= threshold


def _match_titulo(line: str) -> str | None:
    """Return the roman numeral if the line is a título header, else None."""
    m = _TITULO_RE.match(line.strip())
    if not m:
        return None
    heading = m.group("heading")
    if heading is None:  # bare "I." on its own line
        return m.group(1)
    if _is_mostly_uppercase(heading):
        return m.group(1)
    return None


def _match_numeral(line: str) -> str | None:
    """Return the numeral (e.g. '2.1') if the line starts one, else None."""
    s = line.strip()
    for regex in (_NUMERAL_MULTI_RE, _NUMERAL_SINGLE_RE):
        m = regex.match(s)
        if not m:
            continue
        rest = m.groupdict().get("rest")
        # Accept if numeral sits alone on the line, or the heading text that
        # follows starts with an uppercase/opening char (a real heading).
        if rest is None or not rest[:1].islower():
            return m.group(1)
    return None


@dataclass
class _Section:
    """A run of text under one (título, numeral) with its page span."""

    title_section: str
    numeral: str
    lines: list[str]
    page_start: int
    page_end: int


def _split_into_sections(chapter: ParsedChapter) -> list[_Section]:
    """Walk the chapter's lines, cutting a new section at each título/numeral."""
    sections: list[_Section] = []
    current = _Section(title_section="", numeral="0", lines=[], page_start=1, page_end=1)
    started = False  # becomes True once we hit real content

    for page in chapter.pages:
        for raw in page.text.splitlines():
            line = raw.strip()
            if not line or _BOILERPLATE_RE.match(line):
                continue

            titulo = _match_titulo(line)
            if titulo is not None:
                if started:
                    sections.append(current)
                # New título resets the numeral to preamble "0".
                current = _Section(
                    title_section=titulo,
                    numeral="0",
                    lines=[],
                    page_start=page.number,
                    page_end=page.number,
                )
                started = True
                continue

            numeral = _match_numeral(line)
            if numeral is not None:
                if started:
                    sections.append(current)
                current = _Section(
                    title_section=current.title_section,
                    numeral=numeral,
                    lines=[line],
                    page_start=page.number,
                    page_end=page.number,
                )
                started = True
                continue

            current.lines.append(line)
            current.page_end = page.number
            started = True

    if started or current.lines:
        sections.append(current)

    # Drop empty preamble sections (no meaningful text).
    return [s for s in sections if _clean_text(s.lines).strip()]


# A line-ending hyphen after a letter marks a word split across lines by the
# PDF layout (e.g. "socie-" / "dad"); it must be re-joined without a space.
_LINE_END_HYPHEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]-$")


def _clean_text(lines: list[str]) -> str:
    """Join wrapped lines into clean text (de-hyphenated, collapsed spaces)."""
    out = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _LINE_END_HYPHEN_RE.search(out):
            out = out[:-1] + s  # drop hyphen, splice the two word halves
        elif out:
            out = f"{out} {s}"
        else:
            out = s
    return re.sub(r"\s+", " ", out).strip()


def _split_long_text(text: str) -> list[str]:
    """Split text over MAX_TOKENS into overlapping windows on word boundaries."""
    if n_tokens(text) <= MAX_TOKENS:
        return [text]

    words = text.split()
    overlap = int(MAX_TOKENS * OVERLAP_RATIO)
    windows: list[str] = []
    start = 0
    while start < len(words):
        window: list[str] = []
        i = start
        while i < len(words) and n_tokens(" ".join([*window, words[i]])) <= MAX_TOKENS:
            window.append(words[i])
            i += 1
        if not window:  # single word longer than budget; force-take it
            window = [words[start]]
            i = start + 1
        windows.append(" ".join(window))
        if i >= len(words):
            break
        # Step back ~overlap tokens worth of words for context continuity.
        back = 0
        j = len(window) - 1
        while j >= 0 and n_tokens(" ".join(window[j:])) <= overlap:
            back += 1
            j -= 1
        start = i - back if back else i
    return windows


def chunk_chapter(chapter: ParsedChapter) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _split_into_sections(chapter):
        text = _clean_text(section.lines)
        for idx, piece in enumerate(_split_long_text(text)):
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{chapter.chapter_code}::{section.title_section}"
                        f"::{section.numeral}::{idx}"
                    ),
                    chapter_code=chapter.chapter_code,
                    chapter_title=chapter.chapter_title,
                    title_section=section.title_section,
                    numeral=section.numeral,
                    text=piece,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    n_tokens=n_tokens(piece),
                    source_url=chapter.source_url,
                )
            )
    return chunks


def chunk_corpus() -> list[Chunk]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []
    for chapter in parse_corpus():
        chapter_chunks = chunk_chapter(chapter)
        all_chunks.extend(chapter_chunks)
        print(f"  chunked {chapter.chapter_code}: {len(chapter_chunks)} chunks")

    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks -> {CHUNKS_PATH}")
    return all_chunks


if __name__ == "__main__":
    chunk_corpus()
