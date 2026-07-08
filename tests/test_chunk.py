"""Tests for structural chunking of RAN chapters.

Core tests run on synthetic fixtures that reproduce the RAN structure quirks,
so they pass in CI without the (gitignored) downloaded PDFs. A final test runs
against real sample chapters when present and is skipped otherwise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from normarag.corpus.chunk import (
    _clean_text,
    _match_numeral,
    _match_titulo,
    chunk_chapter,
)
from normarag.corpus.parse import Page, ParsedChapter

# A surviving PDF line-break hyphenation artifact: "socie- dad".
HYPHEN_ARTIFACT_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]-\s+[a-záéíóúñ]")


def _chapter(text: str, code: str = "9-9", title: str = "Capítulo de prueba") -> ParsedChapter:
    """Build a one-page ParsedChapter from a text block."""
    return ParsedChapter(
        chapter_code=code,
        chapter_title=title,
        source_url="https://example.test/9-9.pdf",
        pages=[Page(number=1, text=text)],
    )


# --- Numeral detection ------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("1. Apertura de oficinas.", "1"),
        ("3.1. Gobierno corporativo y el rol del Directorio", "3.1"),
        ("3.1.1 Elementos de un buen gobierno corporativo", "3.1.1"),
        ("2.", "2"),
    ],
)
def test_match_numeral_accepts_real_numerals(line: str, expected: str) -> None:
    assert _match_numeral(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "2 CUMPLIMIENTO MATERIAL: La entidad cumple",  # rating-scale list item (no period)
        "66 quinquies, cuando correspondan.",  # wrapped continuation (lowercase follows)
        "10 de esta Recopilación.",  # wrapped continuation
        "Las presentes instrucciones sólo rigen.",  # plain prose
    ],
)
def test_match_numeral_rejects_false_positives(line: str) -> None:
    assert _match_numeral(line) is None


# --- Título (roman) detection ----------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("II. EVALUACIÓN DE LA GESTIÓN DE LOS BANCOS", "II"),
        ("I.", "I"),
        ("IV. IMPROCEDENCIA DE OTORGAR PODERES", "IV"),
    ],
)
def test_match_titulo_accepts_headings(line: str, expected: str) -> None:
    assert _match_titulo(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "II. Para el registro posterior.",  # not a title: lowercase heading
        "Título I de dicha norma, tanto al ordenante.",  # inline mention
    ],
)
def test_match_titulo_rejects_non_headings(line: str) -> None:
    assert _match_titulo(line) is None


# --- Cleaning / de-hyphenation ---------------------------------------------


def test_clean_text_dehyphenates_line_breaks() -> None:
    out = _clean_text(["la socie-", "dad anónima y la Superin-", "tendencia"])
    assert "sociedad" in out
    assert "Superintendencia" in out
    assert "- " not in out


def test_clean_text_collapses_whitespace() -> None:
    assert _clean_text(["  hola   ", "  mundo "]) == "hola mundo"


# --- End-to-end chunking on a synthetic chapter -----------------------------

SAMPLE = """RECOPILACION ACTUALIZADA DE NORMAS
Capítulo 9-9
Hoja 1
CAPÍTULO 9-9
CANALES DE ATENCIÓN

En virtud de lo establecido en la ley, las entidades deberán cumplir.

1. Apertura de oficinas.

1.1 Requisitos para la apertura.
De conformidad con el artículo 37, los bancos deberán informarlo.

2. Cierre de oficinas.
Los bancos que decidan cerrar deberán enviar la informa-
ción correspondiente.

II. EVALUACIÓN DE LA GESTIÓN

1. Categorías.
Conforme al Título V, esta Comisión clasifica a los bancos.
"""


def test_chunk_chapter_detects_numerals_and_titulos() -> None:
    chunks = chunk_chapter(_chapter(SAMPLE))
    keys = {(c.title_section, c.numeral) for c in chunks}
    assert ("", "1") in keys
    assert ("", "1.1") in keys
    assert ("", "2") in keys
    assert ("II", "1") in keys  # numeral 1 re-enumerated under título II


def test_chunk_schema_is_complete() -> None:
    for c in chunk_chapter(_chapter(SAMPLE)):
        assert c.chapter_code
        assert c.numeral
        assert c.text.strip()
        assert c.n_tokens > 0
        assert c.chunk_id.count("::") == 3


def test_chunk_no_boilerplate_or_midword_cuts() -> None:
    for c in chunk_chapter(_chapter(SAMPLE)):
        assert "RECOPILACION ACTUALIZADA" not in c.text
        assert "Hoja 1" not in c.text
        # de-hyphenation: the split "informa-/ción" must be rejoined
        assert not HYPHEN_ARTIFACT_RE.search(c.text)
    joined = " ".join(c.text for c in chunk_chapter(_chapter(SAMPLE)))
    assert "información" in joined


def test_long_numeral_is_subdivided_with_overlap() -> None:
    # Build a numeral whose body far exceeds MAX_TOKENS.
    body = " ".join(f"palabra{i}" for i in range(3000))
    chapter = _chapter(f"1. Seccion larga.\n{body}\n")
    chunks = chunk_chapter(chapter)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 800 for c in chunks)
    # Same metadata across pieces, only the trailing index differs.
    assert {c.numeral for c in chunks} == {"1"}
    indices = sorted(int(c.chunk_id.split("::")[-1]) for c in chunks)
    assert indices == list(range(len(chunks)))
    # Consecutive pieces overlap (shared words for context continuity).
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


# --- Real sample chapters (skipped when not downloaded) ---------------------

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


@pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="run `make ingest` first")
def test_real_chunks_are_coherent() -> None:
    rows = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    assert rows, "chunks.jsonl is empty"
    for r in rows:
        assert r["chapter_code"]
        assert r["numeral"]
        assert r["text"].strip()
        assert r["n_tokens"] > 0
        assert r["n_tokens"] <= 800
        assert r["chunk_id"].count("::") == 3
        # Line-break de-hyphenation is verified on synthetic fixtures; on real
        # text a "word- word" with a space is a legitimate mid-line hyphen
        # (ratings like "BBB-", "Sub-total", dash punctuation), not an artifact.
