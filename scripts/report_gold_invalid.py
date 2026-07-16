"""Report golden references that don't exist in the corpus -> results/gold_to_fix.md.

For each golden example whose (chapter, numeral) is not present in
data/processed/chunks.jsonl, list the numerals that DO exist in that chapter
(and the closest ones), so a human can pick the correct reference. This script
does NOT edit the golden dataset — it only produces a to-fix checklist.
"""

from __future__ import annotations

import argparse
import sys
from difflib import get_close_matches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from normarag.eval.golden import load_golden  # noqa: E402
from normarag.index.store import CHUNKS_PATH, load_chunks  # noqa: E402


def _chapter_numerals(chunks: list[dict]) -> dict[str, list[str]]:
    """chapter_code -> sorted list of existing numerals (excluding preamble '0')."""
    by_chapter: dict[str, set[str]] = {}
    for c in chunks:
        by_chapter.setdefault(c["chapter_code"], set()).add(str(c["numeral"]))
    return {ch: sorted(nums, key=_numeral_key) for ch, nums in by_chapter.items()}


def _numeral_key(n: str) -> tuple:
    # Sort "2", "2.1", "2.10" naturally; non-numeric parts sort last.
    parts = []
    for p in n.split("."):
        parts.append((0, int(p)) if p.isdigit() else (1, p))
    return tuple(parts)


def build_report(golden_path: Path, chunks_path: Path) -> tuple[str, int]:
    chunks = load_chunks(chunks_path)
    known = {(c["chapter_code"], str(c["numeral"])) for c in chunks}
    chapter_numerals = _chapter_numerals(chunks)
    examples = load_golden(golden_path)

    rows: list[str] = []
    n_invalid = 0
    for ex in examples:
        bad = [r for r in ex.references if (r.chapter, r.numeral) not in known]
        if not bad:
            continue
        n_invalid += 1
        rows.append(f"### {ex.id} — {ex.question}")
        rows.append("")
        for ref in bad:
            existing = chapter_numerals.get(ref.chapter)
            if existing is None:
                suggestion = (
                    f"⚠️ el capítulo `{ref.chapter}` no existe en el corpus "
                    "(¿no se indexó, o el código es incorrecto?)"
                )
            elif existing == ["0"]:
                suggestion = (
                    f"⚠️ el capítulo `{ref.chapter}` solo tiene el preámbulo (`0`) indexado; "
                    "el chunker de M1 no detectó numerales aquí. Usa `0` o revisa el parsing "
                    "de ese capítulo."
                )
            else:
                close = get_close_matches(ref.numeral, existing, n=5, cutoff=0.0)
                suggestion = (
                    f"numerales existentes en `{ref.chapter}` (más parecidos primero): "
                    + ", ".join(f"`{n}`" for n in close)
                )
            rows.append(f"- Referencia inválida: `{ref.chapter}::{ref.numeral}`")
            rows.append(f"  - {suggestion}")
            if existing:
                rows.append(
                    f"  - todos los numerales de `{ref.chapter}`: "
                    + ", ".join(f"`{n}`" for n in existing)
                )
        rows.append(f"- Respuesta de referencia (gold): {ex.answer}")
        rows.append("")

    header = [
        "# Golden a corregir — referencias inexistentes en el corpus",
        "",
        f"**{n_invalid}** preguntas del golden citan `(capítulo, numeral)` que NO existen en "
        f"`{chunks_path}`. Elige el numeral correcto de la lista sugerida y corrige "
        f"`{golden_path}` a mano (esto NO lo hace el script).",
        "",
        "> Nota: el numeral `0` es el preámbulo del capítulo (no se sugiere como referencia).",
        "",
    ]
    if n_invalid == 0:
        header.append("✅ No hay referencias inválidas. Nada que corregir.")
    return "\n".join(header + rows) + "\n", n_invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Report invalid golden references.")
    parser.add_argument("--golden", default="data/golden/golden_dataset.jsonl")
    parser.add_argument("--out", default="results/gold_to_fix.md")
    args = parser.parse_args()

    md, n = build_report(Path(args.golden), CHUNKS_PATH)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out} ({n} invalid references)")


if __name__ == "__main__":
    main()
