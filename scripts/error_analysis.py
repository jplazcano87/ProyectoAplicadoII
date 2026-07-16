"""Classify config-C failures into categories and break them down by
difficulty / tags -> results/error_analysis.md.

Categories (one primary per failing question, in priority order):
  gold_invalid     - the gold reference itself does not exist in the corpus
                     (a golden-dataset data issue, not a system error)
  abstention_fail  - should have abstained (answerable=false) and didn't, or
                     should have answered (answerable=true) and abstained
  citation_invalid - cited a (chapter, numeral) that does not exist in the corpus
  retrieval_miss   - the gold numeral never entered the hybrid candidate pool
  ranking_miss     - it was in the pool but fell outside the reranked top-k
  hallucination    - the gold chunk was in the final context, yet the answer's
                     citation is wrong (or RAGAS faithfulness is low)

The retrieval/ranking split re-runs the deterministic hybrid retrieval for each
failing question (query embeddings are cached, so this is free). Reads the C
results written by scripts/run_experiments.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from normarag.eval.metrics import citation_key, known_citation_keys  # noqa: E402
from normarag.index.store import CHUNKS_PATH, load_chunks  # noqa: E402

FAITHFULNESS_FLOOR = 0.5
CATEGORIES = [
    "gold_invalid",
    "abstention_fail",
    "citation_invalid",
    "retrieval_miss",
    "ranking_miss",
    "hallucination",
]


def _refs(items: list[dict]) -> set[tuple[str, str]]:
    return {citation_key(i) for i in items}


def _ctx_refs(contexts: list[dict]) -> set[tuple[str, str]]:
    return {(str(c.get("chapter_code", "")), str(c.get("numeral", ""))) for c in contexts}


def _make_retriever():
    """Hybrid retriever for the candidate pool, or None if unavailable."""
    try:
        from normarag.retrieval.retriever import Retriever

        return Retriever()
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: retriever unavailable ({exc}); retrieval/ranking split disabled.")
        return None


def classify(row: dict, known: set[tuple[str, str]], retriever) -> tuple[str, str] | None:
    """Return (category, explanation) for a failing row, or None if it succeeded."""
    answerable = row["answerable"]
    abstained = bool(row["abstained"])
    gold = _refs(row.get("references", []))

    # Non-answerable: the only failure mode is not abstaining.
    if not answerable:
        if not abstained:
            return ("abstention_fail", "answerable=false pero el sistema respondió (no se abstuvo)")
        return None

    # Answerable from here on.
    if abstained:
        return ("abstention_fail", "answerable=true pero el sistema se abstuvo")

    # Gold reference itself missing from the corpus -> data issue, not the system.
    if gold and not (gold & known):
        bad = ", ".join(f"{c}::{n}" for c, n in sorted(gold - known))
        return ("gold_invalid", f"la referencia del gold no existe en el corpus: {bad}")

    exact_ok = row.get("citation_exact_f1", 0.0) == 1.0
    if exact_ok:
        # Citation matches the gold; only a low-faithfulness answer is a failure.
        faith = row.get("faithfulness")
        if isinstance(faith, (int, float)) and faith < FAITHFULNESS_FLOOR:
            return ("hallucination", f"cita correcta pero faithfulness baja ({faith:.2f})")
        return None

    # Citation is wrong. Diagnose where it broke.
    if row.get("citation_validity", 1.0) < 1.0:
        cited = ", ".join(f"{c}::{n}" for c, n in sorted(_refs(row.get("citations", []))))
        return ("citation_invalid", f"citó algo inexistente en el corpus: {cited}")

    ctx = _ctx_refs(row.get("contexts", []))
    if gold & ctx:
        return ("hallucination", "el chunk correcto estaba en el contexto pero la cita no coincide")

    # Gold not in final context: was it retrieved at all?
    if retriever is not None:
        pool = _ctx_refs(retriever.retrieve(row["question"], use_hybrid=True))
        if not (gold & pool):
            return ("retrieval_miss", "el numeral correcto nunca entró en los candidatos")
        return ("ranking_miss", "entró en los candidatos pero quedó fuera del top-k tras el rerank")
    return ("ranking_miss", "fuera del top-k final (pool no verificable: retriever no disponible)")


def analyze(rows: list[dict], known: set[tuple[str, str]]) -> dict:
    retriever = _make_retriever()
    findings: list[dict] = []
    for row in rows:
        result = classify(row, known, retriever)
        if result is None:
            continue
        cat, why = result
        findings.append({"row": row, "category": cat, "why": why})
    return {"findings": findings, "total": len(rows)}


def _md_examples(findings: list[dict], category: str, n: int = 3) -> list[str]:
    lines: list[str] = []
    for f in [x for x in findings if x["category"] == category][:n]:
        row = f["row"]
        lines.append(f"- **{row['id']}** ({row['difficulty']}, tags={row.get('tags', [])})")
        lines.append(f"  - Pregunta: {row['question']}")
        lines.append(f"  - Gold: {row.get('references', [])} · Citó: {row.get('citations', [])}")
        lines.append(f"  - Diagnóstico: {f['why']}")
    return lines


def _breakdown(findings: list[dict], key: str) -> Counter:
    counter: Counter = Counter()
    for f in findings:
        value = f["row"].get(key)
        if isinstance(value, list):
            for v in value:
                counter[v] += 1
        else:
            counter[value] += 1
    return counter


def build_markdown(analysis: dict) -> str:
    findings = analysis["findings"]
    total = analysis["total"]
    counts = Counter(f["category"] for f in findings)
    n_fail = len(findings)

    out = ["# NormaRAG — Análisis de errores (config C)", ""]
    out.append(f"Sobre **{total}** preguntas del golden set, **{n_fail}** presentan algún fallo.")
    out.append("")
    out.append("## Conteo por categoría")
    out.append("")
    out.append("| Categoría | Conteo |")
    out.append("|---|---|")
    for cat in CATEGORIES:
        out.append(f"| {cat} | {counts.get(cat, 0)} |")
    out.append("")

    out.append("## Desglose por dificultad")
    out.append("")
    diff = _breakdown(findings, "difficulty")
    out.append("| Dificultad | Fallos |")
    out.append("|---|---|")
    for d in ("easy", "medium", "hard"):
        out.append(f"| {d} | {diff.get(d, 0)} |")
    out.append("")

    out.append("## Desglose por tag (top 10)")
    out.append("")
    tags = _breakdown(findings, "tags")
    out.append("| Tag | Fallos |")
    out.append("|---|---|")
    for tag, c in tags.most_common(10):
        out.append(f"| {tag} | {c} |")
    out.append("")

    out.append("## Ejemplos por categoría")
    out.append("")
    for cat in CATEGORIES:
        if counts.get(cat):
            out.append(f"### {cat} ({counts[cat]})")
            out += _md_examples(findings, cat)
            out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify config-C failures.")
    parser.add_argument(
        "--results",
        default="results/main/rag_improved__run1.jsonl",
        help="C results JSONL (falls back to results/ablations/C_full.jsonl).",
    )
    parser.add_argument("--out", default="results/error_analysis.md")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        path = Path("results/ablations/C_full.jsonl")
    if not path.exists():
        print(f"No C results found ({args.results}). Run the experiments first.")
        sys.exit(1)

    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    known = known_citation_keys(load_chunks(CHUNKS_PATH)) if CHUNKS_PATH.exists() else set()
    analysis = analyze(rows, known)
    Path(args.out).write_text(build_markdown(analysis), encoding="utf-8")
    print(f"Wrote {args.out} ({len(analysis['findings'])} failures classified from {path})")


if __name__ == "__main__":
    main()
