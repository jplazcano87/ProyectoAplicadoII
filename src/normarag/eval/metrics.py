"""Evaluation metrics.

Two families:
  1. Custom citation metrics (this module) — 100% programmatic, no LLM. This is
     the project's own, honest measure of citation quality.
  2. RAGAS metrics (eval.ragas_metrics) — faithfulness, answer_relevancy,
     context_precision, context_recall, judged by JUDGE_MODEL (distinct from the
     generator to avoid self-evaluation bias). Loaded lazily/optionally.

Citation shape everywhere: {"chapter": str, "numeral": str}. Reference shape
(from the gold): objects with .chapter / .numeral (or the same dict).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

CitationKey = tuple[str, str]


def _norm(value: object) -> str:
    return str(value if value is not None else "").strip()


def citation_key(citation: object) -> CitationKey:
    """Normalize a citation (dict or object) to a (chapter, numeral) tuple."""
    if isinstance(citation, dict):
        chapter = citation.get("chapter", citation.get("chapter_code"))
        numeral = citation.get("numeral")
    else:
        chapter = getattr(citation, "chapter", None)
        numeral = getattr(citation, "numeral", None)
    return (_norm(chapter), _norm(numeral))


def known_citation_keys(chunks: list[dict]) -> set[CitationKey]:
    """Set of (chapter_code, numeral) present in the indexed corpus."""
    return {(_norm(c.get("chapter_code")), _norm(c.get("numeral"))) for c in chunks}


def citation_validity(citations: Iterable[object], known: set[CitationKey]) -> float:
    """Fraction of cited (chapter, numeral) that actually exist in the corpus.

    Vacuously 1.0 when there are no citations (nothing invalid was emitted).
    """
    keys = [citation_key(c) for c in citations]
    if not keys:
        return 1.0
    valid = sum(1 for k in keys if k in known)
    return valid / len(keys)


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def _prf(predicted: set[CitationKey], gold: set[CitationKey]) -> PRF:
    if not predicted and not gold:
        # No citations expected and none produced: perfect agreement.
        return PRF(1.0, 1.0, 1.0)
    if not predicted or not gold:
        return PRF(0.0, 0.0, 0.0)
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PRF(precision, recall, f1)


def citation_exact_match(citations: Iterable[object], references: Iterable[object]) -> PRF:
    """Precision/recall/F1 at the (chapter, numeral) level."""
    predicted = {citation_key(c) for c in citations}
    gold = {citation_key(r) for r in references}
    return _prf(predicted, gold)


def citation_chapter_match(citations: Iterable[object], references: Iterable[object]) -> PRF:
    """Precision/recall/F1 at the chapter level only (laxer)."""
    predicted = {(chap, "") for chap, _ in (citation_key(c) for c in citations)}
    gold = {(chap, "") for chap, _ in (citation_key(r) for r in references)}
    return _prf(predicted, gold)


# --- Abstention -------------------------------------------------------------

ABSTAIN_TP = "abstain_tp"  # should abstain and did
ABSTAIN_FP = "abstain_fp"  # should NOT abstain but did
ABSTAIN_FN = "abstain_fn"  # should abstain but did not
ABSTAIN_TN = "abstain_tn"  # should NOT abstain and did not


def abstention_outcome(abstained: bool, answerable: bool) -> str:
    """Classify one example for abstention (positive class = abstaining)."""
    should_abstain = not answerable
    if should_abstain and abstained:
        return ABSTAIN_TP
    if should_abstain and not abstained:
        return ABSTAIN_FN
    if not should_abstain and abstained:
        return ABSTAIN_FP
    return ABSTAIN_TN


def abstention_scores(outcomes: Iterable[str]) -> PRF:
    """Aggregate abstention precision/recall/F1 from per-example outcomes."""
    counts = {ABSTAIN_TP: 0, ABSTAIN_FP: 0, ABSTAIN_FN: 0, ABSTAIN_TN: 0}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    tp, fp, fn = counts[ABSTAIN_TP], counts[ABSTAIN_FP], counts[ABSTAIN_FN]
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PRF(precision, recall, f1)


def per_example_metrics(
    output: dict,
    example: object,
    known: set[CitationKey],
) -> dict:
    """Compute all programmatic metrics for one (system output, gold) pair.

    `example` is a GoldenExample (has .references, .answerable). Returns a flat
    dict of scalar metrics plus the abstention outcome tag for aggregation.
    """
    citations = output.get("citations", [])
    references = getattr(example, "references", [])
    answerable = getattr(example, "answerable", True)
    abstained = bool(output.get("abstained", False))

    exact = citation_exact_match(citations, references)
    chapter = citation_chapter_match(citations, references)
    return {
        "citation_validity": citation_validity(citations, known),
        "citation_exact_precision": exact.precision,
        "citation_exact_recall": exact.recall,
        "citation_exact_f1": exact.f1,
        "citation_chapter_precision": chapter.precision,
        "citation_chapter_recall": chapter.recall,
        "citation_chapter_f1": chapter.f1,
        "abstention_outcome": abstention_outcome(abstained, answerable),
    }
