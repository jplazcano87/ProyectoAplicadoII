"""Unit tests for the programmatic citation and abstention metrics."""

from __future__ import annotations

from normarag.eval.metrics import (
    abstention_outcome,
    abstention_scores,
    citation_chapter_match,
    citation_exact_match,
    citation_validity,
    known_citation_keys,
    per_example_metrics,
)

CHUNKS = [
    {"chapter_code": "1-8", "numeral": "1.2"},
    {"chapter_code": "1-8", "numeral": "1.1"},
    {"chapter_code": "1-13", "numeral": "1"},
]
KNOWN = known_citation_keys(CHUNKS)


class Example:
    """Minimal stand-in for a GoldenExample."""

    def __init__(self, references: list[dict], answerable: bool) -> None:
        self.references = [type("R", (), r)() for r in references]
        self.answerable = answerable


# --- citation_validity ------------------------------------------------------


def test_validity_all_exist() -> None:
    cites = [{"chapter": "1-8", "numeral": "1.2"}, {"chapter": "1-13", "numeral": "1"}]
    assert citation_validity(cites, KNOWN) == 1.0


def test_validity_nonexistent_reference() -> None:
    cites = [{"chapter": "1-8", "numeral": "99.99"}]
    assert citation_validity(cites, KNOWN) == 0.0


def test_validity_partial() -> None:
    cites = [{"chapter": "1-8", "numeral": "1.2"}, {"chapter": "1-8", "numeral": "nope"}]
    assert citation_validity(cites, KNOWN) == 0.5


def test_validity_empty_is_vacuously_one() -> None:
    assert citation_validity([], KNOWN) == 1.0


# --- exact and chapter match ------------------------------------------------


def test_exact_match_perfect() -> None:
    cites = [{"chapter": "1-8", "numeral": "1.2"}]
    refs = [{"chapter": "1-8", "numeral": "1.2"}]
    prf = citation_exact_match(cites, refs)
    assert (prf.precision, prf.recall, prf.f1) == (1.0, 1.0, 1.0)


def test_exact_match_partial_precision_recall() -> None:
    # Predicted two, one correct; gold has one -> P=0.5, R=1.0.
    cites = [{"chapter": "1-8", "numeral": "1.2"}, {"chapter": "1-8", "numeral": "1.1"}]
    refs = [{"chapter": "1-8", "numeral": "1.2"}]
    prf = citation_exact_match(cites, refs)
    assert prf.precision == 0.5
    assert prf.recall == 1.0
    assert round(prf.f1, 4) == round(2 / 3, 4)


def test_exact_match_wrong_numeral_but_right_chapter() -> None:
    cites = [{"chapter": "1-8", "numeral": "1.1"}]
    refs = [{"chapter": "1-8", "numeral": "1.2"}]
    # Exact match fails (different numeral)...
    assert citation_exact_match(cites, refs).f1 == 0.0
    # ...but the laxer chapter-level match succeeds.
    assert citation_chapter_match(cites, refs).f1 == 1.0


# --- abstention -------------------------------------------------------------


def test_abstention_outcomes() -> None:
    # Should abstain (not answerable) and did -> TP.
    assert abstention_outcome(abstained=True, answerable=False) == "abstain_tp"
    # Should abstain but did not -> FN.
    assert abstention_outcome(abstained=False, answerable=False) == "abstain_fn"
    # Should not abstain but did -> FP.
    assert abstention_outcome(abstained=True, answerable=True) == "abstain_fp"
    # Should not abstain and did not -> TN.
    assert abstention_outcome(abstained=False, answerable=True) == "abstain_tn"


def test_abstention_scores_aggregate() -> None:
    outcomes = ["abstain_tp", "abstain_tp", "abstain_fn", "abstain_fp", "abstain_tn"]
    prf = abstention_scores(outcomes)
    # TP=2, FP=1, FN=1 -> P=2/3, R=2/3.
    assert round(prf.precision, 4) == round(2 / 3, 4)
    assert round(prf.recall, 4) == round(2 / 3, 4)


def test_abstention_perfect_when_no_positives_expected() -> None:
    # All answerable, none abstained -> no TP/FP/FN; precision/recall = 0 by convention.
    prf = abstention_scores(["abstain_tn", "abstain_tn"])
    assert prf.precision == 0.0
    assert prf.recall == 0.0


# --- per_example_metrics integration ---------------------------------------


def test_per_example_metrics_answerable_correct() -> None:
    output = {"citations": [{"chapter": "1-8", "numeral": "1.2"}], "abstained": False}
    example = Example([{"chapter": "1-8", "numeral": "1.2"}], answerable=True)
    m = per_example_metrics(output, example, KNOWN)
    assert m["citation_validity"] == 1.0
    assert m["citation_exact_f1"] == 1.0
    assert m["abstention_outcome"] == "abstain_tn"


def test_per_example_metrics_correct_abstention() -> None:
    output = {"citations": [], "abstained": True}
    example = Example([], answerable=False)
    m = per_example_metrics(output, example, KNOWN)
    assert m["abstention_outcome"] == "abstain_tp"
    # No citations expected and none given -> exact match is perfect.
    assert m["citation_exact_f1"] == 1.0


def test_per_example_metrics_hallucinated_citation() -> None:
    output = {"citations": [{"chapter": "1-8", "numeral": "99.99"}], "abstained": False}
    example = Example([{"chapter": "1-8", "numeral": "1.2"}], answerable=True)
    m = per_example_metrics(output, example, KNOWN)
    assert m["citation_validity"] == 0.0  # cited numeral does not exist
    assert m["citation_exact_f1"] == 0.0  # and does not match the gold
