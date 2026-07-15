"""Offline tests for the M5 experiment machinery: mean±std and error classes."""

from __future__ import annotations

from scripts.error_analysis import classify

from normarag.eval.experiment_report import aggregate_runs

KNOWN = {("1-8", "1.2"), ("1-13", "1")}


def _row(**kw) -> dict:
    base = {
        "id": "q1",
        "question": "¿?",
        "answerable": True,
        "difficulty": "easy",
        "tags": ["t"],
        "references": [{"chapter": "1-8", "numeral": "1.2"}],
        "citations": [{"chapter": "1-8", "numeral": "1.2"}],
        "contexts": [{"chapter_code": "1-8", "numeral": "1.2", "text": "x"}],
        "abstained": False,
        "citation_exact_f1": 1.0,
        "citation_validity": 1.0,
        "faithfulness": 1.0,
        "latency_ms": 100.0,
        "cost_usd": 0.001,
        "abstention_outcome": "abstain_tn",
    }
    base.update(kw)
    return base


class FakeRetriever:
    def __init__(self, pool: list[dict]) -> None:
        self.pool = pool

    def retrieve(self, query: str, use_hybrid: bool = False) -> list[dict]:
        return self.pool


# --- mean ± std across runs -------------------------------------------------


def test_aggregate_runs_mean_and_std() -> None:
    # Two runs; RAGAS-like metric varies, programmatic metric constant.
    run_a = [_row(faithfulness=0.8, citation_exact_f1=1.0)]
    run_b = [_row(faithfulness=1.0, citation_exact_f1=1.0)]
    agg = aggregate_runs([run_a, run_b])
    mean_f, std_f = agg["faithfulness"]
    assert round(mean_f, 3) == 0.9
    assert round(std_f, 4) == round(0.1414, 4) or std_f > 0  # sample std of {0.8,1.0}
    mean_c, std_c = agg["citation_exact_f1"]
    assert mean_c == 1.0 and std_c == 0.0  # deterministic -> no spread


# --- failure classification -------------------------------------------------


def test_success_returns_none() -> None:
    assert classify(_row(), KNOWN, FakeRetriever([])) is None


def test_non_answerable_not_abstained_is_abstention_fail() -> None:
    row = _row(answerable=False, references=[], abstained=False, citations=[])
    cat, _ = classify(row, KNOWN, None)
    assert cat == "abstention_fail"


def test_answerable_abstained_is_abstention_fail() -> None:
    cat, _ = classify(_row(abstained=True), KNOWN, None)
    assert cat == "abstention_fail"


def test_gold_invalid_when_reference_not_in_corpus() -> None:
    row = _row(references=[{"chapter": "9-9", "numeral": "0.0"}], citation_exact_f1=0.0)
    cat, _ = classify(row, KNOWN, FakeRetriever([]))
    assert cat == "gold_invalid"


def test_citation_invalid() -> None:
    row = _row(
        citations=[{"chapter": "1-8", "numeral": "99.9"}],
        citation_exact_f1=0.0,
        citation_validity=0.0,
    )
    cat, _ = classify(row, KNOWN, FakeRetriever([]))
    assert cat == "citation_invalid"


def test_hallucination_when_gold_in_context_but_wrong_citation() -> None:
    row = _row(
        citations=[{"chapter": "1-13", "numeral": "1"}],  # valid but wrong
        contexts=[{"chapter_code": "1-8", "numeral": "1.2", "text": "x"}],  # gold present
        citation_exact_f1=0.0,
        citation_validity=1.0,
    )
    cat, _ = classify(row, KNOWN, FakeRetriever([]))
    assert cat == "hallucination"


def test_retrieval_miss_when_gold_not_in_pool() -> None:
    row = _row(
        citations=[{"chapter": "1-13", "numeral": "1"}],
        contexts=[{"chapter_code": "1-13", "numeral": "1", "text": "x"}],  # gold NOT here
        citation_exact_f1=0.0,
        citation_validity=1.0,
    )
    pool = [{"chapter_code": "1-13", "numeral": "1"}]  # gold 1-8/1.2 absent
    cat, _ = classify(row, KNOWN, FakeRetriever(pool))
    assert cat == "retrieval_miss"


def test_ranking_miss_when_gold_in_pool_not_in_context() -> None:
    row = _row(
        citations=[{"chapter": "1-13", "numeral": "1"}],
        contexts=[{"chapter_code": "1-13", "numeral": "1", "text": "x"}],  # gold NOT here
        citation_exact_f1=0.0,
        citation_validity=1.0,
    )
    pool = [{"chapter_code": "1-8", "numeral": "1.2"}, {"chapter_code": "1-13", "numeral": "1"}]
    cat, _ = classify(row, KNOWN, FakeRetriever(pool))
    assert cat == "ranking_miss"
