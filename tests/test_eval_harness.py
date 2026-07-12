"""Offline tests for the eval harness: golden loading, runner, and report.

No network: the runner is driven with a fake generator so the whole
generate -> score -> write -> aggregate flow is exercised without API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from normarag.eval.golden import GoldenExample, load_golden, missing_references
from normarag.eval.report import aggregate, generate_report
from normarag.eval.runner import run_config

CHUNKS = [
    {"chapter_code": "1-8", "numeral": "1.2"},
    {"chapter_code": "1-13", "numeral": "1"},
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _dataset() -> list[dict]:
    return [
        {
            "id": "a1",
            "question": "¿Plazo para informar apertura?",
            "answer": "15 días.",
            "references": [{"chapter": "1-8", "numeral": "1.2"}],
            "answerable": True,
            "difficulty": "easy",
            "origin": "seed",
        },
        {
            "id": "a2",
            "question": "Pregunta fuera de alcance",
            "answer": "",
            "references": [],
            "answerable": False,
            "difficulty": "easy",
            "origin": "seed",
        },
    ]


# --- golden schema/loader ---------------------------------------------------


def test_load_golden_and_missing_references(tmp_path: Path) -> None:
    ds = tmp_path / "g.jsonl"
    _write_jsonl(ds, _dataset())
    examples = load_golden(ds)
    assert [e.id for e in examples] == ["a1", "a2"]

    known = {("1-8", "1.2")}
    assert missing_references(examples[0], known) == []
    bad = GoldenExample.model_validate(
        {
            "id": "x",
            "question": "q",
            "answer": "a",
            "references": [{"chapter": "9-9", "numeral": "0.0"}],
            "answerable": True,
            "difficulty": "hard",
            "origin": "human",
        }
    )
    assert missing_references(bad, known) == [("9-9", "0.0")]


def test_golden_rejects_answerable_without_references() -> None:
    with pytest.raises(ValueError):
        GoldenExample.model_validate(
            {
                "id": "x",
                "question": "q",
                "answer": "a",
                "references": [],
                "answerable": True,
                "difficulty": "easy",
                "origin": "seed",
            }
        )


# --- runner + report (offline via fake generator) ---------------------------


def _fake_generator(question: str) -> dict:
    # Answerable question -> correct citation; other -> abstain.
    if "apertura" in question:
        return {
            "answer": "15 días.",
            "citations": [{"chapter": "1-8", "numeral": "1.2"}],
            "abstained": False,
            "contexts": [
                {
                    "chunk_id": "1-8::::1.2::0",
                    "chapter_code": "1-8",
                    "numeral": "1.2",
                    "score": 0.9,
                    "text": "…15 días…",
                }
            ],
            "latency_ms": 12.3,
            "cost_usd": 0.0002,
        }
    return {
        "answer": "",
        "citations": [],
        "abstained": True,
        "contexts": [],
        "latency_ms": 8.0,
        "cost_usd": 0.0001,
    }


def test_run_config_writes_rows_and_metrics(tmp_path: Path) -> None:
    ds = tmp_path / "seed.jsonl"
    _write_jsonl(ds, _dataset())
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(chunks, CHUNKS)
    results = tmp_path / "results"

    rows = run_config(
        "rag_naive",
        ds,
        generate=_fake_generator,
        use_cache=False,
        skip_ragas=True,
        chunks_path=chunks,
        results_dir=results,
        cache_dir=tmp_path / "cache",
        max_workers=2,
    )

    assert (results / "rag_naive.jsonl").exists()
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    # Answerable question scored a perfect, valid citation.
    assert by_id["a1"]["citation_validity"] == 1.0
    assert by_id["a1"]["citation_exact_f1"] == 1.0
    assert by_id["a1"]["abstention_outcome"] == "abstain_tn"
    # Non-answerable question correctly abstained.
    assert by_id["a2"]["abstention_outcome"] == "abstain_tp"


def test_report_aggregate_and_generate(tmp_path: Path) -> None:
    ds = tmp_path / "seed.jsonl"
    _write_jsonl(ds, _dataset())
    chunks = tmp_path / "chunks.jsonl"
    _write_jsonl(chunks, CHUNKS)
    results = tmp_path / "results"

    rows = run_config(
        "baseline",
        ds,
        generate=_fake_generator,
        use_cache=False,
        skip_ragas=True,
        chunks_path=chunks,
        results_dir=results,
        cache_dir=tmp_path / "cache",
    )
    agg = aggregate(rows)
    assert agg["citation_validity"] == 1.0  # both rows valid (one empty=vacuous)
    assert agg["abstention_recall"] == 1.0  # the one non-answerable was abstained
    assert agg["cost_usd_total"] > 0

    aggs = generate_report(results_dir=results)
    assert "baseline" in aggs
    assert (results / "comparativa.md").exists()
    assert (results / "comparativa.csv").exists()
