.PHONY: install ingest index demo eval experiments validate-golden gold-report qdrant test lint

# Full M5 suite: A/B/C x3 (mean±std) + ablations + top-k sweep + report + errors.
# Override DATASET / RUNS as needed, e.g.: make experiments RUNS=3
RUNS ?= 3

experiments:
	uv run python scripts/run_experiments.py --dataset $(DATASET) --runs $(RUNS)
	uv run python scripts/error_analysis.py

# Dataset used by `make eval` / `make validate-golden`. Override for the real
# golden set, e.g.: make eval DATASET=data/golden/golden_dataset.jsonl
DATASET ?= data/golden/seed.jsonl

install:
	uv sync --extra dev
	uv run pre-commit install

ingest:
	uv run python -m normarag.corpus.download
	uv run python -m normarag.corpus.chunk
	uv run python -m normarag.corpus.refs   # committed (chapter, numeral) keys for CI

# Levanta solo Qdrant (prerrequisito de index / eval / experiments / consultas)
qdrant:
	docker compose up -d qdrant

index:
	uv run python -m normarag.index.store

# Report golden references missing from the corpus -> results/gold_to_fix.md
gold-report:
	uv run python scripts/report_gold_invalid.py --golden $(DATASET)

demo:
	docker compose up --build

eval:
	uv run python -m normarag.eval.runner --config baseline     --dataset $(DATASET)
	uv run python -m normarag.eval.runner --config rag_naive     --dataset $(DATASET)
	uv run python -m normarag.eval.runner --config rag_improved  --dataset $(DATASET) --max-workers 1
	uv run python -m normarag.eval.report

validate-golden:
	uv run python scripts/validate_golden.py $(DATASET)

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/
