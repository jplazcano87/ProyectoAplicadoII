.PHONY: install ingest index demo eval validate-golden test lint

# Dataset used by `make eval` / `make validate-golden`. Override for the real
# golden set, e.g.: make eval DATASET=data/golden/golden_dataset.jsonl
DATASET ?= data/golden/seed.jsonl

install:
	uv sync --extra dev
	uv run pre-commit install

ingest:
	uv run python -m normarag.corpus.download
	uv run python -m normarag.corpus.chunk

index:
	uv run python -m normarag.index.store

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
