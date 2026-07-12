.PHONY: install ingest index demo eval validate-golden test lint

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
	uv run python -m normarag.eval.runner --config baseline --dataset data/golden/seed.jsonl
	uv run python -m normarag.eval.runner --config rag_naive --dataset data/golden/seed.jsonl
	uv run python -m normarag.eval.report

validate-golden:
	uv run python scripts/validate_golden.py data/golden/seed.jsonl

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/
