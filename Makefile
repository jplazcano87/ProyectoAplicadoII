.PHONY: install ingest index demo eval test lint

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
	uv run python -m normarag.eval.runner

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/
