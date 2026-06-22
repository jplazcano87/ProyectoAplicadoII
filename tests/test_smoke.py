"""Smoke tests: package importable and settings load without errors."""

from config.settings import Settings


def test_settings_load_defaults() -> None:
    s = Settings()
    assert s.embedding_model == "text-embedding-3-large"
    assert s.generator_model == "gpt-4o-mini"
    assert s.qdrant_collection == "normarag_ran"
    assert s.top_k_rerank == 6


def test_normarag_package_importable() -> None:
    import normarag  # noqa: F401
