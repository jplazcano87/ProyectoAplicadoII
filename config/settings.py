from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys. Cohere's SDK auto-detects CO_API_KEY; we accept either.
    openai_api_key: str = ""
    cohere_api_key: str = ""
    co_api_key: str = ""
    anthropic_api_key: str = ""

    # Models
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072  # text-embedding-3-large native dimension
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    rerank_model: str = "rerank-multilingual-v3.0"

    # RAGAS reliability: prioritize completing over speed (the judge can be slow).
    ragas_timeout_s: int = 600  # per-operation timeout for the RAGAS judge
    ragas_max_workers: int = 4  # judge concurrency (lower = fewer timeouts/limits)
    ragas_max_retries: int = 4

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "normarag_ran"

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 6
    sparse_model: str = "Qdrant/bm25"  # fastembed BM25 sparse model
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    # Min seconds between Cohere rerank calls (trial keys allow ~10/min). Keeps
    # the whole eval under the limit regardless of runner concurrency.
    cohere_min_interval_s: float = 6.5

    # Config C toggles (each improvement is independently ablatable in M5).
    use_hybrid: bool = True  # dense + sparse (BM25) with RRF, vs dense only
    use_rerank: bool = True  # Cohere rerank over fused candidates
    use_citation_prompt: bool = True  # strict citation+abstention prompt vs naive


settings = Settings()
