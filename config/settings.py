from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    openai_api_key: str = ""
    cohere_api_key: str = ""
    anthropic_api_key: str = ""

    # Models
    embedding_model: str = "text-embedding-3-large"
    generator_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    rerank_model: str = "rerank-multilingual-v3.0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "normarag_ran"

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 6


settings = Settings()
