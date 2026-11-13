from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Agentic Rag"
    app_env: str = "dev"
    log_level: str = "WARNING"

    # API Keys
    nvidia_api_key: str = ""
    tavily_api_key: str = ""
    qdrant_api_key: str = ""

    # Qdrant Vector DB
    cluster_end_point: str = ""
    qdrant_index_name: str = "STORAGE"
    qdrant_name_space: str = "DEFAULT"

    # NVIDIA NIM Models
    nvidia_embedding_model: str = "nvidia/nemotron-3.5-embed-1b"
    nvidia_chat_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

    # Retrieval & Ranking Parameters
    top_k: int = 5
    top_p: float = 0.8
    max_retries: int = 2
    rrf_k: int = 60
    rerank_model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    rerank_top_k: int = 5
    max_context_chars: int = 18000
    workflow_recursion_limit: int = 25

    # Security & Storage
    admin_api_key: str = "localhost"
    cache_similarity_threshold: float = 0.85
    max_upload_size_mb: int = 25
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
