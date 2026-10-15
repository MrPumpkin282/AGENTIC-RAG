from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Agentic Rag"
    app_env: str = "dev"
    nvidia_api_key: str = ""
    tavily_api_key: str = ""
    qdrant_api_key: str = ""
    qdrant_index_name: str = ""
    qdrant_name_space: str = ""
    nvidia_embedding_model: str = ""
    nvidia_chat_model: str = ""
    top_k: int = 5
    top_p: float = 0.8
    max_retries: int = 2
    admin_api_key: str = "localhost"
    cluster_end_point: str = ""
    audit_db_path: str = str(BASE_DIR / "data"/ "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
