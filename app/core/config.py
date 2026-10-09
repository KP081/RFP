from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- App / API ----
    app_name: str = "RFP RAG API"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ---- Storage ----
    upload_dir: Path = Path("uploads")
    data_dir: Path = Path("data")
    repository_backend: str = "json"  # next: "postgres" / "sqlite"
    max_upload_mb: int = 20

    # ---- Qdrant ----
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rfp_documents"
    qdrant_upsert_batch_size: int = 128

    # ---- Embeddings ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_query_prefix: str = ""  # e5/bge models must prefix
    embedding_document_prefix: str = ""
    embedding_batch_size: int = 32
    embedding_max_seq_length: int | None = None

    # ---- Chunking ----
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ---- Retrieval ----
    top_k: int = 5
    score_threshold: float | None = None

    # ---- LLM ----
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.0
    rag_prompt_file: str = "rag_answer.txt"
    no_answer_message: str = "I could not find relevant information in the document."

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
