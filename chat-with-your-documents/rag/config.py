# All of these can be overridden with RAG_* env vars or a .env file,
# e.g. RAG_LLM_MODEL=qwen2.5
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    # Storage
    documents_dir: Path = Path("documents")
    database_dir: Path = Path("chroma_db")
    collection_name: str = "documents_v2"

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_query_prefix: str = ""
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama3.2"
    ollama_host: str | None = None
    temperature: float = 0.1
    llm_context_window: int = 8192

    # Chunking (measured in embedding-model tokens)
    chunk_max_tokens: int = Field(default=220, ge=32)
    chunk_overlap_tokens: int = Field(default=40, ge=0)

    # Retrieval
    retrieval_mode: RetrievalMode = "hybrid"
    use_reranker: bool = True
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    rrf_k: int = 60

    # Conversation
    condense_followups: bool = True
    history_turns: int = 3

    # Summarization
    summary_map_tokens: int = 2500
    summary_reduce_group: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
