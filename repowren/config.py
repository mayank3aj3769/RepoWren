"""Configuration for the RepoWren API and external vLLM server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_DATABASE_URL = "postgresql://repowren:repowren@127.0.0.1:5432/repowren"
DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
DEFAULT_ELASTICSEARCH_INDEX = "repowren-chunks"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Existing process variables win over values stored in the local .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from process variables or the local .env file."""

    model_id: str = DEFAULT_MODEL_ID
    vllm_base_url: str = DEFAULT_VLLM_BASE_URL
    api_url: str = DEFAULT_API_URL
    database_url: str = DEFAULT_DATABASE_URL
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    elasticsearch_url: str = DEFAULT_ELASTICSEARCH_URL
    elasticsearch_index: str = DEFAULT_ELASTICSEARCH_INDEX

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build settings from environment variables, falling back to defaults."""
        # Remove trailing slashes once so route construction never creates
        # accidental double slashes such as http://localhost:8001//health.
        return cls(
            model_id=os.getenv("VLLM_MODEL_ID", DEFAULT_MODEL_ID).strip(),
            vllm_base_url=os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL).rstrip("/"),
            api_url=os.getenv(
                "REPOWREN_API_URL",
                os.getenv("LOCAL_AGENT_API_URL", DEFAULT_API_URL),
            ).rstrip("/"),
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip(),
            embedding_model_id=os.getenv(
                "EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID
            ).strip(),
            elasticsearch_url=os.getenv(
                "ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL
            ).rstrip("/"),
            elasticsearch_index=os.getenv(
                "ELASTICSEARCH_INDEX", DEFAULT_ELASTICSEARCH_INDEX
            ).strip(),
        )
