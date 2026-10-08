"""Small, dependency-free configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_API_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Existing process variables win over values stored in the local .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True, slots=True)
class Settings:
    """Addresses of the two local HTTP processes."""

    llama_base_url: str = DEFAULT_LLAMA_BASE_URL
    api_url: str = DEFAULT_API_URL

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read optional overrides without requiring a .env parser."""
        return cls(
            llama_base_url=os.getenv("LLAMA_BASE_URL", DEFAULT_LLAMA_BASE_URL),
            api_url=os.getenv("LOCAL_AGENT_API_URL", DEFAULT_API_URL),
        )
