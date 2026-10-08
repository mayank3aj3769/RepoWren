"""Configuration for the local API and embedded AirLLM runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_CONTEXT = 2_048
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Existing process variables win over values stored in the local .env file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from process variables or the local .env file."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda:0"
    max_context: int = DEFAULT_MAX_CONTEXT
    shards_dir: Path = PROJECT_ROOT / "models" / "airllm"
    hf_home: Path = PROJECT_ROOT / "models" / "huggingface"
    hf_token: str | None = None
    compression: str | None = None
    prefetching: bool = True
    delete_original: bool = False
    api_url: str = DEFAULT_API_URL

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read and validate optional overrides."""
        compression = os.getenv("AIRLLM_COMPRESSION", "").strip() or None
        if compression not in {None, "4bit", "8bit"}:
            raise ValueError("AIRLLM_COMPRESSION must be empty, 4bit, or 8bit.")

        max_context = int(os.getenv("AIRLLM_MAX_CONTEXT", DEFAULT_MAX_CONTEXT))
        if max_context < 256:
            raise ValueError("AIRLLM_MAX_CONTEXT must be at least 256.")

        return cls(
            model_id=os.getenv("AIRLLM_MODEL_ID", DEFAULT_MODEL_ID).strip(),
            device=os.getenv("AIRLLM_DEVICE", "cuda:0").strip(),
            max_context=max_context,
            shards_dir=_project_path(
                os.getenv("AIRLLM_SHARDS_DIR", "models/airllm")
            ),
            hf_home=_project_path(os.getenv("HF_HOME", "models/huggingface")),
            hf_token=os.getenv("HF_TOKEN") or None,
            compression=compression,
            prefetching=_boolean("AIRLLM_PREFETCHING", default=True),
            delete_original=_boolean("AIRLLM_DELETE_ORIGINAL", default=False),
            api_url=os.getenv("LOCAL_AGENT_API_URL", DEFAULT_API_URL),
        )


def _boolean(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
