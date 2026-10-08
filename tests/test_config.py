"""Tests for AirLLM environment configuration."""

from pathlib import Path

import pytest

from local_agent.config import DEFAULT_MODEL_ID, Settings


def test_settings_use_small_public_qwen_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "AIRLLM_MODEL_ID",
        "AIRLLM_DEVICE",
        "AIRLLM_MAX_CONTEXT",
        "AIRLLM_SHARDS_DIR",
        "AIRLLM_COMPRESSION",
        "AIRLLM_PREFETCHING",
        "AIRLLM_DELETE_ORIGINAL",
        "HF_HOME",
        "HF_TOKEN",
        "LOCAL_AGENT_API_URL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment()

    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.hf_token is None
    assert settings.compression is None
    assert settings.prefetching is True
    assert settings.delete_original is False
    assert settings.shards_dir.is_absolute()


def test_settings_read_airllm_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRLLM_MODEL_ID", "owner/large-model")
    monkeypatch.setenv("AIRLLM_DEVICE", "cpu")
    monkeypatch.setenv("AIRLLM_MAX_CONTEXT", "4096")
    monkeypatch.setenv("AIRLLM_SHARDS_DIR", "models/custom")
    monkeypatch.setenv("AIRLLM_COMPRESSION", "8bit")
    monkeypatch.setenv("AIRLLM_PREFETCHING", "false")
    monkeypatch.setenv("AIRLLM_DELETE_ORIGINAL", "true")
    monkeypatch.setenv("HF_TOKEN", "secret-test-token")

    settings = Settings.from_environment()

    assert settings.model_id == "owner/large-model"
    assert settings.device == "cpu"
    assert settings.max_context == 4096
    assert settings.shards_dir == Path("models/custom").resolve()
    assert settings.compression == "8bit"
    assert settings.prefetching is False
    assert settings.delete_original is True
    assert settings.hf_token == "secret-test-token"
