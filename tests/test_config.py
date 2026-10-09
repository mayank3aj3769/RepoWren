"""Tests for vLLM environment configuration."""

import pytest

from local_agent.config import DEFAULT_MODEL_ID, DEFAULT_VLLM_BASE_URL, Settings


def test_settings_use_small_public_qwen_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "VLLM_MODEL_ID",
        "VLLM_BASE_URL",
        "HF_TOKEN",
        "LOCAL_AGENT_API_URL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment()

    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.vllm_base_url == DEFAULT_VLLM_BASE_URL


def test_settings_read_vllm_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MODEL_ID", "owner/large-model")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9001/")

    settings = Settings.from_environment()

    assert settings.model_id == "owner/large-model"
    assert settings.vllm_base_url == "http://127.0.0.1:9001"
