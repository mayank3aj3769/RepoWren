"""Configuration tests for the Hugging Face model downloader."""

from pathlib import Path

import pytest

from scripts.download_model import (
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_ID,
    ModelConfig,
)


def test_model_config_uses_safe_project_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "HF_MODEL_ID",
        "HF_MODEL_FILE",
        "HF_MODEL_REVISION",
        "HF_MODEL_DIR",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    config = ModelConfig.from_environment(tmp_path)

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.filename == DEFAULT_MODEL_FILE
    assert config.destination == tmp_path / "models" / DEFAULT_MODEL_FILE
    assert config.token is None


def test_model_config_reads_token_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_MODEL_ID", "owner/private-model")
    monkeypatch.setenv("HF_MODEL_FILE", "model.gguf")
    monkeypatch.setenv("HF_TOKEN", "secret-test-token")

    config = ModelConfig.from_environment(tmp_path)

    assert config.model_id == "owner/private-model"
    assert config.token == "secret-test-token"


def test_model_file_cannot_escape_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_MODEL_FILE", "../outside.gguf")

    with pytest.raises(ValueError, match="relative Hugging Face filename"):
        ModelConfig.from_environment(tmp_path)
