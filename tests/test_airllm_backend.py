"""Offline tests for the AirLLM adapter."""

from __future__ import annotations

import asyncio
import queue
from pathlib import Path
from typing import Any

from local_agent.api.schemas import ChatMessage
from local_agent.config import Settings
from local_agent.inference.airllm import AirLLMBackend


class FakeTensor:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> "FakeTensor":
        self.device = device
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None
        self.tensor = FakeTensor()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "formatted prompt"

    def __call__(self, prompts: list[str], **options: Any) -> dict[str, FakeTensor]:
        assert prompts == ["formatted prompt"]
        assert options["max_length"] == 496
        return {"input_ids": self.tensor}


class FakeStreamer:
    def __init__(self, _: FakeTokenizer) -> None:
        self._items: queue.Queue[object] = queue.Queue()
        self._end = object()

    def on_finalized_text(self, text: str, *, stream_end: bool = False) -> None:
        if text:
            self._items.put(text)
        if stream_end:
            self._items.put(self._end)

    def __iter__(self) -> "FakeStreamer":
        return self

    def __next__(self) -> str:
        item = self._items.get(timeout=2)
        if item is self._end:
            raise StopIteration
        return str(item)


class FakeModel:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.generation_options: dict[str, Any] | None = None

    def generate(self, _: FakeTensor, **options: Any) -> None:
        self.generation_options = options
        streamer = options["streamer"]
        streamer.on_finalized_text("one")
        streamer.on_finalized_text(" two", stream_end=True)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        model_id="owner/test-model",
        device="cpu",
        max_context=512,
        shards_dir=tmp_path / "shards",
        hf_home=tmp_path / "huggingface",
    )


def test_backend_loads_once_and_streams_formatted_chat(tmp_path: Path) -> None:
    model = FakeModel()
    factory_calls: list[tuple[str, dict[str, Any]]] = []

    def model_factory(model_id: str, **options: Any) -> FakeModel:
        factory_calls.append((model_id, options))
        return model

    backend = AirLLMBackend(
        _settings(tmp_path),
        model_factory=model_factory,
        streamer_factory=FakeStreamer,
    )

    async def collect() -> list[str]:
        first = [
            chunk
            async for chunk in backend.stream_chat(
                [ChatMessage(role="user", content="hello")],
                max_tokens=16,
                temperature=0.0,
            )
        ]
        await backend.ensure_loaded()
        return first

    assert asyncio.run(collect()) == ["one", " two"]
    assert backend.status == "ready"
    assert len(factory_calls) == 1
    assert factory_calls[0][0] == "owner/test-model"
    assert factory_calls[0][1]["device"] == "cpu"
    assert factory_calls[0][1]["layer_shards_saving_path"].endswith(
        "owner--test-model"
    )
    assert model.tokenizer.messages == [{"role": "user", "content": "hello"}]
    assert model.tokenizer.tensor.device == "cpu"
    assert model.generation_options is not None
    assert model.generation_options["do_sample"] is False


def test_model_specific_shard_paths_do_not_collide(tmp_path: Path) -> None:
    first = AirLLMBackend(_settings(tmp_path))
    second_settings = _settings(tmp_path)
    second = AirLLMBackend(
        Settings(
            model_id="another/model",
            device=second_settings.device,
            max_context=second_settings.max_context,
            shards_dir=second_settings.shards_dir,
            hf_home=second_settings.hf_home,
        )
    )

    assert first.model_shards_path != second.model_shards_path
