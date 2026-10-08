"""Lazy, serialized AirLLM inference with streamed text chunks."""

from __future__ import annotations

import asyncio
import os
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from local_agent.api.schemas import ChatMessage
from local_agent.config import Settings
from local_agent.inference.base import InferenceError, InferenceStatus


ModelFactory = Callable[..., Any]
StreamerFactory = Callable[[Any], Any]


class AirLLMBackend:
    """Own one lazily loaded AirLLM model inside the API process."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: ModelFactory | None = None,
        streamer_factory: StreamerFactory | None = None,
    ) -> None:
        self._settings = settings
        self._model_factory = model_factory or _create_model
        self._streamer_factory = streamer_factory or _create_streamer
        self._model: Any | None = None
        self._status: InferenceStatus = "not_loaded"
        self._last_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    @property
    def status(self) -> InferenceStatus:
        return self._status

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def model_shards_path(self) -> Path:
        """Use a model-specific directory so different models never share shards."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "--", self.model_id).strip("-.")
        return self._settings.shards_dir / (slug or "model")

    async def ensure_loaded(self) -> None:
        """Download, split, and initialize the configured model once."""
        if self._model is not None:
            return

        async with self._load_lock:
            if self._model is not None:
                return
            self._status = "loading"
            self._last_error = None
            try:
                self._model = await asyncio.to_thread(self._load_model)
            except Exception as exc:
                self._status = "error"
                self._last_error = str(exc)
                raise InferenceError(f"AirLLM model load failed: {exc}") from exc
            self._status = "ready"

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        await self.ensure_loaded()

        async with self._generation_lock:
            try:
                async for text in self._stream_generation(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    yield text
            except InferenceError:
                raise
            except Exception as exc:
                raise InferenceError(f"AirLLM generation failed: {exc}") from exc

    def _load_model(self) -> Any:
        if self._settings.device.startswith("cuda"):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - dependency is installed in normal use
                raise RuntimeError(
                    "CUDA was requested but Torch is not installed. "
                    "Install the CUDA Torch wheel from the README."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but Torch cannot see an NVIDIA GPU. "
                    "Check the NVIDIA driver and install the CUDA Torch wheel."
                )

        self.model_shards_path.mkdir(parents=True, exist_ok=True)
        self._settings.hf_home.mkdir(parents=True, exist_ok=True)
        # Resolve relative values from .env against this repository before AirLLM imports
        # huggingface_hub. This keeps model files in the documented local cache.
        os.environ["HF_HOME"] = str(self._settings.hf_home)
        return self._model_factory(
            self.model_id,
            device=self._settings.device,
            max_seq_len=self._settings.max_context,
            layer_shards_saving_path=str(self.model_shards_path),
            compression=self._settings.compression,
            hf_token=self._settings.hf_token,
            prefetching=self._settings.prefetching,
            delete_original=self._settings.delete_original,
        )

    async def _stream_generation(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        model = self._model
        if model is None:
            raise InferenceError("AirLLM model is not loaded.")

        tokenizer = model.tokenizer
        prompt = tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_limit = max(1, self._settings.max_context - max_tokens)
        encoded = tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=prompt_limit,
        )
        input_ids = encoded["input_ids"].to(self._settings.device)
        streamer = self._streamer_factory(tokenizer)
        errors: list[BaseException] = []

        generation_options: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature
        if tokenizer.pad_token_id is not None:
            generation_options["pad_token_id"] = tokenizer.pad_token_id

        def generate() -> None:
            try:
                model.generate(input_ids, **generation_options)
            except BaseException as exc:
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = threading.Thread(target=generate, name="airllm-generate", daemon=True)
        worker.start()
        iterator: Iterator[str] = iter(streamer)
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            if chunk:
                yield str(chunk)
        await asyncio.to_thread(worker.join)

        if errors:
            raise InferenceError(f"AirLLM generation failed: {errors[0]}")


def _create_model(model_id: str, **options: Any) -> Any:
    from airllm import AutoModel

    return AutoModel.from_pretrained(model_id, **options)


def _create_streamer(tokenizer: Any) -> Any:
    from transformers import TextIteratorStreamer

    return TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
