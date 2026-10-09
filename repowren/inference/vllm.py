"""Async client for an OpenAI-compatible vLLM server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from repowren.api.schemas import ChatMessage
from repowren.config import Settings
from repowren.inference.base import InferenceError, InferenceStatus


class VLLMClient:
    """Keep vLLM's HTTP protocol out of the rest of the application."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
            transport=transport,
        )
        self._status: InferenceStatus = "not_loaded"

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    @property
    def status(self) -> InferenceStatus:
        return self._status

    async def close(self) -> None:
        await self._client.aclose()

    async def check_ready(self) -> bool:
        """Check whether the configured model server has finished loading."""
        try:
            response = await self._client.get("/health")
        except httpx.HTTPError:
            self._status = "not_loaded"
            return False

        if response.status_code == 200:
            self._status = "ready"
            return True
        self._status = "loading" if response.status_code in {202, 503} else "error"
        return False

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        if not await self.check_ready():
            raise InferenceError(
                f"vLLM server is not ready at {self._settings.vllm_base_url}. "
                "Start the configured vLLM service and try again."
            )

        payload = {
            "model": self.model_id,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    text = choices[0].get("delta", {}).get("content")
                    if text:
                        yield text
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
            self._status = "error"
            raise InferenceError(f"vLLM request failed: {exc}") from exc
