"""Typed, streaming client for llama.cpp's OpenAI-compatible API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from local_agent.api.schemas import ChatMessage


class InferenceError(RuntimeError):
    """The local inference server could not complete a request."""


class LlamaCppClient:
    """Hide llama.cpp's wire format from the rest of the application."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        payload = {
            "model": "local-model",
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
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
            raise InferenceError(f"llama.cpp request failed: {exc}") from exc
