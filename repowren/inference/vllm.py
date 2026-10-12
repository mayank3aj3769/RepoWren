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
        """Create the reusable HTTP client used for readiness and chat calls."""
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
        """Release connections held by the reusable HTTP client."""
        await self._client.aclose()

    async def check_ready(self) -> bool:
        """Check whether the configured model server has finished loading.

        A `200` means requests are ready. `202` or `503` is treated as loading,
        another HTTP status becomes error, and a network failure means the
        server is currently not reachable.
        """
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
        """Send OpenAI-style messages to vLLM and yield text as it arrives.

        For messages ``[ChatMessage(role='user', content='Hello')]``, the JSON
        payload contains ``{'messages': [{'role': 'user', 'content': 'Hello'}],
        'stream': True, ...}``. A vLLM chunk containing text ``'Hi'`` causes
        this async generator to yield ``'Hi'`` immediately.
        """
        if not await self.check_ready():
            raise InferenceError(
                f"vLLM server is not ready at {self._settings.vllm_base_url}. "
                "Start the configured vLLM service and try again."
            )

        # vLLM exposes the same basic request shape as OpenAI chat completions.
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
                    # Streaming responses use Server-Sent Events. Each useful
                    # line begins with "data:" and contains one JSON chunk.
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    # Example data after removing the prefix:
                    # {"choices":[{"delta":{"content":"Hello"}}]}
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    # Some chunks only describe roles or completion state and
                    # have no content. Only actual text is sent to the caller.
                    text = choices[0].get("delta", {}).get("content")
                    if text:
                        yield text
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
            self._status = "error"
            raise InferenceError(f"vLLM request failed: {exc}") from exc
