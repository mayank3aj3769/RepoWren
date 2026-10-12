"""Coordinate a chat request without depending on an inference engine."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from repowren.api.schemas import ChatRequest
from repowren.inference.base import InferenceBackend, InferenceError


class ChatService:
    """Convert model tokens into a small newline-delimited event protocol."""

    def __init__(self, inference: InferenceBackend) -> None:
        """Accept any inference backend that follows the small shared protocol."""
        self._inference = inference

    async def stream_events(self, request: ChatRequest) -> AsyncIterator[str]:
        """Translate model output into RepoWren's newline-delimited events.

        If inference yields ``'Hello'`` and ``' world'``, the client receives
        two `token` JSON lines followed by one `done` line. An `InferenceError`
        becomes one `error` line instead of crashing the streaming response.
        """
        try:
            if self._inference.status != "ready":
                yield self._event(
                    "status",
                    message=f"Waiting for vLLM model server ({self._inference.model_id})...",
                )
            async for text in self._inference.stream_chat(
                request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield self._event("token", text=text)
            yield self._event("done")
        except InferenceError as exc:
            yield self._event("error", message=str(exc))

    @staticmethod
    def _event(event_type: str, **data: str) -> str:
        """Encode one event as a JSON line that clients can process immediately.

        ``_event('token', text='Hi')`` returns
        ``'{"type": "token", "text": "Hi"}\\n'``.
        """
        return json.dumps({"type": event_type, **data}) + "\n"
