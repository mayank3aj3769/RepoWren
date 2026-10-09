"""Coordinate a chat request without depending on an inference engine."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from repowren.api.schemas import ChatRequest
from repowren.inference.base import InferenceBackend, InferenceError


class ChatService:
    """Convert model tokens into a small newline-delimited event protocol."""

    def __init__(self, inference: InferenceBackend) -> None:
        self._inference = inference

    async def stream_events(self, request: ChatRequest) -> AsyncIterator[str]:
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
        return json.dumps({"type": event_type, **data}) + "\n"
