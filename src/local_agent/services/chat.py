"""Coordinate a chat request without knowing llama.cpp's wire format."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from local_agent.api.schemas import ChatRequest
from local_agent.inference.llama_cpp import InferenceError, LlamaCppClient


class ChatService:
    """Convert model tokens into a small newline-delimited event protocol."""

    def __init__(self, inference: LlamaCppClient) -> None:
        self._inference = inference

    async def stream_events(self, request: ChatRequest) -> AsyncIterator[str]:
        try:
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
