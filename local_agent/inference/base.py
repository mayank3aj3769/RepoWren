"""Small interface shared by the vLLM client and offline fakes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal, Protocol

from local_agent.api.schemas import ChatMessage


InferenceStatus = Literal["not_loaded", "loading", "ready", "error"]


class InferenceError(RuntimeError):
    """The configured inference backend could not complete a request."""


class InferenceBackend(Protocol):
    """What the application needs from any local inference engine."""

    @property
    def model_id(self) -> str: ...

    @property
    def status(self) -> InferenceStatus: ...

    async def check_ready(self) -> bool: ...

    async def close(self) -> None: ...

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]: ...
