"""Validated request and response shapes for the local HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    """A bounded request sent to the local model."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=256, ge=1, le=1_024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class StatusResponse(BaseModel):
    """State of the API and its lazily loaded model."""

    api: Literal["ok"] = "ok"
    inference: Literal["not_loaded", "loading", "ready", "error"]
    model: str
