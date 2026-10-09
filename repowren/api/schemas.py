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


class RepositoryCreate(BaseModel):
    """Register a local Git repository that RepoWren may inspect."""

    path: str = Field(min_length=1, max_length=4_096)
    name: str | None = Field(default=None, min_length=1, max_length=200)


class RepositoryResponse(BaseModel):
    """Public representation of a registered repository."""

    id: int
    name: str
    root_path: str
    is_active: bool


class FileMetadataResponse(BaseModel):
    """Safe metadata returned for a repository file."""

    path: str
    size_bytes: int
    modified_ns: int
    sha256: str


class FileContentResponse(BaseModel):
    """UTF-8 contents of one repository file."""

    path: str
    content: str


class SearchRequest(BaseModel):
    """Bounded literal source-code search."""

    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=50, ge=1, le=100)


class SearchMatchResponse(BaseModel):
    """One source line containing the requested text."""

    path: str
    line: int
    text: str
