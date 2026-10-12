"""Small data objects shared by repository services and persistence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepositoryRecord:
    """One local repository registered with RepoWren.

    Example: ``RepositoryRecord(3, 'shop', 'C:\\code\\shop', True)`` means
    database row 3 is the repository currently used for repository-aware chat.
    """

    id: int
    name: str
    root_path: str
    is_active: bool


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """The two facts shown by the live ``/files`` endpoint."""

    path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RepositoryChunk:
    """One small source-code block ready to store in pgvector.

    For example, lines 1-20 of ``src/cart.py`` become one object. ``metadata``
    holds the path and line range that PostgreSQL stores as JSONB, while
    ``embedding`` is the 384-number meaning vector produced for ``content``.
    """

    content: str
    metadata: dict[str, str | int]
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """One semantically similar chunk returned by PostgreSQL."""

    path: str
    start_line: int
    end_line: int
    text: str
    score: float
