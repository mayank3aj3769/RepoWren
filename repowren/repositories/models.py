"""Small data objects shared by repository services and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def build_chunk_key(
    repository_id: int,
    path: str,
    start_line: int,
    end_line: int,
) -> str:
    """Return the shared identity for one PostgreSQL/Elasticsearch chunk.

    For example, repository 7 and ``src/pricing.py`` lines 1-20 become
    ``7:src/pricing.py:1:20``. The existing PostgreSQL metadata already holds
    these values, so a separate database column is unnecessary.
    """
    return f"{repository_id}:{path}:{start_line}:{end_line}"


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
    """One ranked chunk returned by a repository search backend."""

    chunk_key: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float


GraphRelation = Literal["adjacent", "calls", "imports", "tests"]
GraphTraversalRelation = Literal[
    "adjacent",
    "calls",
    "called_by",
    "imports",
    "imported_by",
    "tests",
    "tested_by",
]


@dataclass(frozen=True, slots=True)
class RepositoryGraphEdge:
    """One deterministic relationship between two safe repository chunks.

    For example, a chunk containing ``calculate_total()`` may have a ``calls``
    edge to the chunk that defines ``calculate_total``. ``detail`` stores that
    symbol name so graph-expanded context can explain why it was included.
    """

    source_chunk_key: str
    target_chunk_key: str
    relation: GraphRelation
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class GraphSearchMatch:
    """A bounded graph neighbor plus provenance back to the previous hop."""

    seed_chunk_key: str
    relation: GraphTraversalRelation
    detail: str | None
    match: SearchMatch
    depth: int = 1
