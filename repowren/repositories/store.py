"""Persistence interface used by repository workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from repowren.repositories.models import (
    GraphSearchMatch,
    RepositoryChunk,
    RepositoryGraphEdge,
    RepositoryRecord,
    SearchMatch,
)


class RepositoryStoreError(RuntimeError):
    """Repository metadata could not be read from or written to storage."""


class RepositoryStore(Protocol):
    """Minimal persistence operations needed by Phase 2."""

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def register(self, name: str, root_path: str) -> RepositoryRecord: ...

    async def list_repositories(self) -> list[RepositoryRecord]: ...

    async def get(self, repository_id: int) -> RepositoryRecord | None: ...

    async def select(self, repository_id: int) -> RepositoryRecord | None: ...

    async def replace_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
        graph_edges: Sequence[RepositoryGraphEdge],
    ) -> None: ...

    async def search_chunks(
        self,
        repository_id: int,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchMatch]: ...

    async def graph_neighbors(
        self,
        repository_id: int,
        seed_chunk_keys: Sequence[str],
        limit: int,
    ) -> list[GraphSearchMatch]: ...
