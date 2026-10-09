"""Persistence interface used by repository workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from repowren.repositories.models import FileMetadata, RepositoryRecord


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

    async def replace_files(
        self,
        repository_id: int,
        files: Sequence[FileMetadata],
    ) -> None: ...
