"""Tests for bounded, read-only repository access."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from repowren.repositories.models import FileMetadata, RepositoryRecord
from repowren.repositories.service import RepositoryAccessError, RepositoryService


class MemoryStore:
    def __init__(self) -> None:
        self.repositories: list[RepositoryRecord] = []
        self.files: dict[int, list[FileMetadata]] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def register(self, name: str, root_path: str) -> RepositoryRecord:
        repository = RepositoryRecord(
            id=len(self.repositories) + 1,
            name=name,
            root_path=root_path,
            is_active=not self.repositories,
        )
        self.repositories.append(repository)
        return repository

    async def list_repositories(self) -> list[RepositoryRecord]:
        return self.repositories

    async def get(self, repository_id: int) -> RepositoryRecord | None:
        return next(
            (item for item in self.repositories if item.id == repository_id), None
        )

    async def select(self, repository_id: int) -> RepositoryRecord | None:
        selected = await self.get(repository_id)
        if selected is None:
            return None
        self.repositories = [
            RepositoryRecord(
                id=item.id,
                name=item.name,
                root_path=item.root_path,
                is_active=item.id == repository_id,
            )
            for item in self.repositories
        ]
        return await self.get(repository_id)

    async def replace_files(
        self,
        repository_id: int,
        files: Sequence[FileMetadata],
    ) -> None:
        self.files[repository_id] = list(files)


def _make_repository(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Example repository\n", encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text(
        "def greet(name):\n    return f'Hello {name}'\n",
        encoding="utf-8",
    )
    (tmp_path / "ignored.py").write_text("secret = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"a\x00b")


def test_register_scans_only_safe_text_files(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    service = RepositoryService(store)  # type: ignore[arg-type]

    repository = asyncio.run(service.register(str(tmp_path), "example"))

    paths = {item.path for item in store.files[repository.id]}
    assert "README.md" in paths
    assert "src/main.py" in paths
    assert ".env" not in paths
    assert "ignored.py" not in paths
    assert "binary.dat" not in paths
    assert all(len(item.sha256) == 64 for item in store.files[repository.id])


def test_read_search_and_context_are_repository_bounded(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    service = RepositoryService(store)  # type: ignore[arg-type]
    repository = asyncio.run(service.register(str(tmp_path)))

    assert "def greet" in service.read_file(repository, "src/main.py")
    matches = service.search_code(repository, "hello")
    assert [(item.path, item.line) for item in matches] == [("src/main.py", 2)]

    context = service.build_context(repository, "Where is greet implemented?")
    assert "src/main.py" in context
    assert "1: def greet" in context

    with pytest.raises(RepositoryAccessError, match="inside the repository"):
        service.read_file(repository, "../outside.txt")


def test_registration_requires_a_git_repository(tmp_path) -> None:
    store = MemoryStore()
    service = RepositoryService(store)  # type: ignore[arg-type]

    with pytest.raises(RepositoryAccessError, match="Not a Git repository"):
        asyncio.run(service.register(str(tmp_path)))
