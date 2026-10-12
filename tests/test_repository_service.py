"""Tests for simple, safe semantic repository access."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from repowren.repositories.models import RepositoryChunk, RepositoryRecord, SearchMatch
from repowren.repositories.service import RepositoryAccessError, RepositoryService


class FakeEmbeddings:
    """Tiny deterministic vectors keep tests offline and fast."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float("total" in text.casefold()), 0.0] for text in texts]


class MemoryStore:
    def __init__(self) -> None:
        self.repositories: list[RepositoryRecord] = []
        self.chunks: dict[int, list[RepositoryChunk]] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def register(self, name: str, root_path: str) -> RepositoryRecord:
        existing = next(
            (item for item in self.repositories if item.root_path == root_path), None
        )
        if existing is not None:
            return existing
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
        return await self.get(repository_id)

    async def replace_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> None:
        self.chunks[repository_id] = list(chunks)

    async def search_chunks(
        self,
        repository_id: int,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchMatch]:
        del query_embedding
        return [
            SearchMatch(
                path=str(chunk.metadata["path"]),
                start_line=int(chunk.metadata["start_line"]),
                end_line=int(chunk.metadata["end_line"]),
                text=chunk.content,
                score=1.0,
            )
            for chunk in self.chunks.get(repository_id, [])[:limit]
        ]


def _make_repository(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Example repository\n", encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text(
        "\n".join(f"line {number}" for number in range(1, 26)),
        encoding="utf-8",
    )
    (tmp_path / "ignored.py").write_text("secret = True\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"a\x00b")


def test_register_stores_simple_chunks_and_metadata(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    service = RepositoryService(store, FakeEmbeddings())

    repository = asyncio.run(service.register(str(tmp_path), "example"))

    chunks = store.chunks[repository.id]
    paths = {str(item.metadata["path"]) for item in chunks}
    assert paths == {"README.md", "src/main.py"}
    source_chunks = [item for item in chunks if item.metadata["path"] == "src/main.py"]
    assert [
        (item.metadata["start_line"], item.metadata["end_line"])
        for item in source_chunks
    ] == [(1, 20), (21, 25)]
    assert source_chunks[0].embedding == [1.0, 0.0, 0.0]


def test_read_search_context_and_exclusions_are_repository_bounded(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    service = RepositoryService(store, FakeEmbeddings())
    repository = asyncio.run(service.register(str(tmp_path)))

    assert "line 1" in service.read_file(repository, "src/main.py")
    matches = asyncio.run(service.search_code(repository, "find the implementation"))
    assert matches[0].path == "README.md"
    assert matches[0].start_line == 1

    context = asyncio.run(service.build_context(repository, "Explain the code"))
    assert "### src/main.py lines 1-20" in context
    assert "line 1" in context

    with pytest.raises(RepositoryAccessError, match="inside the repository"):
        service.read_file(repository, "../outside.txt")
    with pytest.raises(RepositoryAccessError, match="excluded"):
        service.read_file(repository, ".env")
    with pytest.raises(RepositoryAccessError, match="excluded"):
        service.read_file(repository, "ignored.py")


def test_reregister_replaces_the_old_chunk_snapshot(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    service = RepositoryService(store, FakeEmbeddings())
    repository = asyncio.run(service.register(str(tmp_path)))
    (tmp_path / "README.md").write_text("# Changed\n", encoding="utf-8")

    same_repository = asyncio.run(service.register(str(tmp_path)))

    assert same_repository.id == repository.id
    readme = next(
        chunk
        for chunk in store.chunks[repository.id]
        if chunk.metadata["path"] == "README.md"
    )
    assert readme.content == "# Changed"


def test_registration_requires_a_git_repository(tmp_path) -> None:
    service = RepositoryService(MemoryStore(), FakeEmbeddings())

    with pytest.raises(RepositoryAccessError, match="Not a Git repository"):
        asyncio.run(service.register(str(tmp_path)))
