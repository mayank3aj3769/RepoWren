"""Offline API coverage for repository-aware Phase 2 endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx

from repowren.api.app import create_app
from repowren.api.schemas import ChatMessage
from repowren.repositories.models import FileMetadata, RepositoryRecord


class MemoryStore:
    def __init__(self) -> None:
        self.repositories: list[RepositoryRecord] = []
        self.files: dict[int, list[FileMetadata]] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def register(self, name: str, root_path: str) -> RepositoryRecord:
        item = RepositoryRecord(
            id=len(self.repositories) + 1,
            name=name,
            root_path=root_path,
            is_active=not self.repositories,
        )
        self.repositories.append(item)
        return item

    async def list_repositories(self) -> list[RepositoryRecord]:
        return self.repositories

    async def get(self, repository_id: int) -> RepositoryRecord | None:
        return next(
            (item for item in self.repositories if item.id == repository_id), None
        )

    async def select(self, repository_id: int) -> RepositoryRecord | None:
        if await self.get(repository_id) is None:
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


class CapturingInference:
    model_id = "test/model"
    status = "ready"

    def __init__(self) -> None:
        self.messages: Sequence[ChatMessage] = []

    async def check_ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        self.messages = messages
        yield "repository answer"


def test_repository_endpoints_and_context_chat(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "app.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    store = MemoryStore()
    inference = CapturingInference()
    app = create_app(inference, store)  # type: ignore[arg-type]

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            registered = await client.post(
                "/v1/repositories",
                json={"path": str(tmp_path), "name": "sample"},
            )
            assert registered.status_code == 201
            assert registered.json()["is_active"] is True

            listed = await client.get("/v1/repositories/1/files")
            assert [item["path"] for item in listed.json()] == ["app.py"]

            read = await client.get(
                "/v1/repositories/1/file", params={"path": "app.py"}
            )
            assert "calculate_total" in read.json()["content"]

            escaped = await client.get(
                "/v1/repositories/1/file", params={"path": "../secret"}
            )
            assert escaped.status_code == 422

            search = await client.post(
                "/v1/repositories/1/search", json={"query": "sum"}
            )
            assert search.json()[0]["line"] == 2

            chat = await client.post(
                "/v1/repositories/1/chat/stream",
                json={
                    "messages": [
                        {"role": "user", "content": "Explain calculate_total"}
                    ]
                },
            )
            assert chat.status_code == 200
            events = [json.loads(line) for line in chat.text.splitlines()]
            assert events[-1] == {"type": "done"}

    asyncio.run(exercise_api())

    assert inference.messages[0].role == "system"
    assert "app.py" in inference.messages[0].content
    assert inference.messages[-1].content == "Explain calculate_total"
