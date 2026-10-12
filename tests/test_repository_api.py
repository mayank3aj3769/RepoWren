"""Offline API coverage for semantic repository endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx

from repowren.api.app import create_app
from repowren.api.schemas import ChatMessage
from repowren.repositories.models import RepositoryChunk, RepositoryRecord, SearchMatch


class FakeEmbeddings:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class MemoryStore:
    def __init__(self) -> None:
        self.repositories: list[RepositoryRecord] = []
        self.chunks: dict[int, list[RepositoryChunk]] = {}

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
                score=0.95,
            )
            for chunk in self.chunks[repository_id][:limit]
        ]


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
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    store = MemoryStore()
    inference = CapturingInference()
    app = create_app(inference, store, FakeEmbeddings())

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
            assert listed.json() == [{"path": "app.py", "size_bytes": 52}]

            read = await client.get(
                "/v1/repositories/1/file", params={"path": "app.py"}
            )
            assert "calculate_total" in read.json()["content"]

            excluded = await client.get(
                "/v1/repositories/1/file", params={"path": ".env"}
            )
            assert excluded.status_code == 422

            escaped = await client.get(
                "/v1/repositories/1/file", params={"path": "../secret"}
            )
            assert escaped.status_code == 422

            search = await client.post(
                "/v1/repositories/1/search", json={"query": "compute a total"}
            )
            assert search.json() == [
                {
                    "path": "app.py",
                    "start_line": 1,
                    "end_line": 2,
                    "text": "def calculate_total(items):\n    return sum(items)",
                    "score": 0.95,
                }
            ]

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
    assert "### app.py lines 1-2" in inference.messages[0].content
    assert inference.messages[-1].content == "Explain calculate_total"
