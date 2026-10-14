"""Offline API coverage for semantic repository endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest

from repowren.api.app import create_app
from repowren.api.schemas import ChatMessage
from repowren.lexical import LexicalSearchError
from repowren.repositories.models import (
    GraphSearchMatch,
    RepositoryChunk,
    RepositoryGraphEdge,
    RepositoryRecord,
    SearchMatch,
    build_chunk_key,
)


class FakeEmbeddings:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeLexicalIndex:
    def __init__(
        self,
        *,
        replace_error: bool = False,
        search_error: bool = False,
    ) -> None:
        self.chunks: dict[int, list[RepositoryChunk]] = {}
        self.replace_error = replace_error
        self.search_error = search_error

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def replace_repository_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> None:
        if self.replace_error:
            raise LexicalSearchError("Elasticsearch repository indexing failed.")
        self.chunks[repository_id] = list(chunks)

    async def search(
        self,
        repository_id: int,
        queries: Sequence[str],
        limit: int,
    ) -> list[SearchMatch]:
        del queries
        if self.search_error:
            raise LexicalSearchError("Elasticsearch repository search failed.")
        return [
            SearchMatch(
                chunk_key=build_chunk_key(
                    repository_id,
                    str(chunk.metadata["path"]),
                    int(chunk.metadata["start_line"]),
                    int(chunk.metadata["end_line"]),
                ),
                path=str(chunk.metadata["path"]),
                start_line=int(chunk.metadata["start_line"]),
                end_line=int(chunk.metadata["end_line"]),
                text=chunk.content,
                score=2.0,
            )
            for chunk in self.chunks.get(repository_id, [])[:limit]
        ]


class MemoryStore:
    def __init__(self) -> None:
        self.repositories: list[RepositoryRecord] = []
        self.chunks: dict[int, list[RepositoryChunk]] = {}
        self.graph_edges: dict[int, list[RepositoryGraphEdge]] = {}

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
        graph_edges: Sequence[RepositoryGraphEdge],
    ) -> None:
        self.chunks[repository_id] = list(chunks)
        self.graph_edges[repository_id] = list(graph_edges)

    async def search_chunks(
        self,
        repository_id: int,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchMatch]:
        del query_embedding
        return [
            SearchMatch(
                chunk_key=build_chunk_key(
                    repository_id,
                    str(chunk.metadata["path"]),
                    int(chunk.metadata["start_line"]),
                    int(chunk.metadata["end_line"]),
                ),
                path=str(chunk.metadata["path"]),
                start_line=int(chunk.metadata["start_line"]),
                end_line=int(chunk.metadata["end_line"]),
                text=chunk.content,
                score=0.95,
            )
            for chunk in self.chunks[repository_id][:limit]
        ]

    async def graph_neighbors(
        self,
        repository_id: int,
        seed_chunk_keys: Sequence[str],
        limit: int,
    ) -> list[GraphSearchMatch]:
        seeds = set(seed_chunk_keys)
        chunks_by_key = {
            build_chunk_key(
                repository_id,
                str(chunk.metadata["path"]),
                int(chunk.metadata["start_line"]),
                int(chunk.metadata["end_line"]),
            ): chunk
            for chunk in self.chunks.get(repository_id, [])
        }
        results: list[GraphSearchMatch] = []
        seen: set[str] = set()
        for edge in self.graph_edges.get(repository_id, []):
            if edge.source_chunk_key in seeds:
                seed_key, neighbor_key = edge.source_chunk_key, edge.target_chunk_key
                relation = edge.relation
            elif edge.target_chunk_key in seeds:
                seed_key, neighbor_key = edge.target_chunk_key, edge.source_chunk_key
                relation = {
                    "calls": "called_by",
                    "imports": "imported_by",
                }.get(edge.relation, edge.relation)
            else:
                continue
            if neighbor_key in seeds or neighbor_key in seen:
                continue
            chunk = chunks_by_key[neighbor_key]
            seen.add(neighbor_key)
            results.append(
                GraphSearchMatch(
                    seed_chunk_key=seed_key,
                    relation=relation,
                    detail=edge.detail,
                    match=SearchMatch(
                        chunk_key=neighbor_key,
                        path=str(chunk.metadata["path"]),
                        start_line=int(chunk.metadata["start_line"]),
                        end_line=int(chunk.metadata["end_line"]),
                        text=chunk.content,
                        score=0.0,
                    ),
                )
            )
            if len(results) == limit:
                break
        return results


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
        if messages[0].role == "system" and "hybrid searches" in messages[0].content:
            assert max_tokens == 192
            assert temperature == 0.0
            yield (
                '{"embedding_query":"compute an item total",'
                '"lexical_queries":["calculate_total","sum(items)"],'
                '"cross_questions":["Who calls calculate_total?"]}'
            )
            return
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
    app = create_app(inference, store, FakeEmbeddings(), FakeLexicalIndex())

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
            assert search.json()["query_plan"] == {
                "embedding_query": "compute an item total",
                "lexical_queries": [
                    "calculate_total",
                    "sum(items)",
                    "compute a total",
                ],
                "cross_questions": ["Who calls calculate_total?"],
            }
            assert search.json()["matches"][0] == {
                "path": "app.py",
                "start_line": 1,
                "end_line": 2,
                "text": "def calculate_total(items):\n    return sum(items)",
                "score": pytest.approx(4 / 61),
            }
            assert search.json()["graph_matches"] == []

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
            assert events[0]["type"] == "retrieval"
            assert events[0]["query_plan"]["cross_questions"] == [
                "Who calls calculate_total?"
            ]
            assert events[-1] == {"type": "done"}

    asyncio.run(exercise_api())

    assert inference.messages[0].role == "system"
    assert "### app.py lines 1-2" in inference.messages[0].content
    assert inference.messages[-1].content == "Explain calculate_total"


@pytest.mark.parametrize("operation", ["replace", "search"])
def test_elasticsearch_failures_return_503(tmp_path, operation: str) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    lexical = FakeLexicalIndex(
        replace_error=operation == "replace",
        search_error=operation == "search",
    )
    app = create_app(
        CapturingInference(),
        MemoryStore(),
        FakeEmbeddings(),
        lexical,
    )

    async def exercise_api() -> int:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            registered = await client.post(
                "/v1/repositories",
                json={"path": str(tmp_path)},
            )
            if operation == "replace":
                return registered.status_code
            searched = await client.post(
                "/v1/repositories/1/search",
                json={"query": "value"},
            )
            return searched.status_code

    assert asyncio.run(exercise_api()) == 503
