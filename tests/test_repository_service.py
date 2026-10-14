"""Tests for simple, safe semantic repository access."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from repowren.api.schemas import ChatMessage
from repowren.inference.base import InferenceError
from repowren.repositories.models import (
    GraphSearchMatch,
    RepositoryChunk,
    RepositoryGraphEdge,
    RepositoryRecord,
    SearchMatch,
    build_chunk_key,
)
from repowren.repositories.service import (
    RepositoryAccessError,
    RepositoryService,
    build_repository_graph,
    chunk_ranges,
    fuse_search_results,
    parse_search_plan,
)


class FakeEmbeddings:
    """Tiny deterministic vectors keep tests offline and fast."""

    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.last_texts = list(texts)
        return [[1.0, float("total" in text.casefold()), 0.0] for text in texts]


class FakeQueryInference:
    model_id = "test/model"
    status = "ready"

    def __init__(
        self,
        response: str = (
            '{"embedding_query":"calculate an item total",'
            '"lexical_queries":["calculate_total","sum(items)"],'
            '"cross_questions":["Who calls calculate_total?"]}'
        ),
        *,
        error: bool = False,
    ) -> None:
        self.response = response
        self.error = error
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
        assert max_tokens == 192
        assert temperature == 0.0
        if self.error:
            raise InferenceError("vLLM is starting")
        yield self.response


class FakeLexicalIndex:
    def __init__(self) -> None:
        self.chunks: dict[int, list[RepositoryChunk]] = {}
        self.search_results: list[SearchMatch] | None = None
        self.last_search_limit: int | None = None
        self.last_queries: tuple[str, ...] = ()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def replace_repository_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> None:
        self.chunks[repository_id] = list(chunks)

    async def search(
        self,
        repository_id: int,
        queries: Sequence[str],
        limit: int,
    ) -> list[SearchMatch]:
        self.last_queries = tuple(queries)
        self.last_search_limit = limit
        if self.search_results is not None:
            return self.search_results[:limit]
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
        self.last_search_limit: int | None = None

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
        self.last_search_limit = limit
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
                score=1.0,
            )
            for chunk in self.chunks.get(repository_id, [])[:limit]
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
                seed_key = edge.source_chunk_key
                neighbor_key = edge.target_chunk_key
                relation = edge.relation
            elif edge.target_chunk_key in seeds:
                seed_key = edge.target_chunk_key
                neighbor_key = edge.source_chunk_key
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
    lexical = FakeLexicalIndex()
    service = RepositoryService(
        store,
        FakeEmbeddings(),
        lexical,
        FakeQueryInference(),
    )

    repository = asyncio.run(service.register(str(tmp_path), "example"))

    chunks = store.chunks[repository.id]
    paths = {str(item.metadata["path"]) for item in chunks}
    assert paths == {"README.md", "src/main.py"}
    source_chunks = [item for item in chunks if item.metadata["path"] == "src/main.py"]
    assert [
        (item.metadata["start_line"], item.metadata["end_line"])
        for item in source_chunks
    ] == [(1, 25)]
    assert source_chunks[0].embedding == [1.0, 0.0, 0.0]
    assert lexical.chunks[repository.id] == chunks
    assert source_chunks[0].metadata["chunk_kind"] == "text_window"
    assert build_chunk_key(repository.id, "src/main.py", 1, 25) == (
        "1:src/main.py:1:25"
    )


def test_read_search_context_and_exclusions_are_repository_bounded(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    lexical = FakeLexicalIndex()
    embeddings = FakeEmbeddings()
    inference = FakeQueryInference()
    service = RepositoryService(store, embeddings, lexical, inference)
    repository = asyncio.run(service.register(str(tmp_path)))

    assert "line 1" in service.read_file(repository, "src/main.py")
    matches = asyncio.run(service.search_code(repository, "find the implementation"))
    assert matches[0].path == "README.md"
    assert matches[0].start_line == 1
    assert store.last_search_limit == 20
    assert lexical.last_search_limit == 20
    assert embeddings.last_texts == [
        "calculate an item total",
        "Who calls calculate_total?",
    ]
    assert lexical.last_queries == ("calculate_total", "sum(items)")

    context = asyncio.run(service.build_context(repository, "Explain the code"))
    assert inference.messages[-1].content == "Explain the code"
    assert "### src/main.py lines 1-25" in context
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
    lexical = FakeLexicalIndex()
    service = RepositoryService(
        store,
        FakeEmbeddings(),
        lexical,
        FakeQueryInference(),
    )
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
    lexical_readme = next(
        chunk
        for chunk in lexical.chunks[repository.id]
        if chunk.metadata["path"] == "README.md"
    )
    assert lexical_readme.content == "# Changed"


def test_registration_requires_a_git_repository(tmp_path) -> None:
    service = RepositoryService(
        MemoryStore(),
        FakeEmbeddings(),
        FakeLexicalIndex(),
        FakeQueryInference(),
    )

    with pytest.raises(RepositoryAccessError, match="Not a Git repository"):
        asyncio.run(service.register(str(tmp_path)))


def _match(key: str, score: float = 1.0) -> SearchMatch:
    return SearchMatch(
        chunk_key=key,
        path=f"{key}.py",
        start_line=1,
        end_line=20,
        text=key,
        score=score,
    )


def test_rrf_combines_shared_and_single_backend_chunks() -> None:
    semantic = [_match("shared"), _match("semantic-only")]
    lexical = [_match("lexical-only"), _match("shared")]

    results = fuse_search_results(semantic, lexical, max_results=3)

    assert [item.chunk_key for item in results] == [
        "shared",
        "lexical-only",
        "semantic-only",
    ]
    assert results[0].score == pytest.approx((1 / 61) + (1 / 62))
    assert results[1].score == pytest.approx(1 / 61)


def test_rrf_enforces_max_results_and_keeps_ties_deterministic() -> None:
    results = fuse_search_results(
        [_match("semantic-first"), _match("semantic-second")],
        [_match("lexical-first"), _match("lexical-second")],
        max_results=2,
    )

    assert [item.chunk_key for item in results] == [
        "semantic-first",
        "lexical-first",
    ]


def test_langgraph_workflows_expose_explicit_indexing_and_retrieval_nodes() -> None:
    service = RepositoryService(
        MemoryStore(), FakeEmbeddings(), FakeLexicalIndex(), FakeQueryInference()
    )

    assert set(service._indexing_graph.get_graph().nodes) >= {
        "scan_and_chunk",
        "embed_chunks",
        "link_relationships",
        "persist_snapshot",
    }
    assert set(service._retrieval_graph.get_graph().nodes) >= {
        "plan_queries",
        "retrieve_candidates",
        "fuse_rankings",
        "expand_relationships",
        "assemble_context",
    }


def test_python_chunking_keeps_normal_symbols_whole_and_splits_large_symbols() -> None:
    lines = ["import os", "", "def small():", "    return 1", ""]
    lines.extend(["def large():", *["    value = 1" for _ in range(105)]])

    ranges = chunk_ranges("src/example.py", lines)

    assert ranges[0][:2] == (1, 2)
    assert ranges[1][:2] == (3, 5)
    assert ranges[1][2]["symbol"] == "small"
    assert ranges[2][:2] == (6, 105)
    assert ranges[3][:2] == (106, 111)
    assert ranges[2][2] == {
        "chunk_kind": "python_symbol",
        "symbol": "large",
        "part": 1,
    }


def test_graph_builder_adds_adjacent_calls_and_imports_but_skips_ambiguity() -> None:
    chunks = [
        RepositoryChunk(
            content="import src.pricing\n\ndef checkout():\n    return calculate_total([])",
            metadata={"path": "src/checkout.py", "start_line": 1, "end_line": 4},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="def calculate_total(items):\n    return sum(items)",
            metadata={"path": "src/pricing.py", "start_line": 1, "end_line": 2},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="def duplicate():\n    return 1",
            metadata={"path": "src/one.py", "start_line": 1, "end_line": 2},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="def duplicate():\n    return 2\n\ndef caller():\n    return duplicate()",
            metadata={"path": "src/two.py", "start_line": 1, "end_line": 5},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="first block",
            metadata={"path": "notes.txt", "start_line": 1, "end_line": 20},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="second block",
            metadata={"path": "notes.txt", "start_line": 21, "end_line": 25},
            embedding=[1.0],
        ),
    ]

    edges = build_repository_graph(7, chunks)

    assert any(edge.relation == "calls" and edge.detail == "calculate_total" for edge in edges)
    assert any(edge.relation == "imports" and edge.detail == "src.pricing" for edge in edges)
    assert any(edge.relation == "adjacent" for edge in edges)
    assert not any(edge.relation == "calls" and edge.detail == "duplicate" for edge in edges)
    known_keys = {
        build_chunk_key(
            7,
            str(chunk.metadata["path"]),
            int(chunk.metadata["start_line"]),
            int(chunk.metadata["end_line"]),
        )
        for chunk in chunks
    }
    assert all(edge.source_chunk_key in known_keys for edge in edges)
    assert all(edge.target_chunk_key in known_keys for edge in edges)


def test_graph_builder_preserves_relative_import_and_test_relationships() -> None:
    chunks = [
        RepositoryChunk(
            content="def total():\n    return 1",
            metadata={"path": "src/shop/pricing.py", "start_line": 1, "end_line": 2},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="from . import pricing\n\ndef checkout():\n    return pricing.total()",
            metadata={"path": "src/shop/checkout.py", "start_line": 1, "end_line": 4},
            embedding=[1.0],
        ),
        RepositoryChunk(
            content="def test_total():\n    assert total() == 1",
            metadata={"path": "tests/test_pricing.py", "start_line": 1, "end_line": 2},
            embedding=[1.0],
        ),
    ]

    edges = build_repository_graph(3, chunks)

    assert any(
        edge.relation == "imports" and edge.detail == "shop.pricing"
        for edge in edges
    )
    assert any(edge.relation == "tests" and edge.detail == "total" for edge in edges)


def test_repository_chat_context_keeps_five_seeds_and_adds_bounded_graph_evidence(
    tmp_path,
) -> None:
    store = MemoryStore()
    repository = RepositoryRecord(1, "example", str(tmp_path), True)
    store.repositories.append(repository)
    store.chunks[1] = [
        RepositoryChunk(
            content=f"chunk {number}",
            metadata={
                "path": f"src/file_{number}.py",
                "start_line": 1,
                "end_line": 1,
            },
            embedding=[1.0, 0.0, 0.0],
        )
        for number in range(6)
    ]
    lexical = FakeLexicalIndex()
    lexical.search_results = []
    store.graph_edges[1] = [
        RepositoryGraphEdge(
            source_chunk_key=build_chunk_key(1, "src/file_0.py", 1, 1),
            target_chunk_key=build_chunk_key(1, "src/file_5.py", 1, 1),
            relation="calls",
            detail="helper",
        )
    ]
    service = RepositoryService(
        store,
        FakeEmbeddings(),
        lexical,
        FakeQueryInference(),
    )

    context = asyncio.run(service.build_context(repository, "explain"))

    assert context.count("[hybrid retrieval seed]") == 5
    assert context.count("[graph depth 1:") == 1
    assert "src/file_4.py" in context
    assert "src/file_5.py" in context
    assert "calls (helper)" in context


def test_repository_chat_context_is_bounded_before_chat_validation(tmp_path) -> None:
    repository = RepositoryRecord(
        id=4,
        name="large",
        root_path=str(tmp_path),
        is_active=True,
    )
    store = MemoryStore()
    store.repositories.append(repository)
    store.chunks[repository.id] = [
        RepositoryChunk(
            content="x" * 5_000,
            metadata={
                "path": f"src/file_{index}.py",
                "start_line": 1,
                "end_line": 20,
            },
            embedding=[1.0, 0.0, 0.0],
        )
        for index in range(5)
    ]
    service = RepositoryService(
        store,
        FakeEmbeddings(),
        FakeLexicalIndex(),
        FakeQueryInference(),
    )

    context = asyncio.run(service.build_context(repository, "explain"))

    assert len(context) < 12_000
    assert "[chunk truncated]" in context


def test_search_plan_parser_validates_deduplicates_and_limits_model_json() -> None:
    plan = parse_search_plan(
        """```json
{"embedding_query":" locate the payment retry logic ","lexical_queries":["retryPayment","retryPayment","too","PAYMENT_FAILED",7]}
```""",
        fallback_query="fallback",
    )

    assert plan.embedding_query == "locate the payment retry logic"
    assert plan.lexical_queries == ("retryPayment", "PAYMENT_FAILED")
    assert plan.cross_questions == ()


def test_search_plan_parser_falls_back_when_model_output_is_not_json() -> None:
    plan = parse_search_plan("Here is your answer", fallback_query="find retries")

    assert plan.embedding_query == "find retries"
    assert plan.lexical_queries == ("find retries",)
    assert plan.cross_questions == ()


def test_search_plan_parser_bounds_cross_questions() -> None:
    plan = parse_search_plan(
        '{"embedding_query":"retry logic","lexical_queries":["retry"],'
        '"cross_questions":["Who calls it?","How does it fail?",'
        '"Which tests cover it?","Ignored fourth question"]}',
        fallback_query="fallback",
    )

    assert plan.cross_questions == (
        "Who calls it?",
        "How does it fail?",
        "Which tests cover it?",
    )


def test_search_falls_back_when_vllm_planner_is_unavailable(tmp_path) -> None:
    _make_repository(tmp_path)
    store = MemoryStore()
    lexical = FakeLexicalIndex()
    embeddings = FakeEmbeddings()
    service = RepositoryService(
        store,
        embeddings,
        lexical,
        FakeQueryInference(error=True),
    )
    repository = asyncio.run(service.register(str(tmp_path)))

    matches = asyncio.run(service.search_code(repository, "find retry handling"))

    assert matches
    assert embeddings.last_texts == ["find retry handling"]
    assert lexical.last_queries == ("find retry handling",)
