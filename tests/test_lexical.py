"""Offline coverage for the small official Elasticsearch client adapter."""

from __future__ import annotations

import asyncio

from repowren.lexical import ElasticsearchLexicalSearchIndex, INDEX_PROPERTIES
from repowren.repositories.models import RepositoryChunk


class FakeIndices:
    def __init__(self, *, exists: bool = True) -> None:
        self.index_exists = exists
        self.created: dict | None = None
        self.refreshed: str | None = None

    async def exists(self, *, index: str) -> bool:
        del index
        return self.index_exists

    async def create(self, *, index: str, mappings: dict) -> None:
        self.created = {"index": index, "mappings": mappings}

    async def get_mapping(self, *, index: str) -> dict:
        return {index: {"mappings": {"properties": INDEX_PROPERTIES}}}

    async def refresh(self, *, index: str) -> None:
        self.refreshed = index


class FakeElasticsearchClient:
    def __init__(self, *, index_exists: bool = True) -> None:
        self.indices = FakeIndices(exists=index_exists)
        self.deleted: dict | None = None
        self.search_request: dict | None = None
        self.closed = False

    async def delete_by_query(self, **request) -> None:
        self.deleted = request

    async def search(self, **request) -> dict:
        self.search_request = request
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 3.5,
                        "_source": {
                            "chunk_key": "7:src/pricing.py:1:20",
                            "repository_id": 7,
                            "path": "src/pricing.py",
                            "start_line": 1,
                            "end_line": 20,
                            "content": "def price(): ...",
                        },
                    }
                ]
            }
        }

    async def close(self) -> None:
        self.closed = True


def test_initialize_creates_the_requested_mapping() -> None:
    client = FakeElasticsearchClient(index_exists=False)
    index = ElasticsearchLexicalSearchIndex(
        "http://unused",
        "repowren-test",
        client=client,  # type: ignore[arg-type]
    )

    asyncio.run(index.initialize())

    assert client.indices.created == {
        "index": "repowren-test",
        "mappings": {"properties": INDEX_PROPERTIES},
    }


def test_replacement_uses_bulk_actions_and_one_refresh(monkeypatch) -> None:
    client = FakeElasticsearchClient()
    index = ElasticsearchLexicalSearchIndex(
        "http://unused",
        "repowren-test",
        client=client,  # type: ignore[arg-type]
    )
    actions: list[dict] = []

    async def fake_bulk(_client, bulk_actions, **options):
        del _client
        assert options == {"chunk_size": 500, "refresh": False}
        actions.extend(bulk_actions)
        return len(actions), []

    monkeypatch.setattr("repowren.lexical.async_bulk", fake_bulk)
    chunks = [
        RepositoryChunk(
            content="def price(): ...",
            metadata={"path": "src/pricing.py", "start_line": 1, "end_line": 20},
            embedding=[0.1, 0.2],
        )
    ]

    asyncio.run(index.replace_repository_chunks(7, chunks))

    assert client.deleted == {
        "index": "repowren-test",
        "query": {"term": {"repository_id": 7}},
        "conflicts": "proceed",
        "refresh": False,
    }
    assert actions[0]["_id"] == "7:src/pricing.py:1:20"
    assert "embedding" not in actions[0]["_source"]
    assert client.indices.refreshed == "repowren-test"


def test_search_translates_bm25_hits() -> None:
    client = FakeElasticsearchClient()
    index = ElasticsearchLexicalSearchIndex(
        "http://unused",
        "repowren-test",
        client=client,  # type: ignore[arg-type]
    )

    matches = asyncio.run(index.search(7, ["calculate_price", "price total"], 20))

    assert client.search_request == {
        "index": "repowren-test",
        "size": 20,
        "query": {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "content": {"query": "calculate_price", "boost": 3.0}
                        }
                    },
                    {
                        "match_phrase": {
                            "content": {"query": "price total", "boost": 3.0}
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": "calculate_price price total",
                                "operator": "or",
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
                "filter": [{"term": {"repository_id": 7}}],
            }
        },
    }
    assert matches[0].chunk_key == "7:src/pricing.py:1:20"
    assert matches[0].score == 3.5
