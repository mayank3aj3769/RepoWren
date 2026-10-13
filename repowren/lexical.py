"""Small Elasticsearch boundary for lexical repository search."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from elasticsearch import ApiError, AsyncElasticsearch, TransportError
from elasticsearch.helpers import BulkIndexError, async_bulk

from repowren.repositories.models import (
    RepositoryChunk,
    SearchMatch,
    build_chunk_key,
)


BULK_BATCH_SIZE = 500

INDEX_PROPERTIES = {
    "chunk_key": {"type": "keyword"},
    "repository_id": {"type": "long"},
    "path": {"type": "keyword"},
    "start_line": {"type": "integer"},
    "end_line": {"type": "integer"},
    "content": {"type": "text"},
}


class LexicalSearchError(RuntimeError):
    """Elasticsearch could not initialize, replace, or search an index."""


class LexicalSearchIndex(Protocol):
    """Only the lexical-index operations used by RepoWren."""

    async def initialize(self) -> None: ...

    async def replace_repository_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> None: ...

    async def search(
        self,
        repository_id: int,
        queries: Sequence[str],
        limit: int,
    ) -> list[SearchMatch]: ...

    async def close(self) -> None: ...


class ElasticsearchLexicalSearchIndex:
    """Store source text in Elasticsearch and retrieve it with BM25."""

    def __init__(
        self,
        url: str,
        index_name: str,
        *,
        client: AsyncElasticsearch | None = None,
    ) -> None:
        self._index_name = index_name
        self._client = client if client is not None else AsyncElasticsearch(url)

    async def initialize(self) -> None:
        """Create the index once, or verify its required field mappings."""
        try:
            exists = await self._client.indices.exists(index=self._index_name)
            if not exists:
                await self._client.indices.create(
                    index=self._index_name,
                    mappings={"properties": INDEX_PROPERTIES},
                )
                return
            response = await self._client.indices.get_mapping(index=self._index_name)
        except (ApiError, TransportError) as exc:
            raise LexicalSearchError(
                f"Elasticsearch index initialization failed: {exc}"
            ) from exc

        properties = response[self._index_name]["mappings"].get("properties", {})
        for field_name, expected in INDEX_PROPERTIES.items():
            actual_type = properties.get(field_name, {}).get("type")
            if actual_type != expected["type"]:
                raise LexicalSearchError(
                    "Elasticsearch index mapping is incompatible: "
                    f"{field_name} must be {expected['type']}."
                )

    async def replace_repository_chunks(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> None:
        """Delete one repository snapshot, bulk-index its replacement, then refresh.

        A PostgreSQL chunk such as ``src/cart.py`` lines 1-20 is sent as one
        lexical document. Its embedding is deliberately ignored: Elasticsearch
        stores only the text and location fields needed for BM25 retrieval.
        """
        try:
            await self._client.delete_by_query(
                index=self._index_name,
                query={"term": {"repository_id": repository_id}},
                conflicts="proceed",
                refresh=False,
            )
        except (ApiError, TransportError) as exc:
            raise LexicalSearchError(
                f"Elasticsearch repository deletion failed: {exc}"
            ) from exc

        if chunks:
            try:
                await async_bulk(
                    self._client,
                    self._bulk_actions(repository_id, chunks),
                    chunk_size=BULK_BATCH_SIZE,
                    refresh=False,
                )
            except (ApiError, TransportError, BulkIndexError) as exc:
                raise LexicalSearchError(
                    f"Elasticsearch bulk indexing failed: {exc}"
                ) from exc

        try:
            # One refresh after the complete replacement makes the next search
            # see the new snapshot without refreshing every bulk batch.
            await self._client.indices.refresh(index=self._index_name)
        except (ApiError, TransportError) as exc:
            raise LexicalSearchError(
                f"Elasticsearch index refresh failed: {exc}"
            ) from exc

    async def search(
        self,
        repository_id: int,
        queries: Sequence[str],
        limit: int,
    ) -> list[SearchMatch]:
        """Return BM25 results, strongly favoring planned exact strings."""
        should_queries = [
            {
                "match_phrase": {
                    "content": {
                        "query": query,
                        "boost": 3.0,
                    }
                }
            }
            for query in queries
        ]
        # The broad match gives Elasticsearch useful recall when no complete
        # phrase exists, while each phrase clause boosts identifiers such as
        # calculate_invoice_total or exact quoted text.
        should_queries.append(
            {
                "match": {
                    "content": {
                        "query": " ".join(queries),
                        "operator": "or",
                    }
                }
            }
        )
        try:
            response = await self._client.search(
                index=self._index_name,
                size=limit,
                query={
                    "bool": {
                        "should": should_queries,
                        "minimum_should_match": 1,
                        "filter": [{"term": {"repository_id": repository_id}}],
                    }
                },
            )
        except (ApiError, TransportError) as exc:
            raise LexicalSearchError(
                f"Elasticsearch repository search failed: {exc}"
            ) from exc

        matches: list[SearchMatch] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            matches.append(
                SearchMatch(
                    chunk_key=str(source["chunk_key"]),
                    path=str(source["path"]),
                    start_line=int(source["start_line"]),
                    end_line=int(source["end_line"]),
                    text=str(source["content"]),
                    score=float(hit["_score"] or 0.0),
                )
            )
        return matches

    async def close(self) -> None:
        """Close the HTTP connections owned by the official client."""
        await self._client.close()

    def _bulk_actions(
        self,
        repository_id: int,
        chunks: Sequence[RepositoryChunk],
    ) -> Iterator[dict[str, object]]:
        """Yield ordinary bulk actions without constructing NDJSON by hand."""
        for chunk in chunks:
            path = str(chunk.metadata["path"])
            start_line = int(chunk.metadata["start_line"])
            end_line = int(chunk.metadata["end_line"])
            chunk_key = build_chunk_key(
                repository_id,
                path,
                start_line,
                end_line,
            )
            yield {
                "_op_type": "index",
                "_index": self._index_name,
                "_id": chunk_key,
                "_source": {
                    "chunk_key": chunk_key,
                    "repository_id": repository_id,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": chunk.content,
                },
            }
