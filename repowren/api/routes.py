"""HTTP routes for inference and repository-aware workflows."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from repowren.api.schemas import (
    ChatMessage,
    ChatRequest,
    FileContentResponse,
    FileMetadataResponse,
    GraphSearchMatchResponse,
    RepositoryCreate,
    RepositoryResponse,
    SearchMatchResponse,
    SearchQueryPlanResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from repowren.inference.base import InferenceBackend, InferenceError
from repowren.lexical import LexicalSearchError
from repowren.repositories.models import RepositoryRecord
from repowren.repositories.service import RepositoryAccessError, RepositoryService
from repowren.repositories.store import RepositoryStore, RepositoryStoreError
from repowren.services.chat import ChatService


router = APIRouter()


def _inference(request: Request) -> InferenceBackend:
    """Return the shared model client created when the app started."""
    return request.app.state.inference


def _store(request: Request) -> RepositoryStore:
    """Return the shared repository store created when the app started."""
    return request.app.state.repository_store


def _repository_service(request: Request) -> RepositoryService:
    """Return the shared service with the store and embedding model wired in."""
    return request.app.state.repository_service


def _repository_response(record: RepositoryRecord) -> RepositoryResponse:
    """Convert the internal repository object into a public API response.

    Example: `RepositoryRecord(id=3, ...)` becomes a Pydantic object that
    FastAPI can validate and serialize to JSON.
    """
    return RepositoryResponse(
        id=record.id,
        name=record.name,
        root_path=record.root_path,
        is_active=record.is_active,
    )


async def _repository_or_404(
    store: RepositoryStore,
    repository_id: int,
) -> RepositoryRecord:
    """Load one repository or return a clear HTTP error to the client.

    Missing data and unavailable storage are different problems: an unknown ID
    becomes 404, while a PostgreSQL failure becomes 503.
    """
    try:
        repository = await store.get(repository_id)
    except RepositoryStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return repository


@router.get("/status", response_model=StatusResponse)
async def status_endpoint(request: Request) -> StatusResponse:
    """Check vLLM readiness without sending a chat request."""
    inference = _inference(request)
    await inference.check_ready()
    return StatusResponse(inference=inference.status, model=inference.model_id)


@router.post("/v1/chat/stream")
async def stream_chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream a normal chat that does not use repository source context."""
    service = ChatService(_inference(request))
    return StreamingResponse(
        service.stream_events(body),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post(
    "/v1/repositories",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_repository(
    request: Request,
    body: RepositoryCreate,
) -> RepositoryResponse:
    """Validate, register, and scan a local Git repository.

    FastAPI has already converted request JSON into `RepositoryCreate`. This
    function then delegates filesystem and database rules to `RepositoryService`.
    """
    service = _repository_service(request)
    try:
        repository = await service.register(body.path, body.name)
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RepositoryStoreError, LexicalSearchError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _repository_response(repository)


@router.get("/v1/repositories", response_model=list[RepositoryResponse])
async def list_repositories(request: Request) -> list[RepositoryResponse]:
    """Return every registered repository, with the active one first."""
    try:
        repositories = await _store(request).list_repositories()
    except RepositoryStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [_repository_response(repository) for repository in repositories]


@router.post(
    "/v1/repositories/{repository_id}/select",
    response_model=RepositoryResponse,
)
async def select_repository(
    request: Request,
    repository_id: int,
) -> RepositoryResponse:
    """Make one registered repository the active repository."""
    try:
        repository = await _store(request).select(repository_id)
    except RepositoryStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return _repository_response(repository)


@router.get(
    "/v1/repositories/{repository_id}/files",
    response_model=list[FileMetadataResponse],
)
async def list_repository_files(
    request: Request,
    repository_id: int,
    query: str | None = Query(default=None, max_length=500),
) -> list[FileMetadataResponse]:
    """Scan and return safe files, optionally filtered by path text.

    `FileMetadata` domain objects are converted one-by-one to response objects.
    Example: `FileMetadata(path='src/cart.py', size_bytes=812)` becomes JSON
    with those same two fields.
    """
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    service = _repository_service(request)
    try:
        files = await asyncio.to_thread(service.list_files, repository, query=query)
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [
        FileMetadataResponse(
            path=item.path,
            size_bytes=item.size_bytes,
        )
        for item in files
    ]


@router.get(
    "/v1/repositories/{repository_id}/file",
    response_model=FileContentResponse,
)
async def read_repository_file(
    request: Request,
    repository_id: int,
    path: str = Query(min_length=1, max_length=4_096),
) -> FileContentResponse:
    """Return one bounded UTF-8 file from inside the repository root."""
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    service = _repository_service(request)
    try:
        content = await asyncio.to_thread(service.read_file, repository, path)
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileContentResponse(path=path, content=content)


@router.post(
    "/v1/repositories/{repository_id}/search",
    response_model=SearchResponse,
)
async def search_repository(
    request: Request,
    repository_id: int,
    body: SearchRequest,
) -> SearchResponse:
    """Return source chunks ranked by semantic and lexical retrieval.

    The API-supplied `max_results` is passed into the service instead of letting
    callers request an unlimited scan result.
    """
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    service = _repository_service(request)
    try:
        query_plan, matches, graph_matches = await service.search_code_with_graph(
            repository,
            body.query,
            max_results=body.max_results,
            max_graph_results=min(body.max_results, 10),
        )
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RepositoryStoreError, LexicalSearchError, InferenceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    seeds_by_key = {match.chunk_key: match for match in matches}
    seeds_by_key.update({item.match.chunk_key: item.match for item in graph_matches})
    return SearchResponse(
        query_plan=SearchQueryPlanResponse(
            embedding_query=query_plan.embedding_query,
            lexical_queries=list(query_plan.lexical_queries),
            cross_questions=list(query_plan.cross_questions),
        ),
        matches=[
            SearchMatchResponse(
                path=item.path,
                start_line=item.start_line,
                end_line=item.end_line,
                text=item.text,
                score=item.score,
            )
            for item in matches
        ],
        graph_matches=[
            GraphSearchMatchResponse(
                path=item.match.path,
                start_line=item.match.start_line,
                end_line=item.match.end_line,
                text=item.match.text,
                relation=item.relation,
                detail=item.detail,
                seed_path=seeds_by_key[item.seed_chunk_key].path,
                seed_start_line=seeds_by_key[item.seed_chunk_key].start_line,
                seed_end_line=seeds_by_key[item.seed_chunk_key].end_line,
                depth=item.depth,
            )
            for item in graph_matches
            if item.seed_chunk_key in seeds_by_key
        ],
    )


@router.post("/v1/repositories/{repository_id}/chat/stream")
async def stream_repository_chat(
    request: Request,
    repository_id: int,
    body: ChatRequest,
) -> StreamingResponse:
    """Add relevant source excerpts before streaming a model response.

    If the original list is ``[user('Explain cart')]``, the enriched list is
    ``[system('<source excerpts>'), user('Explain cart')]``. Existing user and
    assistant messages keep their original order after the new system message.
    """
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    # Context should describe the newest question, not an older chat message.
    user_message = next(
        (message.content for message in reversed(body.messages) if message.role == "user"),
        None,
    )
    if user_message is None:
        raise HTTPException(status_code=422, detail="A user message is required.")

    repository_service = _repository_service(request)
    try:
        context, query_plan, matches, graph_matches = (
            await repository_service.build_context_with_trace(
                repository,
                user_message,
            )
        )
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RepositoryStoreError, LexicalSearchError, InferenceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Keep the user's original conversation unchanged and prepend source context
    # only to the request sent for this response.
    enriched = body.model_copy(
        update={
            "messages": [ChatMessage(role="system", content=context), *body.messages]
        }
    )
    chat_service = ChatService(_inference(request))

    sources_by_key = {match.chunk_key: match for match in matches}
    sources_by_key.update({item.match.chunk_key: item.match for item in graph_matches})
    retrieval_event = {
        "type": "retrieval",
        "query_plan": {
            "embedding_query": query_plan.embedding_query,
            "lexical_queries": list(query_plan.lexical_queries),
            "cross_questions": list(query_plan.cross_questions),
        },
        "matches": [
            {
                "path": item.path,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "text": item.text,
                "score": item.score,
            }
            for item in matches
        ],
        "graph_matches": [
            {
                "path": item.match.path,
                "start_line": item.match.start_line,
                "end_line": item.match.end_line,
                "text": item.match.text,
                "relation": item.relation,
                "detail": item.detail,
                "seed_path": sources_by_key[item.seed_chunk_key].path,
                "seed_start_line": sources_by_key[item.seed_chunk_key].start_line,
                "seed_end_line": sources_by_key[item.seed_chunk_key].end_line,
                "depth": item.depth,
            }
            for item in graph_matches
            if item.seed_chunk_key in sources_by_key
        ],
    }

    async def stream_with_retrieval() -> AsyncIterator[str]:
        yield json.dumps(retrieval_event) + "\n"
        async for event in chat_service.stream_events(enriched):
            yield event

    return StreamingResponse(
        stream_with_retrieval(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )
