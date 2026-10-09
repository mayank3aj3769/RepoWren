"""HTTP routes for inference and repository-aware workflows."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from repowren.api.schemas import (
    ChatMessage,
    ChatRequest,
    FileContentResponse,
    FileMetadataResponse,
    RepositoryCreate,
    RepositoryResponse,
    SearchMatchResponse,
    SearchRequest,
    StatusResponse,
)
from repowren.inference.base import InferenceBackend
from repowren.repositories.models import RepositoryRecord
from repowren.repositories.service import RepositoryAccessError, RepositoryService
from repowren.repositories.store import RepositoryStore, RepositoryStoreError
from repowren.services.chat import ChatService


router = APIRouter()


def _inference(request: Request) -> InferenceBackend:
    return request.app.state.inference


def _store(request: Request) -> RepositoryStore:
    return request.app.state.repository_store


def _repository_response(record: RepositoryRecord) -> RepositoryResponse:
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
    try:
        repository = await store.get(repository_id)
    except RepositoryStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return repository


@router.get("/status", response_model=StatusResponse)
async def status_endpoint(request: Request) -> StatusResponse:
    inference = _inference(request)
    await inference.check_ready()
    return StatusResponse(inference=inference.status, model=inference.model_id)


@router.post("/v1/chat/stream")
async def stream_chat(request: Request, body: ChatRequest) -> StreamingResponse:
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
    service = RepositoryService(_store(request))
    try:
        repository = await service.register(body.path, body.name)
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RepositoryStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _repository_response(repository)


@router.get("/v1/repositories", response_model=list[RepositoryResponse])
async def list_repositories(request: Request) -> list[RepositoryResponse]:
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
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    service = RepositoryService(store)
    try:
        files = await asyncio.to_thread(service.list_files, repository, query=query)
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [
        FileMetadataResponse(
            path=item.path,
            size_bytes=item.size_bytes,
            modified_ns=item.modified_ns,
            sha256=item.sha256,
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
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    service = RepositoryService(store)
    try:
        content = await asyncio.to_thread(service.read_file, repository, path)
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileContentResponse(path=path, content=content)


@router.post(
    "/v1/repositories/{repository_id}/search",
    response_model=list[SearchMatchResponse],
)
async def search_repository(
    request: Request,
    repository_id: int,
    body: SearchRequest,
) -> list[SearchMatchResponse]:
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    service = RepositoryService(store)
    try:
        matches = await asyncio.to_thread(
            service.search_code,
            repository,
            body.query,
            max_results=body.max_results,
        )
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [
        SearchMatchResponse(path=item.path, line=item.line, text=item.text)
        for item in matches
    ]


@router.post("/v1/repositories/{repository_id}/chat/stream")
async def stream_repository_chat(
    request: Request,
    repository_id: int,
    body: ChatRequest,
) -> StreamingResponse:
    store = _store(request)
    repository = await _repository_or_404(store, repository_id)
    user_message = next(
        (message.content for message in reversed(body.messages) if message.role == "user"),
        None,
    )
    if user_message is None:
        raise HTTPException(status_code=422, detail="A user message is required.")

    repository_service = RepositoryService(store)
    try:
        context = await asyncio.to_thread(
            repository_service.build_context,
            repository,
            user_message,
        )
    except RepositoryAccessError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    enriched = body.model_copy(
        update={
            "messages": [ChatMessage(role="system", content=context), *body.messages]
        }
    )
    chat_service = ChatService(_inference(request))
    return StreamingResponse(
        chat_service.stream_events(enriched),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )
