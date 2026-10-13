"""FastAPI application and console entry point."""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from repowren.api.routes import router
from repowren.config import Settings
from repowren.embeddings import EmbeddingBackend, SentenceTransformerEmbeddings
from repowren.inference.base import InferenceBackend
from repowren.inference.vllm import VLLMClient
from repowren.lexical import ElasticsearchLexicalSearchIndex, LexicalSearchIndex
from repowren.persistence.postgres import PostgresRepositoryStore
from repowren.repositories.store import RepositoryStore
from repowren.repositories.service import RepositoryService


class HealthResponse(BaseModel):
    """Response returned when the backend process is healthy."""

    status: Literal["ok"]


def create_app(
    inference: InferenceBackend | None = None,
    repository_store: RepositoryStore | None = None,
    embeddings: EmbeddingBackend | None = None,
    lexical_index: LexicalSearchIndex | None = None,
) -> FastAPI:
    """Build and connect the API's inference and repository dependencies.

    Tests can pass fake objects here, which lets them exercise the real routes
    without starting PostgreSQL, Docker, or a model server.
    """
    owns_inference = inference is None
    settings = Settings.from_environment()
    if inference is None:
        inference = VLLMClient(settings)
    owns_repository_store = repository_store is None
    if repository_store is None:
        repository_store = PostgresRepositoryStore(settings.database_url)
    if embeddings is None:
        embeddings = SentenceTransformerEmbeddings(settings.embedding_model_id)
    owns_lexical_index = lexical_index is None
    if lexical_index is None:
        lexical_index = ElasticsearchLexicalSearchIndex(
            settings.elasticsearch_url,
            settings.elasticsearch_index,
        )
    repository_service = RepositoryService(
        repository_store,
        embeddings,
        lexical_index,
        inference,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Prepare owned resources at startup and release them at shutdown."""
        try:
            # Both indexes must be ready before the API accepts requests.
            if owns_repository_store:
                await repository_store.initialize()
            if owns_lexical_index:
                await lexical_index.initialize()
            yield
        finally:
            # Only close objects created here; injected test objects belong to
            # the caller that supplied them.
            if owns_repository_store:
                await repository_store.close()
            if owns_lexical_index:
                await lexical_index.close()
            if owns_inference:
                await inference.close()

    application = FastAPI(
        title="RepoWren",
        description="Local HTTP backend for the RepoWren coding agent.",
        version="0.6.0",
        lifespan=lifespan,
    )
    application.state.inference = inference
    application.state.repository_store = repository_store
    application.state.lexical_index = lexical_index
    application.state.repository_service = repository_service
    application.include_router(router)

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Confirm that the backend process is running."""
        return HealthResponse(status="ok")

    return application


app = create_app()


def selector_loop_factory(*, use_subprocess: bool = False):
    """Create the Windows loop required by psycopg's async connection.

    Uvicorn creates a Proactor loop by default on Windows. Psycopg cannot use
    that loop, so the console command gives Uvicorn this explicit factory.
    ``use_subprocess`` is accepted because Uvicorn passes it to loop factories.
    """
    del use_subprocess
    return asyncio.SelectorEventLoop()


def main() -> None:
    """Start Uvicorn when the user runs the ``repowren-api`` command."""
    import uvicorn

    loop = (
        "repowren.api.app:selector_loop_factory"
        if sys.platform == "win32"
        else "asyncio"
    )
    uvicorn.run("repowren.api.app:app", host="127.0.0.1", port=8000, loop=loop)
