"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from local_agent.api.routes import router
from local_agent.config import Settings
from local_agent.inference.llama_cpp import LlamaCppClient


class HealthResponse(BaseModel):
    """Response returned when the backend process is healthy."""

    status: Literal["ok"]


def create_app(inference: LlamaCppClient | None = None) -> FastAPI:
    """Build the app, allowing an offline fake client in tests."""
    owns_client = inference is None
    if inference is None:
        settings = Settings.from_environment()
        inference = LlamaCppClient(settings.llama_base_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_client:
            await inference.close()

    application = FastAPI(
        title="Local Coding Agent",
        description="Local HTTP backend for the coding agent.",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.inference = inference
    application.include_router(router)

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Confirm that the backend process is running."""
        return HealthResponse(status="ok")

    return application


app = create_app()
