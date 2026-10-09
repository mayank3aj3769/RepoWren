"""FastAPI application and console entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from repowren.api.routes import router
from repowren.config import Settings
from repowren.inference.base import InferenceBackend
from repowren.inference.vllm import VLLMClient


class HealthResponse(BaseModel):
    """Response returned when the backend process is healthy."""

    status: Literal["ok"]


def create_app(inference: InferenceBackend | None = None) -> FastAPI:
    """Build the app, allowing an offline fake backend in tests."""
    owns_inference = inference is None
    if inference is None:
        settings = Settings.from_environment()
        inference = VLLMClient(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_inference:
            await inference.close()

    application = FastAPI(
        title="RepoWren",
        description="Local HTTP backend for the RepoWren coding agent.",
        version="0.5.0",
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


def main() -> None:
    """Run the development API server from the ``repowren-api`` command."""
    import uvicorn

    uvicorn.run("repowren.api.app:app", host="127.0.0.1", port=8000)
