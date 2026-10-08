"""FastAPI application entry point."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from local_agent.api.routes import router
from local_agent.config import Settings
from local_agent.inference.airllm import AirLLMBackend
from local_agent.inference.base import InferenceBackend


class HealthResponse(BaseModel):
    """Response returned when the backend process is healthy."""

    status: Literal["ok"]


def create_app(inference: InferenceBackend | None = None) -> FastAPI:
    """Build the app, allowing an offline fake backend in tests."""
    if inference is None:
        settings = Settings.from_environment()
        inference = AirLLMBackend(settings)

    application = FastAPI(
        title="Local Coding Agent",
        description="Local HTTP backend for the coding agent.",
        version="0.3.0",
    )
    application.state.inference = inference
    application.include_router(router)

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Confirm that the backend process is running."""
        return HealthResponse(status="ok")

    return application


app = create_app()
