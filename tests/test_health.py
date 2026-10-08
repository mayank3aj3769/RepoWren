"""Tests for the backend health endpoint."""

import asyncio

import httpx

from local_agent.api.app import app


def test_health_returns_ok() -> None:
    """The health endpoint reports that the backend process is alive."""

    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
