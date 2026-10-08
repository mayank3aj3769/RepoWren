"""Offline API tests with a fake inference client."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx

from local_agent.api.app import create_app
from local_agent.api.schemas import ChatMessage


class FakeInference:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    async def is_ready(self) -> bool:
        return self.ready

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        assert messages[-1].content == "hello"
        assert max_tokens == 16
        assert temperature == 0.2
        yield "Hello"
        yield " locally!"


def test_stream_chat_returns_ordered_ndjson_events() -> None:
    async def request_chat() -> httpx.Response:
        app = create_app(FakeInference())  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 16,
                },
            )

    response = asyncio.run(request_chat())
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events == [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " locally!"},
        {"type": "done"},
    ]


def test_chat_returns_503_when_model_server_is_unavailable() -> None:
    async def request_chat() -> httpx.Response:
        app = create_app(FakeInference(ready=False))  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/chat/stream",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

    response = asyncio.run(request_chat())

    assert response.status_code == 503
    assert response.json()["detail"] == "The local llama.cpp server is not ready."
