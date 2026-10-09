"""Offline API tests with a fake inference backend."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

import httpx

from repowren.api.app import create_app
from repowren.api.schemas import ChatMessage
from repowren.inference.base import InferenceStatus


class FakeInference:
    model_id = "test/model"

    def __init__(self, status: InferenceStatus = "ready") -> None:
        self.status = status

    async def check_ready(self) -> bool:
        return self.status == "ready"

    async def close(self) -> None:
        return None

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
        self.status = "ready"
        yield "Hello"
        yield " locally!"


def _post_chat(inference: FakeInference) -> httpx.Response:
    async def request_chat() -> httpx.Response:
        app = create_app(inference)  # type: ignore[arg-type]
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

    return asyncio.run(request_chat())


def test_stream_chat_returns_ordered_ndjson_events() -> None:
    response = _post_chat(FakeInference())
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events == [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " locally!"},
        {"type": "done"},
    ]


def test_chat_announces_lazy_model_loading() -> None:
    response = _post_chat(FakeInference(status="not_loaded"))
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events[0] == {
        "type": "status",
        "message": "Waiting for vLLM model server (test/model)...",
    }
    assert events[-1] == {"type": "done"}


def test_status_reports_model_without_loading_it() -> None:
    async def request_status() -> httpx.Response:
        app = create_app(FakeInference(status="not_loaded"))  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get("/status")

    response = asyncio.run(request_status())

    assert response.status_code == 200
    assert response.json() == {
        "api": "ok",
        "inference": "not_loaded",
        "model": "test/model",
    }
