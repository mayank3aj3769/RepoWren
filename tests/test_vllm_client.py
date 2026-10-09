"""Offline tests for the vLLM HTTP adapter."""

from __future__ import annotations

import asyncio
import json

import httpx

from local_agent.api.schemas import ChatMessage
from local_agent.config import Settings
from local_agent.inference.vllm import VLLMClient


def _settings() -> Settings:
    return Settings(
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        vllm_base_url="http://vllm.test",
    )


def test_stream_chat_extracts_vllm_sse() -> None:
    async def collect() -> list[str]:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/health":
                return httpx.Response(200)
            assert request.url.path == "/v1/chat/completions"
            assert request.method == "POST"
            payload = json.loads(request.read())
            assert payload == {
                "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "messages": [{"role": "user", "content": "count"}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": True,
            }
            body = (
                'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" two"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body)

        client = VLLMClient(
            _settings(),
            transport=httpx.MockTransport(handler),
        )
        try:
            return [
                token
                async for token in client.stream_chat(
                    [ChatMessage(role="user", content="count")],
                    max_tokens=8,
                    temperature=0.0,
                )
            ]
        finally:
            await client.close()

    assert asyncio.run(collect()) == ["one", " two"]


def test_status_is_not_ready_when_vllm_health_check_fails() -> None:
    async def check() -> tuple[bool, str]:
        client = VLLMClient(
            _settings(),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(503, text="loading")
            ),
        )
        try:
            ready = await client.check_ready()
            return ready, client.status
        finally:
            await client.close()

    assert asyncio.run(check()) == (False, "loading")
