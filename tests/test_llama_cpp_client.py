"""Offline tests for llama.cpp's streaming wire format."""

import asyncio

import httpx

from local_agent.api.schemas import ChatMessage
from local_agent.inference.llama_cpp import LlamaCppClient


def test_stream_chat_extracts_visible_text() -> None:
    async def collect_tokens() -> list[str]:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            body = (
                'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" two"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body)

        client = LlamaCppClient(
            "http://llama.test", transport=httpx.MockTransport(handler)
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

    assert asyncio.run(collect_tokens()) == ["one", " two"]
