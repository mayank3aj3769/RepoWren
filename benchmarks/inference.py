"""Measure one streamed response through the local RepoWren API."""

from __future__ import annotations

import json
import time

import httpx

from local_agent.config import Settings


def main() -> int:
    settings = Settings.from_environment()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Write a short Python function that adds two integers.",
            }
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    started = time.perf_counter()
    first_text_at: float | None = None
    text_parts: list[str] = []

    with httpx.Client(base_url=settings.api_url, timeout=None) as client:
        with client.stream("POST", "/v1/chat/stream", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "token":
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    text_parts.append(event["text"])
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])

    finished = time.perf_counter()
    if first_text_at is None:
        raise RuntimeError("The API completed without returning visible text.")

    print("\n--- response ---")
    print("".join(text_parts).strip())
    print("--- measurements ---")
    print(f"time to first text: {first_text_at - started:.3f} s")
    print(f"total request time: {finished - started:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
