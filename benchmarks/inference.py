"""Measure one direct streaming request to a running llama.cpp server."""

from __future__ import annotations

import argparse
import json
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--prompt",
        default="Explain what a Python function is in two short sentences.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    completion_tokens: int | None = None

    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    with httpx.Client(base_url=args.url, timeout=None) as client:
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line.removeprefix("data:").strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                usage = chunk.get("usage") or {}
                if "completion_tokens" in usage:
                    completion_tokens = usage["completion_tokens"]
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content")
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(text)

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("The server completed without returning visible text.")

    generation_seconds = max(finished - first_token_at, 0.001)
    print("\n--- response ---")
    print("".join(text_parts).strip())
    print("--- measurements ---")
    print(f"time to first token: {first_token_at - started:.3f} s")
    print(f"total request time: {finished - started:.3f} s")
    if completion_tokens is not None:
        print(f"completion tokens: {completion_tokens}")
        print(f"streamed generation rate: {completion_tokens / generation_seconds:.2f} tok/s")
    else:
        print("completion token count: not reported by server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
