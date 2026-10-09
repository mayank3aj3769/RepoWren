"""Small terminal chat client for the RepoWren HTTP API."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import httpx

from repowren.config import Settings


def _stream_reply(
    client: httpx.Client,
    messages: list[dict[str, str]],
) -> str:
    """Print one streamed reply and return the complete assistant message."""
    text_parts: list[str] = []
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={"messages": messages},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("type")
            if event_type == "token":
                text = event.get("text", "")
                text_parts.append(text)
                print(text, end="", flush=True)
            elif event_type == "status":
                print(f"[{event.get('message', 'Waiting for model...')}]", flush=True)
            elif event_type == "error":
                raise RuntimeError(event.get("message", "Inference failed."))
    print()
    return "".join(text_parts)


def run_chat(
    api_url: str,
    *,
    prompts: Iterable[str] | None = None,
) -> int:
    """Run an interactive or supplied-prompt chat session."""
    messages: list[dict[str, str]] = []
    prompt_iterator = iter(prompts) if prompts is not None else None

    print(f"RepoWren connected to {api_url}")
    print("Type /exit to quit.")

    with httpx.Client(base_url=api_url, timeout=None) as client:
        while True:
            try:
                prompt = (
                    next(prompt_iterator)
                    if prompt_iterator is not None
                    else input("\nYou> ")
                )
            except (EOFError, StopIteration):
                break

            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt.lower() in {"/exit", "/quit"}:
                break

            messages.append({"role": "user", "content": prompt})
            print("RepoWren> ", end="", flush=True)
            try:
                answer = _stream_reply(client, messages)
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
                messages.pop()
                print(f"\nError: {exc}")
                return 1
            messages.append({"role": "assistant", "content": answer})

    return 0


def main() -> int:
    """Parse command-line options and start a terminal chat."""
    settings = Settings.from_environment()
    parser = argparse.ArgumentParser(description="Chat with the local RepoWren API.")
    parser.add_argument(
        "--api-url",
        default=settings.api_url,
        help=f"RepoWren API base URL (default: {settings.api_url})",
    )
    args = parser.parse_args()
    return run_chat(args.api_url.rstrip("/"))


if __name__ == "__main__":
    raise SystemExit(main())
