"""Small interactive terminal client for the local backend."""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from local_agent.config import Settings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=Settings.from_environment().api_url)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    messages: list[dict[str, str]] = []
    print("Local Coding Agent. Enter /exit to quit.")

    with httpx.Client(base_url=args.api_url, timeout=None) as client:
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if prompt in {"/exit", "/quit"}:
                return 0
            if not prompt:
                continue

            messages.append({"role": "user", "content": prompt})
            answer: list[str] = []
            try:
                with client.stream(
                    "POST",
                    "/v1/chat/stream",
                    json={"messages": messages, "max_tokens": args.max_tokens},
                ) as response:
                    response.raise_for_status()
                    print("agent> ", end="", flush=True)
                    for line in response.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event["type"] == "token":
                            text = event["text"]
                            answer.append(text)
                            print(text, end="", flush=True)
                        elif event["type"] == "error":
                            raise RuntimeError(event["message"])
                    print()
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
                messages.pop()
                print(f"request failed: {exc}", file=sys.stderr)
                continue

            if answer:
                messages.append({"role": "assistant", "content": "".join(answer)})


if __name__ == "__main__":
    raise SystemExit(main())
