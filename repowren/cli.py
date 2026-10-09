"""Terminal client for chat and read-only repository exploration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import httpx

from repowren.config import Settings


COMMAND_HELP = """Commands:
  /repo add <path>   Register and select a local Git repository
  /repo list         List registered repositories
  /repo use <id>     Select a registered repository
  /files [query]     List files in the selected repository
  /read <path>       Read a UTF-8 file from the selected repository
  /search <query>    Search source lines in the selected repository
  /help              Show this help
  /exit              Quit"""


def _stream_reply(
    client: httpx.Client,
    messages: list[dict[str, str]],
    repository_id: int | None = None,
) -> str:
    """Print one streamed reply and return the complete assistant message."""
    path = (
        f"/v1/repositories/{repository_id}/chat/stream"
        if repository_id is not None
        else "/v1/chat/stream"
    )
    text_parts: list[str] = []
    with client.stream("POST", path, json={"messages": messages}) as response:
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


def _require_repository(repository_id: int | None) -> int:
    if repository_id is None:
        raise RuntimeError("Select a repository first with /repo add or /repo use.")
    return repository_id


def _handle_command(
    client: httpx.Client,
    command: str,
    active_repository_id: int | None,
) -> tuple[bool, int | None]:
    """Handle one slash command and return its active repository selection."""
    if command == "/help":
        print(COMMAND_HELP)
        return True, active_repository_id

    if command == "/repo list":
        response = client.get("/v1/repositories")
        response.raise_for_status()
        repositories = response.json()
        if not repositories:
            print("No repositories registered.")
        for repository in repositories:
            marker = "*" if repository["is_active"] else " "
            print(
                f"{marker} {repository['id']}: {repository['name']} "
                f"({repository['root_path']})"
            )
            if repository["is_active"]:
                active_repository_id = repository["id"]
        return True, active_repository_id

    if command.startswith("/repo add "):
        path = command.removeprefix("/repo add ").strip().strip('"')
        if not path:
            raise RuntimeError("Usage: /repo add <path>")
        response = client.post("/v1/repositories", json={"path": path})
        response.raise_for_status()
        repository = response.json()
        if not repository["is_active"]:
            select_response = client.post(
                f"/v1/repositories/{repository['id']}/select"
            )
            select_response.raise_for_status()
            repository = select_response.json()
        print(f"Using repository {repository['id']}: {repository['name']}")
        return True, repository["id"]

    if command.startswith("/repo use "):
        raw_id = command.removeprefix("/repo use ").strip()
        try:
            repository_id = int(raw_id)
        except ValueError as exc:
            raise RuntimeError("Usage: /repo use <numeric-id>") from exc
        response = client.post(f"/v1/repositories/{repository_id}/select")
        response.raise_for_status()
        repository = response.json()
        print(f"Using repository {repository['id']}: {repository['name']}")
        return True, repository["id"]

    if command == "/repo" or command.startswith("/repo "):
        raise RuntimeError("Usage: /repo add <path>, /repo list, or /repo use <id>")

    if command == "/files" or command.startswith("/files "):
        repository_id = _require_repository(active_repository_id)
        query = command.removeprefix("/files").strip()
        response = client.get(
            f"/v1/repositories/{repository_id}/files",
            params={"query": query} if query else None,
        )
        response.raise_for_status()
        files = response.json()
        for item in files:
            print(f"{item['path']} ({item['size_bytes']} bytes)")
        if not files:
            print("No matching files.")
        return True, active_repository_id

    if command.startswith("/read "):
        repository_id = _require_repository(active_repository_id)
        path = command.removeprefix("/read ").strip()
        response = client.get(
            f"/v1/repositories/{repository_id}/file",
            params={"path": path},
        )
        response.raise_for_status()
        print(response.json()["content"])
        return True, active_repository_id

    if command.startswith("/search "):
        repository_id = _require_repository(active_repository_id)
        query = command.removeprefix("/search ").strip()
        response = client.post(
            f"/v1/repositories/{repository_id}/search",
            json={"query": query},
        )
        response.raise_for_status()
        matches = response.json()
        for match in matches:
            print(f"{match['path']}:{match['line']}: {match['text']}")
        if not matches:
            print("No matches.")
        return True, active_repository_id

    if command.startswith("/"):
        raise RuntimeError("Unknown command. Type /help for available commands.")
    return False, active_repository_id


def run_chat(
    api_url: str,
    *,
    prompts: Iterable[str] | None = None,
) -> int:
    """Run an interactive or supplied-prompt chat session."""
    messages: list[dict[str, str]] = []
    active_repository_id: int | None = None
    prompt_iterator = iter(prompts) if prompts is not None else None

    print(f"RepoWren connected to {api_url}")
    print("Type /help for repository commands and /exit to quit.")

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

            try:
                handled, active_repository_id = _handle_command(
                    client,
                    prompt,
                    active_repository_id,
                )
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
                print(f"Error: {exc}")
                continue
            if handled:
                continue

            messages.append({"role": "user", "content": prompt})
            print("RepoWren> ", end="", flush=True)
            try:
                answer = _stream_reply(client, messages, active_repository_id)
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
