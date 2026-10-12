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
  /search <query>    Semantically search indexed source chunks
  /help              Show this help
  /exit              Quit"""


def _stream_reply(
    client: httpx.Client,
    messages: list[dict[str, str]],
    repository_id: int | None = None,
) -> str:
    """Print streamed tokens and return the complete answer for chat history.

    For token events `Hello` and ` world`, the user sees text appear in two
    steps while `text_parts` becomes `['Hello', ' world']`. Joining that list
    returns `Hello world`, which is stored as the assistant's history message.
    """
    # Once a repository is selected, use the endpoint that adds source context.
    path = (
        f"/v1/repositories/{repository_id}/chat/stream"
        if repository_id is not None
        else "/v1/chat/stream"
    )
    text_parts: list[str] = []
    with client.stream("POST", path, json={"messages": messages}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            # The server sends one small JSON object per line, allowing output
            # to appear before the model has completed the full response.
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
    """Return the selected ID or explain how the user can select a repository."""
    if repository_id is None:
        raise RuntimeError("Select a repository first with /repo add or /repo use.")
    return repository_id


def _handle_command(
    client: httpx.Client,
    command: str,
    active_repository_id: int | None,
) -> tuple[bool, int | None]:
    """Run one slash command and return whether it consumed the input.

    The pair ``(True, 7)`` means the input was a command and repository 7 is
    now selected. ``(False, 7)`` means it was normal chat text, so `run_chat`
    should send it to the repository-aware chat endpoint.
    """
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
            # Example API object:
            # {"id": 7, "name": "shop", "root_path": "C:/code/shop",
            #  "is_active": true}
            marker = "*" if repository["is_active"] else " "
            print(
                f"{marker} {repository['id']}: {repository['name']} "
                f"({repository['root_path']})"
            )
            if repository["is_active"]:
                active_repository_id = repository["id"]
        return True, active_repository_id

    if command.startswith("/repo add "):
        # Quotes are convenient when a Windows path contains spaces.
        path = command.removeprefix("/repo add ").strip().strip('"')
        if not path:
            raise RuntimeError("Usage: /repo add <path>")
        response = client.post("/v1/repositories", json={"path": path})
        response.raise_for_status()
        repository = response.json()
        if not repository["is_active"]:
            # Registering a later repository does not automatically change the
            # database selection, so select it explicitly for this CLI session.
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
        # "/files src" becomes query "src". Plain "/files" becomes an empty
        # query and asks the API for every safe file.
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
        # Everything after "/read " is a path. Therefore questions must be sent
        # separately; extra words would accidentally become part of the path.
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
            print(
                f"{match['path']}:{match['start_line']}-{match['end_line']} "
                f"(score {match['score']:.3f})\n{match['text']}"
            )
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
    """Run the terminal loop until input ends or the user asks to exit."""
    # Conversation text lives only in this client process. Repository records
    # live in PostgreSQL and therefore survive client restarts.
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

            # Only ordinary prompts enter model history; slash commands do not.
            messages.append({"role": "user", "content": prompt})
            print("RepoWren> ", end="", flush=True)
            try:
                answer = _stream_reply(client, messages, active_repository_id)
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
                messages.pop()
                print(f"\nError: {exc}")
                return 1
            # Example after one turn:
            # [{"role": "user", "content": "Explain cart"},
            #  {"role": "assistant", "content": "The cart..."}]
            messages.append({"role": "assistant", "content": answer})

    return 0


def main() -> int:
    """Read CLI options and start the interactive terminal client."""
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
