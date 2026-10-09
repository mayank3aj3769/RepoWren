"""Tests for the terminal chat client."""

import httpx

from repowren.cli import run_chat


def test_terminal_chat_prints_streamed_reply(
    monkeypatch,
    capsys,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/stream"
        return httpx.Response(
            200,
            text=(
                '{"type":"token","text":"Hello"}\n'
                '{"type":"token","text":" there"}\n'
                '{"type":"done"}\n'
            ),
        )

    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr("repowren.cli.httpx.Client", client_factory)

    exit_code = run_chat("http://testserver", prompts=["hello", "/exit"])

    assert exit_code == 0
    assert "RepoWren> Hello there" in capsys.readouterr().out


def test_terminal_chat_reports_api_failure(monkeypatch, capsys) -> None:
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda _: httpx.Response(503, text="unavailable")
        )
        return original_client(*args, **kwargs)

    monkeypatch.setattr("repowren.cli.httpx.Client", client_factory)

    exit_code = run_chat("http://testserver", prompts=["hello"])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().out


def test_repository_commands_select_and_use_repository_chat(
    monkeypatch,
    capsys,
) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/v1/repositories":
            return httpx.Response(
                201,
                json={
                    "id": 7,
                    "name": "sample",
                    "root_path": "C:/code/sample",
                    "is_active": True,
                },
            )
        if request.url.path == "/v1/repositories/7/files":
            return httpx.Response(
                200,
                json=[
                    {
                        "path": "app.py",
                        "size_bytes": 42,
                        "modified_ns": 1,
                        "sha256": "a" * 64,
                    }
                ],
            )
        if request.url.path == "/v1/repositories/7/chat/stream":
            return httpx.Response(
                200,
                text='{"type":"token","text":"Found it"}\n{"type":"done"}\n',
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr("repowren.cli.httpx.Client", client_factory)

    exit_code = run_chat(
        "http://testserver",
        prompts=["/repo add C:/code/sample", "/files", "Explain app.py", "/exit"],
    )

    assert exit_code == 0
    assert requested_paths[-1] == "/v1/repositories/7/chat/stream"
    output = capsys.readouterr().out
    assert "Using repository 7: sample" in output
    assert "app.py (42 bytes)" in output
    assert "RepoWren> Found it" in output
