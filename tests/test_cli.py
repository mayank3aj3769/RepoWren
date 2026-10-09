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
