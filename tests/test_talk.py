"""The operator console. No model and no socket."""

from __future__ import annotations

import io

import pytest

from favlo.talk import is_quit, is_reload, start


def test_quit_and_reload_are_the_commands() -> None:
    assert is_quit("quit")
    assert is_quit("/quit")
    assert not is_quit("quit now")
    assert is_reload("reload")
    assert is_reload("/reload")
    assert not is_reload("reload now")


def test_a_line_that_is_not_a_command_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[int] = []
    lines: list[str] = []
    prompts: list[bool] = []

    def refuse(*_args: object) -> None:
        raise AssertionError("the daemon was called")

    monkeypatch.setattr("favlo.talk.pipeline_command", refuse)
    thread = start(
        "127.0.0.1",
        9,
        io.StringIO("\nhello\nquit\n"),
        lines.append,
        lambda: stopped.append(1),
        prompt=prompts.append,
    )
    thread.join(2)
    assert not thread.is_alive()
    assert lines == []
    assert stopped == [1]
    assert prompts[0] is False


def test_reload_stays_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    lines: list[str] = []
    stopped: list[int] = []

    def reload(host: str, port: int, action: str) -> dict[str, bool]:
        assert (host, port) == ("127.0.0.1", 9)
        calls.append(action)
        return {"ok": True}

    monkeypatch.setattr("favlo.talk.pipeline_command", reload)
    thread = start(
        "127.0.0.1",
        9,
        io.StringIO("/reload\n/quit\n"),
        lines.append,
        lambda: stopped.append(1),
        prompt=lambda _after: None,
    )
    thread.join(2)
    assert calls == ["reload"]
    assert lines == ["reloaded"]
    assert stopped == [1]


def test_a_failed_reload_is_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []

    def fail(*_args: object) -> None:
        raise RuntimeError("busy")

    monkeypatch.setattr("favlo.talk.pipeline_command", fail)
    thread = start(
        "127.0.0.1",
        9,
        io.StringIO("reload\nquit\n"),
        lines.append,
        lambda: None,
        prompt=lambda _after: None,
    )
    thread.join(2)
    assert lines == ["busy"]
