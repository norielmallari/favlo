"""The serve daemon starts once and the terminal does not become that process."""

import io
import sys
import threading
import time
from pathlib import Path

import pytest

from favlo import listener
from favlo.daemon import (
    DAEMON_ENV,
    DaemonError,
    ensure,
    follow_events,
    plain_text,
    read_pid,
    stop_pid,
    telemetry_event,
)


class _Proc:
    def __init__(self, pid: int, code: int | None) -> None:
        self.pid = pid
        self._code = code

    def poll(self) -> int | None:
        return self._code


def test_plain_text_keeps_log_lines_and_summaries() -> None:
    blob = "\n".join(
        [
            '{"v":1,"type":"log","line":"listening on http://127.0.0.1:8129"}',
            '{"v":1,"type":"tick","phase":"prefill","done":1}',
            '{"v":1,"type":"end","summary":"prefill 10.00 tok/s (2 tok in 0.20s)"}',
            "decode 3.00 tok/s (1 tok in 0.33s)",
        ]
    )
    text = plain_text(blob)
    assert "listening on http://127.0.0.1:8129" in text
    assert "prefill 10.00 tok/s" in text
    assert text.splitlines()[-1].startswith("decode ")
    assert "tick" not in text


def test_follow_events_keeps_ticks(tmp_path: Path) -> None:
    raw = '{"type":"tick","phase":"load","done":3}'
    assert telemetry_event(raw) == {"type": "tick", "phase": "load", "done": 3}
    assert plain_text(raw) == ""
    path = tmp_path / "serve.log"
    path.write_text(raw + "\n", encoding="utf-8")
    seen: list[dict[str, object]] = []
    stop = threading.Event()
    follow_events(path, seen.append, stop, offset=0)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not seen:
        time.sleep(0.05)
    stop.set()
    assert seen and seen[0]["type"] == "tick"
    assert seen[0]["done"] == 3


def test_a_shutdown_cancel_starts_a_response() -> None:
    import asyncio

    from favlo.http_app import _QuietCancel

    sent: list[dict[str, object]] = []

    async def app(_scope: object, _receive: object, _send: object) -> None:
        raise asyncio.CancelledError

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    asyncio.run(_QuietCancel(app)({"type": "http"}, None, send))
    assert sent[0]["status"] == 503
    assert sent[1]["type"] == "http.response.body"


def test_follow_events_skips_a_shutdown_traceback(tmp_path: Path) -> None:
    path = tmp_path / "serve.log"
    path.write_text(
        "\n".join(
            [
                "Cancel 1 running task(s), timeout graceful shutdown exceeded",
                "Exception in ASGI application",
                "Traceback (most recent call last):",
                '  File "src/favlo/http_app.py", line 388, in _locked',
                "    await asyncio.to_thread(reply.ready.wait)",
                "    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
                "asyncio.exceptions.CancelledError: Task cancelled, timeout graceful shutdown exceeded",
                '{"type":"log","line":"client closed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seen: list[dict[str, object]] = []
    stop = threading.Event()
    follow_events(path, seen.append, stop, offset=0)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not seen:
        time.sleep(0.05)
    stop.set()
    assert [event.get("line") for event in seen] == ["client closed"]


def test_a_foreign_listener_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_args, **_kwargs: True)
    with pytest.raises(DaemonError, match="already in use"):
        ensure("127.0.0.1", 8129, home=tmp_path, root=tmp_path)


def test_ensure_starts_one_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[int] = []
    open_port = {"up": False}

    def spawn(*_args, **_kwargs) -> _Proc:
        open_port["up"] = True
        started.append(4242)
        return _Proc(4242, None)

    monkeypatch.setattr("favlo.daemon._spawn", spawn)
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_a, **_k: open_port["up"])
    monkeypatch.setattr("favlo.daemon.alive", lambda pid: pid == 4242)
    pid, created = ensure("127.0.0.1", 8129, home=tmp_path, root=tmp_path)
    assert (pid, created) == (4242, True)
    assert read_pid(8129, tmp_path) == 4242
    again, started_again = ensure("127.0.0.1", 8129, home=tmp_path, root=tmp_path)
    assert (again, started_again) == (4242, False)
    assert started == [4242]


def test_a_daemon_that_exits_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.daemon._spawn", lambda *_a, **_k: _Proc(4242, 1))
    monkeypatch.setattr("favlo.daemon.port_open", lambda *_a, **_k: False)
    monkeypatch.setattr("favlo.daemon.alive", lambda _pid: False)
    monkeypatch.setattr("favlo.daemon.stop_pid", lambda _pid: None)
    with pytest.raises(DaemonError, match="did not start"):
        ensure("127.0.0.1", 8129, home=tmp_path, root=tmp_path)
    assert read_pid(8129, tmp_path) is None


def test_stop_pid_ignores_a_dead_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("favlo.daemon.alive", lambda _pid: False)
    monkeypatch.setattr(
        "favlo.daemon._signal",
        lambda pid, _signum: calls.append(pid),
    )
    stop_pid(123)
    assert calls == []


def test_the_daemon_starts_a_pipeline_heartbeat() -> None:
    from types import SimpleNamespace

    from favlo.listener import _schedule_pipeline_heartbeat

    seen: list[tuple[object, object]] = []

    def pump(runtime: object, selection: object) -> None:
        seen.append((runtime, selection))

    _schedule_pipeline_heartbeat(SimpleNamespace(idle_heartbeat=pump), "runtime", "row")
    deadline = time.monotonic() + 1
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == [("runtime", "row")]
    before = {thread.name for thread in threading.enumerate()}
    _schedule_pipeline_heartbeat(SimpleNamespace(), "runtime", "row")
    assert {thread.name for thread in threading.enumerate()} == before


def test_non_interactive_reports_a_daemon_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_args: object, **_kwargs: object) -> tuple[int, bool]:
        raise DaemonError("not listening")

    monkeypatch.setattr("favlo.daemon.ensure", fail)
    assert listener.serve_main(["--non-interactive"]) == 1
    assert "not listening" in capsys.readouterr().err


def test_non_interactive_starts_the_daemon_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("favlo.daemon.ensure", lambda *_a, **_k: (7, True))
    assert listener.serve_main(["--non-interactive"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "listening on http://127.0.0.1:8129\n"
    assert captured.err == ""


def test_a_terminal_is_the_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    seen: list[str] = []
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(listener, "attach_process_log", lambda _ui: None)
    monkeypatch.setattr("favlo.daemon.occupy", lambda *_a, **_k: None)
    monkeypatch.setattr("favlo.daemon.release_pid", lambda *_a, **_k: None)

    def serve(args: object) -> int:
        seen.append(getattr(args, "console_ui").mode)
        return 0

    monkeypatch.setattr(listener, "serve", serve)
    assert listener.serve_main(["--port", "8129", "--console", "plain"]) == 0
    assert seen == ["plain"]
    assert "favlo 0.3.0\n" in stream.getvalue()


def test_a_taken_address_does_not_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def taken(*_args: object, **_kwargs: object) -> None:
        raise DaemonError("http://127.0.0.1:8129 is already in use")

    monkeypatch.setattr("favlo.daemon.occupy", taken)
    assert listener.serve_main(["--port", "8129"]) == 1
    assert "already in use" in capsys.readouterr().err


def test_daemon_process_does_not_spawn_another(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(DAEMON_ENV, "1")
    monkeypatch.setattr(listener, "attach_process_log", lambda _ui: None)
    monkeypatch.setattr(listener, "serve", lambda _args: 0)
    assert listener.serve_main(["--port", "9"]) == 0
    captured = capsys.readouterr()
    assert "listening on" not in captured.out
