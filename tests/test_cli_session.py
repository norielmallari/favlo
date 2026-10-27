"""The operator console. No model load and no daemon process."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from favlo import listener
from favlo.console import VERSION
from favlo.daemon import DaemonError


def _home(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr("favlo.daemon.favlo_home", lambda: root)


def _append(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _server(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    assert isinstance(port, int)
    return server, port


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    stream: io.StringIO,
    stdin: str,
    port: int,
    *,
    created: bool = True,
) -> list[int]:
    stopped: list[int] = []
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    monkeypatch.setattr("favlo.setup.prepare", lambda *_args, **_kwargs: 0)

    def ensure(*_args: object, **_kwargs: object) -> tuple[int, bool]:
        return 7, created

    monkeypatch.setattr("favlo.daemon.ensure", ensure)
    monkeypatch.setattr("favlo.daemon.stop_pid", lambda pid: stopped.append(pid))
    assert listener.serve_main(["--port", str(port), "--console", "pretty"]) == 0
    return stopped


def test_a_daemon_error_is_printed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stream = io.StringIO()
    _home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    def fail(*_args: object, **_kwargs: object) -> tuple[int, bool]:
        raise DaemonError("not listening")

    monkeypatch.setattr("favlo.daemon.ensure", fail)
    assert listener.serve_main(["--console", "pretty"]) == 1
    assert "not listening" in stream.getvalue()
    assert "F A V L O" in stream.getvalue()


def test_setup_failure_does_not_open_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("favlo.setup.prepare", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr("favlo.daemon.ensure", lambda *_args, **_kwargs: called.append(1))
    assert listener.serve_main([]) == 1
    assert called == []


def test_attach_paints_telemetry_and_does_not_greet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    posts: list[str] = []
    _home(monkeypatch, tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            posts.append(self.path)
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server, port = _server(Handler)
    log = tmp_path / "logs" / f"serve-{port}.log"
    try:
        _append(log, {"type": "log", "line": f"favlo {VERSION}"})
        _append(log, {"type": "log", "line": "request open"})
        _append(log, {"type": "log", "line": "load settled"})
        _append(log, {"type": "log", "line": f"listening on http://127.0.0.1:{port}"})
        _append(log, {"type": "log", "line": "request closed"})
        stopped = _serve(monkeypatch, stream, "hello\n/quit\n", port)
    finally:
        server.shutdown()
        server.server_close()
    text = stream.getvalue()
    assert posts == []
    assert "F A V L O" in text
    assert f"listening on http://127.0.0.1:{port}" in text
    assert "load settled" in text
    assert "request open" not in text
    assert "request closed" not in text
    assert f"favlo {VERSION}" not in text
    assert "you:" not in text
    assert "favlo:" not in text
    assert stopped == [7]


def test_reload_stays_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    posts: list[str] = []
    _home(monkeypatch, tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            posts.append(self.path)
            body = b'{"unloaded": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server, port = _server(Handler)
    try:
        stopped = _serve(monkeypatch, stream, "reload\nquit\n", port)
    finally:
        server.shutdown()
        server.server_close()
    text = stream.getvalue()
    assert posts == ["/v1/pipelines/reload"]
    assert "reloaded" in text
    assert "favlo:" not in text
    assert "you:" not in text
    assert stopped == [7]


def test_quit_does_not_wait_on_a_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    posts: list[str] = []
    _home(monkeypatch, tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            posts.append(self.path)
            threading.Event().wait(30)

        def log_message(self, format: str, *args: object) -> None:
            return

    server, port = _server(Handler)
    box: dict[str, list[int]] = {}

    def run() -> None:
        box["stopped"] = _serve(monkeypatch, stream, "quit\n", port)

    thread = threading.Thread(target=run)
    try:
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        assert box["stopped"] == [7]
        assert posts == []
    finally:
        thread.join(2)
        server.shutdown()
        server.server_close()


def test_closing_the_terminal_leaves_the_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    _home(monkeypatch, tmp_path)
    stopped = _serve(monkeypatch, stream, "", 9, created=True)
    text = stream.getvalue()
    assert "F A V L O" in text
    assert "favlo:" not in text
    assert stopped == []
