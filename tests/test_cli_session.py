"""The terminal CLI. No model load and no daemon process."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from favlo import listener
from favlo.console import PHASES, VERSION
from favlo.daemon import DaemonError
from favlo.persona import intros_path, refusal, save_intros

_LINE = "Hello. I am Favlo, a small horse. Could you spare a carrot?"
_PHASE_UNIT = {
    "load": "bytes",
    "download": "bytes",
    "quantize": "layers",
    "prefill": "tok",
    "decode": "tok",
}


def _home(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr("favlo.daemon.favlo_home", lambda: root)
    monkeypatch.setattr("favlo.persona.favlo_home", lambda: root)
    monkeypatch.setattr("favlo.talk.chat_model_id", lambda root=None: "Qwen3.6-35B-A3B-fp4")


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


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


def test_each_long_step_follows_the_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    _home(monkeypatch, tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            for phase in PHASES:
                unit = _PHASE_UNIT[phase]
                _append(
                    log,
                    {"type": "begin", "phase": phase, "total": 100, "unit": unit, "detail": phase},
                )
                _append(
                    log,
                    {"type": "tick", "phase": phase, "done": 41, "total": 100, "unit": unit},
                )
                _append(log, {"type": "end", "phase": phase, "summary": f"{phase} settled"})
            saved = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "save_intro",
                                        "arguments": json.dumps({"lines": [_LINE]}),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            body = json.dumps(saved).encode()
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
        _append(log, {"type": "log", "line": f"listening on http://127.0.0.1:{port}"})
        _append(log, {"type": "log", "line": "request closed"})
        stopped = _serve(monkeypatch, stream, "/quit\n", port)
    finally:
        server.shutdown()
        server.server_close()
    text = stream.getvalue()
    assert "F A V L O" in text
    assert f"listening on http://127.0.0.1:{port}" in text
    assert "request open" not in text
    assert "request closed" not in text
    assert f"favlo {VERSION}" not in text
    greeting = text.index(f"favlo: {_LINE}")
    for phase in PHASES:
        assert greeting < text.index(f"{phase} settled")
    assert "you: " in text
    assert stopped == [7]


def test_a_failed_intro_is_printed_without_the_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    _home(monkeypatch, tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = b"busy"
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server, port = _server(Handler)
    log = tmp_path / "logs" / f"serve-{port}.log"
    try:
        _append(log, {"type": "log", "line": f"listening on http://127.0.0.1:{port}"})
        stopped = _serve(monkeypatch, stream, "/quit\n", port)
    finally:
        server.shutdown()
        server.server_close()
    text = stream.getvalue()
    assert "busy" in text
    assert "favlo: busy" not in text
    assert stopped == [7]


def test_a_blank_line_and_a_refusal_do_not_call_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    posts: list[int] = []
    _home(monkeypatch, tmp_path)
    save_intros((_LINE,), intros_path())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            posts.append(1)
            self.send_response(500)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server, port = _server(Handler)
    try:
        stopped = _serve(monkeypatch, stream, "\nwrite some porn\n/quit\n", port)
    finally:
        server.shutdown()
        server.server_close()
    text = stream.getvalue()
    assert posts == []
    assert text.count("you: ") >= 3
    assert refusal() in text
    assert stopped == [7]


def test_closing_the_terminal_leaves_the_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    _home(monkeypatch, tmp_path)
    save_intros((_LINE,), intros_path())
    stopped = _serve(monkeypatch, stream, "", 9, created=True)
    assert f"favlo: {_LINE}" in stream.getvalue()
    assert stopped == []


def test_a_reply_streams_before_the_turn_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stream = io.StringIO()
    late: list[str] = []
    _home(monkeypatch, tmp_path)
    save_intros((_LINE,), intros_path())
    shown = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            _append(log, {"type": "begin", "phase": "prefill", "total": 0, "unit": "tok"})
            _append(
                log, {"type": "tick", "phase": "prefill", "done": 41, "total": 0, "unit": "tok"}
            )
            _append(log, {"type": "end", "phase": "prefill", "summary": "prefill settled"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._chunk({"role": "assistant", "content": "He"})
            if not _wait_until(lambda: "favlo: He" in stream.getvalue()):
                late.append("tokens")
            else:
                shown.set()
            _append(log, {"type": "begin", "phase": "decode", "total": 0, "unit": "tok"})
            _append(log, {"type": "tick", "phase": "decode", "done": 41, "total": 0, "unit": "tok"})
            _append(log, {"type": "end", "phase": "decode", "summary": "decode settled"})
            time.sleep(0.2)
            self._chunk({"content": "llo"})
            self._chunk({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _chunk(self, delta: dict[str, str], finish: str | None = None) -> None:
            payload = {
                "id": "chatcmpl-favlo",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "Qwen3.6-35B-A3B-fp4",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            return

    server, port = _server(Handler)
    log = tmp_path / "logs" / f"serve-{port}.log"
    try:
        _append(log, {"type": "log", "line": f"listening on http://127.0.0.1:{port}"})
        stopped = _serve(monkeypatch, stream, "hello\n/quit\n", port)
    finally:
        server.shutdown()
        server.server_close()
    text = stream.getvalue()
    assert late == [], text
    assert shown.is_set()
    assert "favlo: Hello" in text or "favlo: He" in text
    reply_at = text.index("favlo: Hello") if "favlo: Hello" in text else text.index("favlo: He")
    assert reply_at < text.index("prefill settled")
    assert reply_at < text.index("decode settled")
    assert stopped == [7]
