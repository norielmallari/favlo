"""The smoke lock closes an idle serve and leaves the process up."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from favlo.smoke_lock import acquire, clear_hold, hold, owner_pid, release


class _Scripted(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    status = 200
    closed = "example-model"

    def log_message(self, fmt: str, *args) -> None:
        del fmt, args

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        self.rfile.read(length)
        if self.path == "/v1/release" and self.status == 200:
            body = json.dumps({"released": True, "closed": self.closed}).encode()
            code = 200
        elif self.path == "/v1/resume":
            body = json.dumps({"resumed": True}).encode()
            code = 200
        else:
            body = json.dumps({"error": {"message": "a request is open", "type": "busy"}}).encode()
            code = 409
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(status: int) -> tuple[ThreadingHTTPServer, int]:
    _Scripted.status = status
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def test_acquire_closes_an_idle_model(tmp_path: Path, capsys) -> None:
    server, port = _serve(200)
    try:
        code = acquire(
            root=tmp_path,
            pipeline="qwen36-35b-a3b",
            command="smoke",
            port=port,
            pid=os.getpid(),
        )
    finally:
        server.shutdown()
        server.server_close()
    text = capsys.readouterr().out
    assert code == 0
    assert "released example-model" in text
    body = json.loads((tmp_path / "artifacts" / "pipeline-test.lock").read_text())
    assert body == {"pipeline": "qwen36-35b-a3b", "pid": os.getpid(), "command": "smoke"}


def test_acquire_leaves_the_lock_when_a_request_is_open(tmp_path: Path) -> None:
    server, port = _serve(409)
    try:
        code = acquire(
            root=tmp_path,
            pipeline="qwen36-35b-a3b",
            command="smoke",
            port=port,
            pid=os.getpid(),
        )
    finally:
        server.shutdown()
        server.server_close()
    assert code == 2
    assert not (tmp_path / "artifacts" / "pipeline-test.lock").exists()


def test_acquire_holds_the_lock_when_serve_is_down(tmp_path: Path) -> None:
    code = acquire(
        root=tmp_path,
        pipeline="qwen38-27b",
        command="kernels",
        port=1,
        pid=os.getpid(),
    )
    assert code == 0
    assert (tmp_path / "artifacts" / "pipeline-test.lock").is_file()


def test_clear_hold_leaves_the_lock_file(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "pipeline-test.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"pipeline": "qwen38-27b", "pid": os.getpid(), "command": "smoke"}))
    server, port = _serve(200)
    try:
        code = clear_hold(port=port, pid=os.getpid())
    finally:
        server.shutdown()
        server.server_close()
    assert code == 0
    assert path.is_file()


def test_release_removes_only_this_pid(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "pipeline-test.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"pipeline": "qwen38-27b", "pid": os.getpid(), "command": "smoke"}))
    server, port = _serve(200)
    try:
        code = release(root=tmp_path, port=port, pid=os.getpid())
    finally:
        server.shutdown()
        server.server_close()
    assert code == 0
    assert not path.exists()


def test_owner_pid_is_the_shell_above_uv(monkeypatch) -> None:
    parents = {10: 9, 9: 8, 8: 1}
    names = {10: "Python", 9: "uv", 8: "zsh"}
    monkeypatch.setattr("favlo.smoke_lock._parent", lambda pid: parents.get(pid, 0))
    monkeypatch.setattr("favlo.smoke_lock._ps", lambda _column, pid: names.get(pid, ""))
    monkeypatch.setattr(os, "getpid", lambda: 10)
    assert owner_pid() == 8


def test_acquire_replaces_a_dead_pid(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "pipeline-test.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"pipeline": "qwen38-27b", "pid": 2**30, "command": "smoke"}))
    code = acquire(
        root=tmp_path,
        pipeline="qwen38-27b",
        command="agent-smoke",
        port=1,
        pid=os.getpid(),
    )
    assert code == 0
    assert json.loads(path.read_text())["pid"] == os.getpid()


def test_hold_leaves_a_lock_this_shell_already_owns(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "artifacts" / "pipeline-test.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"pipeline": "qwen38-27b", "pid": 4242, "command": "smoke"}))
    monkeypatch.setattr("favlo.smoke_lock.owner_pid", lambda: 4242)
    monkeypatch.setattr("favlo.smoke_lock.alive", lambda pid: pid == 4242)
    created = hold(
        root=tmp_path,
        pipeline="qwen38-27b",
        command="file-agent",
        port=1,
        pid=4242,
    )
    assert created is False
    assert json.loads(path.read_text())["command"] == "smoke"


def test_acquire_does_not_steal_a_live_lock(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "pipeline-test.lock"
    path.parent.mkdir()
    path.write_text(json.dumps({"pipeline": "qwen38-27b", "pid": os.getppid(), "command": "smoke"}))
    code = acquire(
        root=tmp_path,
        pipeline="qwen36-35b-a3b",
        command="smoke",
        port=1,
        pid=os.getpid(),
    )
    assert code == 3
    assert json.loads(path.read_text())["pid"] == os.getppid()
