"""One request at a time, a dropped client, and signal shutdown."""

import base64
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import qwen38_flash_next.commands as commands
from qwen38_flash_next.runtime import RequestClosed

from favlo import listener


class _Encoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _Tokenizer:
    def encode(self, _text: str) -> _Encoded:
        return _Encoded([1])


class _Runtime:
    """Stand-in forward. The server still owns the socket and the queue."""

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.release = threading.Event()
        self.entered = threading.Event()
        self.canceled = threading.Event()
        self.lock = threading.Lock()
        self.order: list[tuple[str, str]] = []
        self._name = ""
        self._stop = None
        self._hit_limit = False
        self.tool_names: list[str] = []
        self.required_function = None
        self.required_closes = False
        self.last_logits = None
        self.closed = False
        self.halted = False
        self.watch_cancel = False
        self.hold = False
        self.persisted = 0

    def encode_chat(
        self,
        messages,
        tools,
        enable_thinking,
        reasoning_effort=None,
        continue_final=False,
    ):
        self._name = messages[0]["content"]
        return [1]

    def generate(self, prompt, max_tokens, temperature, top_p, top_k):
        if self.watch_cancel:
            self._wait_for_cancel()
        self._mark("start")
        self.entered.set()
        if self.hold:
            self._hold()
        self._mark("end")
        return [1]

    def align(self, prompt):
        return 0

    def _prompt_logits(self, prompt):
        return self.forward(prompt)

    def forward(self, tokens):
        if self.watch_cancel:
            self._wait_for_cancel()
        return [1]

    def _new_tokens(self, logits, max_tokens, temperature, top_p, top_k):
        yield 1

    def decode(self, tokens):
        return "ok"

    def _log_decode(self) -> None:
        return

    def halt(self) -> None:
        self.halted = True
        self._stop = lambda: True

    def close(self) -> None:
        self.closed = True
        self.release.set()

    def _persist_prefix(self) -> None:
        self.persisted += 1

    def _hold(self) -> None:
        deadline = time.monotonic() + 5
        while not self.release.is_set() and time.monotonic() < deadline:
            if self.halted:
                raise RequestClosed()
            time.sleep(0.02)

    def _mark(self, kind: str) -> None:
        with self.lock:
            self.order.append((kind, self._name))

    def _wait_for_cancel(self) -> None:
        self.watch_cancel = False
        self.entered.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            stop = self._stop
            if stop is not None and stop():
                self.canceled.set()
                raise RequestClosed()
            time.sleep(0.02)
        message = "dropped client did not stop the request"
        raise AssertionError(message)


def _args() -> SimpleNamespace:
    return SimpleNamespace(host="127.0.0.1", port=0, transcript=None)


def _request(content: str, *, stream: bool) -> bytes:
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "enable_thinking": False,
        "stream": stream,
    }
    body = json.dumps(payload).encode()
    head = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return head.encode() + body


def _exchange(port: int, content: str, *, stream: bool = False, on_sent=None) -> bytes:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(5)
    try:
        sock.sendall(_request(content, stream=stream))
        if on_sent is not None:
            on_sent()
        chunks: list[bytes] = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    finally:
        sock.close()
    return b"".join(chunks)


def _open_request(port: int, content: str, *, stream: bool) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(5)
    sock.sendall(_request(content, stream=stream))
    return sock


def _serve_background(runtime: _Runtime):
    state: dict = {"runtime": runtime, "server": None, "module": commands}
    thread = threading.Thread(
        target=listener._serve_loop,
        args=(_args(), state),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while state["server"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    server = state["server"]
    assert server is not None
    return state, thread


def _stop_background(state, thread) -> None:
    server = state["server"]
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()
    assert not thread.is_alive()


def _port_closed(port: int) -> bool:
    probe = socket.socket()
    try:
        probe.connect(("127.0.0.1", port))
    except OSError:
        return True
    finally:
        probe.close()
    return False


def _serial(order: list[tuple[str, str]]) -> list[str]:
    names: list[str] = []
    assert len(order) % 2 == 0
    for index in range(0, len(order), 2):
        start, end = order[index], order[index + 1]
        assert start[0] == "start"
        assert end == ("end", start[1])
        names.append(start[1])
    return names


def test_a_silent_client_releases_the_request_slot(monkeypatch) -> None:
    monkeypatch.setattr(listener, "_REQUEST_READ_TIMEOUT", 0.2)
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    stuck = socket.create_connection(("127.0.0.1", port), timeout=2)
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        sock.settimeout(2)
        try:
            sock.sendall(
                b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
            )
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
    finally:
        stuck.close()
        _stop_background(state, thread)
    assert b"200" in data.split(b"\r\n", 1)[0]


def test_an_idle_kept_alive_connection_does_not_hold_the_slot() -> None:
    """An editor that probes the port and keeps the socket open does not block a chat."""
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    idle = socket.create_connection(("127.0.0.1", port), timeout=2)
    silent = socket.create_connection(("127.0.0.1", port), timeout=2)
    try:
        idle.settimeout(2)
        idle.sendall(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        assert idle.recv(4096).startswith(b"HTTP/1.1 200")
        started = time.monotonic()
        body = _exchange(port, "after the probe")
        elapsed = time.monotonic() - started
    finally:
        idle.close()
        silent.close()
        _stop_background(state, thread)
    assert body.startswith(b"HTTP/1.1 200")
    assert elapsed < 1.0


def test_the_model_list_answers_while_a_completion_is_open(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(listener, "_log_sink", lines.append)
    runtime = _Runtime()
    runtime.hold = True
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    worker = threading.Thread(target=_exchange, args=(port, "held"))
    second = threading.Thread(target=_exchange, args=(port, "queued"))
    try:
        worker.start()
        assert runtime.entered.wait(3)
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        sock.settimeout(2)
        try:
            sock.sendall(
                b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
            )
            head = sock.recv(4096)
        finally:
            sock.close()
        second.start()
        deadline = time.monotonic() + 3
        while "waiting for the open request" not in lines and time.monotonic() < deadline:
            time.sleep(0.02)
        runtime.release.set()
        worker.join(timeout=5)
        second.join(timeout=5)
    finally:
        runtime.release.set()
        _stop_background(state, thread)
    assert head.startswith(b"HTTP/1.1 200")
    assert "waiting for the open request" in lines


def test_requests_wait_until_the_current_one_finishes() -> None:
    runtime = _Runtime()
    runtime.hold = True
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    sent = {name: threading.Event() for name in ("one", "two", "three")}
    results: dict[str, bytes] = {}

    def run(name: str) -> None:
        results[name] = _exchange(port, name, on_sent=sent[name].set)

    workers = [threading.Thread(target=run, args=(name,)) for name in ("one", "two", "three")]
    try:
        workers[0].start()
        assert runtime.entered.wait(3)
        workers[1].start()
        workers[2].start()
        assert sent["two"].wait(3)
        assert sent["three"].wait(3)
        with runtime.lock:
            assert runtime.order == [("start", "one")]
        runtime.release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive()
        assert _serial(runtime.order)[0] == "one"
        assert sorted(_serial(runtime.order)) == ["one", "three", "two"]
        for name in ("one", "two", "three"):
            assert results[name].startswith(b"HTTP/1.1 200")
            assert b'"content": "ok"' in results[name]
    finally:
        runtime.release.set()
        _stop_background(state, thread)


def test_dropped_client_stops_the_request_and_the_next_one_runs() -> None:
    runtime = _Runtime()
    runtime.watch_cancel = True
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    try:
        sock = _open_request(port, "drop", stream=True)
        assert runtime.entered.wait(3)
        sock.close()
        assert runtime.canceled.wait(3)
        follow = _exchange(port, "next")
        assert follow.startswith(b"HTTP/1.1 200")
        assert b'"content": "ok"' in follow
        assert runtime.persisted == 1
        with runtime.lock:
            assert runtime.order == [("start", "next"), ("end", "next")]
    finally:
        runtime.release.set()
        _stop_background(state, thread)


def test_closed_post_stops_generation() -> None:
    runtime = _Runtime()
    runtime.watch_cancel = True
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    try:
        sock = _open_request(port, "drop", stream=False)
        assert runtime.entered.wait(3)
        sock.close()
        assert runtime.canceled.wait(3)
        follow = _exchange(port, "after")
        assert follow.startswith(b"HTTP/1.1 200")
        assert b'"content": "ok"' in follow
        assert runtime.persisted == 1
    finally:
        runtime.release.set()
        _stop_background(state, thread)


def test_a_job_reply_is_frozen_when_json_returns() -> None:
    from favlo.reply import LiveReply

    gone = threading.Event()
    reply = LiveReply(gone)
    job = {"status": "in_progress"}
    reply.json(200, job)
    job["status"] = "completed"
    assert reply.encoded == b'{"status": "in_progress"}'
    assert reply.payload == {"status": "completed"}
    gone.set()
    assert reply.stopped() is False


def test_a_stream_stops_when_the_client_is_gone() -> None:
    from favlo.reply import LiveReply

    gone = threading.Event()
    reply = LiveReply(gone)
    reply.begin_events()
    assert reply.stopped() is False
    gone.set()
    assert reply.stopped() is True


def _signal_server(sig: int, *, during_request: bool) -> None:
    runtime = _Runtime()
    runtime.hold = during_request
    shutdown = listener._Shutdown()
    shutdown.runtime = runtime
    state: dict = {"runtime": runtime, "server": None, "module": commands}
    ports: list[int] = []
    done = threading.Event()

    def poke() -> None:
        deadline = time.monotonic() + 5
        while state["server"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        server = state["server"]
        if server is None:
            return
        ports.append(int(server.server_address[1]))
        if during_request:
            assert runtime.entered.wait(3)
        else:
            time.sleep(0.05)
        os.kill(os.getpid(), sig)

    def failsafe() -> None:
        if done.wait(4):
            return
        server = state["server"]
        if server is not None:
            server.shutdown()

    shutdown.arm()
    threading.Thread(target=failsafe, daemon=True).start()
    threading.Thread(target=poke, daemon=True).start()
    client = None
    if during_request:

        def post() -> None:
            deadline = time.monotonic() + 5
            while not ports and time.monotonic() < deadline:
                time.sleep(0.01)
            if not ports:
                return
            try:
                _exchange(ports[0], "held")
            except OSError:
                return

        client = threading.Thread(target=post)
        client.start()
    try:
        code = listener._run_until_stopped(_args(), state, shutdown)
    finally:
        done.set()
        if client is not None:
            client.join(timeout=5)
        if not shutdown.closing:
            listener._release(runtime, shutdown, state.get("server"))
    assert code == 0
    assert shutdown.signaled
    assert runtime.halted
    assert runtime.closed
    assert ports
    assert _port_closed(ports[0])
    if during_request:
        assert ("end", "held") not in runtime.order


def test_usage_excludes_the_cached_prefix() -> None:
    runtime = SimpleNamespace(_cached_tokens=900, _prefill_seconds=2.0, _decode_seconds=4.0)
    usage = commands._usage(runtime, 1187, 480, commands.DEFAULT_MAX_PROMPT_TOKENS)
    timings = usage["timings"]
    assert usage["prompt_tokens"] == 1187
    assert timings["prompt_n"] == 287
    assert timings["predicted_n"] == 480
    assert timings["prompt_per_second"] == 143.5
    assert timings["predicted_per_second"] == 120.0
    assert usage["max_prompt_tokens"] == commands.DEFAULT_MAX_PROMPT_TOKENS
    assert usage["context_length"] == commands.DEFAULT_CONTEXT_LENGTH
    assert usage["prompt_tokens_details"]["cached_tokens"] == 900
    idle = commands._usage(SimpleNamespace(), 10, 4)
    assert idle["timings"]["prompt_per_second"] == 0.0
    assert idle["timings"]["predicted_per_second"] == 0.0
    assert idle["timings"]["prompt_n"] == 10


def test_lazy_stream_uses_the_runtime_loaded_for_the_request(monkeypatch) -> None:
    runtime = _Runtime()
    state: dict = {
        "runtime": None,
        "server": None,
        "record": None,
        "requests": 0,
        "module": commands,
    }

    def bind(body, bound_state, bound_args):
        del body, bound_args
        bound_state["runtime"] = runtime
        bound_state["record"] = "packed"
        return SimpleNamespace(requested_id="packed", enable_thinking=None, sampling=None)

    monkeypatch.setattr(listener, "_bind_request", bind)
    args = _args()
    args.lazy = True
    thread = threading.Thread(target=listener._serve_loop, args=(args, state), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while state["server"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    server = state["server"]
    assert server is not None
    port = int(server.server_address[1])
    try:
        body = _exchange(port, "lazy", stream=True)
    finally:
        _stop_background(state, thread)
    assert body.startswith(b"HTTP/1.1 200")
    assert b"data: [DONE]" in body
    assert b"AttributeError" not in body


def test_streamed_completion_reports_timings() -> None:
    runtime = _Runtime()
    runtime._cached_tokens = 1
    runtime._prefill_seconds = 2.0
    runtime._decode_seconds = 4.0
    runtime._output_ids = [7, 8]
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    try:
        body = _exchange(port, "timed", stream=True)
    finally:
        _stop_background(state, thread)
    assert body.startswith(b"HTTP/1.1 200")
    assert b'"prompt_n": 0' in body
    assert b'"predicted_n": 2' in body
    assert b'"predicted_per_second": 0.5' in body
    assert b"data: [DONE]" in body


def test_sigint_during_a_request_releases_the_server() -> None:
    import signal

    _signal_server(signal.SIGINT, during_request=True)


def test_sigterm_releases_the_listening_socket() -> None:
    import signal

    _signal_server(signal.SIGTERM, during_request=False)


def test_a_different_pipeline_closes_the_resident_runtime(monkeypatch) -> None:
    opened: list[_Runtime] = []

    def opener(selection, console, max_memory=None):
        del console, max_memory
        if opened:
            assert opened[-1].closed
        runtime = _Runtime()
        runtime._name = selection.pipeline
        opened.append(runtime)
        return runtime

    monkeypatch.setattr(listener, "_open_runtime", opener)
    state: dict = {"runtime": None, "record": None, "pipeline": None}
    args = _args()
    first = SimpleNamespace(
        folder="flash",
        pipeline="qwen38-flash-next",
        runtime="metal41",
        custom=False,
        alias=None,
        resident_percent=None,
        kv_cache="e4m3",
        path="flash",
    )
    second = SimpleNamespace(
        folder="dense",
        pipeline="qwen38-27b",
        runtime="metal41",
        custom=False,
        alias="qwen38-27b",
        resident_percent=None,
        kv_cache="e4m3",
        path="dense",
    )
    listener._install_runtime(state, args, first)
    listener._install_runtime(state, args, first)
    assert len(opened) == 1
    assert state["pipeline"] == "qwen38-flash-next"
    listener._install_runtime(state, args, second)
    assert opened[0].closed
    assert state["runtime"] is opened[1]
    assert state["record"] == "dense"
    assert state["pipeline"] == "qwen38-27b"


def _post_json(port: int, path: str, payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    head = (
        f"POST {path} HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    )
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.settimeout(5)
    try:
        sock.sendall(head.encode() + body)
        chunks: list[bytes] = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    finally:
        sock.close()
    return b"".join(chunks)


def test_an_idle_release_closes_the_model_and_holds_chat() -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["record"] = "example-model"
    port = int(state["server"].server_address[1])
    try:
        released = _post_json(port, "/v1/release", {"pid": os.getpid()})
        refused = _exchange(port, "later")
    finally:
        _stop_background(state, thread)
    assert released.startswith(b"HTTP/1.1 200")
    assert b'"closed": "example-model"' in released
    assert runtime.closed
    assert state["runtime"] is None
    assert refused.startswith(b"HTTP/1.1 503")


def test_a_release_waits_while_a_request_is_open() -> None:
    runtime = _Runtime()
    runtime.hold = True
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    worker = threading.Thread(target=_exchange, args=(port, "held"))
    try:
        worker.start()
        assert runtime.entered.wait(3)
        released = _post_json(port, "/v1/release", {"pid": os.getpid()})
        runtime.release.set()
        worker.join(timeout=5)
    finally:
        runtime.release.set()
        _stop_background(state, thread)
    assert released.startswith(b"HTTP/1.1 409")
    assert runtime.closed is False


def test_resume_clears_the_smoke_hold() -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    try:
        _post_json(port, "/v1/release", {"pid": os.getpid()})
        resumed = _post_json(port, "/v1/resume", {"pid": os.getpid()})
        refused = _exchange(port, "again")
    finally:
        _stop_background(state, thread)
    assert resumed.startswith(b"HTTP/1.1 200")
    assert state["smoke_hold"] is None
    assert refused.startswith(b"HTTP/1.1 404")


def _http(
    port: int,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    raw: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, str, bytes]:
    data = raw if raw is not None else None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": content_type} if data is not None else {}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            kind = response.headers.get("Content-Type", "")
            return response.status, kind, response.read()
    except urllib.error.HTTPError as exc:
        kind = exc.headers.get("Content-Type", "")
        return exc.code, kind, exc.read()


def _meta(*, tasks: bool) -> dict[str, Any]:
    extensions: dict[str, Any] = {}
    if tasks:
        extensions["io.modelcontextprotocol/tasks"] = {}
    return {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {"extensions": extensions},
        "io.modelcontextprotocol/clientInfo": {"name": "favlo-test", "version": "0"},
    }


def _mcp(
    port: int,
    method: str,
    params: dict | None,
    *,
    rpc_id: int = 1,
    tasks: bool = False,
    name: str | None = None,
) -> tuple[int, Any]:
    body = dict(params or {})
    body["_meta"] = _meta(tasks=tasks)
    payload = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": body}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": method,
    }
    if name is not None:
        headers["mcp-name"] = name
    return _mcp_raw(port, json.dumps(payload).encode(), headers)


def _mcp_raw(port: int, raw: bytes, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=raw,
        headers=headers
        or {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-protocol-version": "2026-07-28",
            "mcp-method": "tools/list",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    try:
        parsed = json.loads(body.decode())
    except json.JSONDecodeError:
        parsed = body.decode()
    return status, parsed


def _tool(
    port: int,
    name: str,
    arguments: dict,
    *,
    tasks: bool = True,
    rpc_id: int = 1,
) -> tuple[int, Any]:
    return _mcp(
        port,
        "tools/call",
        {"name": name, "arguments": arguments},
        rpc_id=rpc_id,
        tasks=tasks,
        name=name,
    )


def _task_params(task_id: str) -> dict[str, Any]:
    return {"taskId": task_id}


def _wait_task(port: int, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    last: Any = None
    while time.monotonic() < deadline:
        _status, body = _mcp(port, "tasks/get", _task_params(task_id), tasks=True)
        last = body
        result = body.get("result") if isinstance(body, dict) else None
        done = {"completed", "failed", "cancelled"}
        if isinstance(result, dict) and result.get("status") in done:
            return result
        time.sleep(0.02)
    message = f"task {task_id} did not finish: {last}"
    raise AssertionError(message)


def _store_png(home: Path, image_id: str, payload: bytes) -> None:
    from favlo.images import image_path

    path = image_path(home, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class _QueueMedia:
    """Image handler the queue tests drive. It does not touch a GPU."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.order: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.first_file = threading.Event()
        self.hold_release = threading.Event()
        self.log: list[str] = []
        self.saw_a = threading.Event()
        self.chat_entered = threading.Event()
        self.chat_release = threading.Event()
        self.video_called = False

    def handle_chat(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, body, selection, args
        self.log.append("chat")
        self.chat_entered.set()
        self.chat_release.wait(5)
        reply.json(200, {"id": "chat", "choices": []})

    def handle_image(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args
        prompt = body.get("prompt")
        if prompt == "hold":
            self.order.append("start")
            self.entered.set()
            self.release.wait(5)
            _store_png(self.home, "image_hold", b"hold")
            self.order.append("end")
            reply.json(200, {"data": [{"id": "image_hold", "b64_json": "aG9sZA=="}]})
            return
        if prompt == "later":
            self.order.append("later")
            _store_png(self.home, "image_later", b"later")
            reply.json(200, {"data": [{"id": "image_later", "b64_json": "bGF0ZXI="}]})
            return
        if prompt == "batch":
            wrote = []
            for index in range(int(body.get("n") or 1)):
                if state["cancel"].is_set():
                    reply.json(200, {"status": "in_progress", "data": wrote})
                    return
                image_id = f"image_{index}"
                _store_png(self.home, image_id, b"batch")
                wrote.append({"id": image_id, "b64_json": "YmF0Y2g="})
                if index == 0:
                    self.first_file.set()
                    self.hold_release.wait(5)
            reply.json(200, {"data": wrote})
            return
        if prompt == "pair":
            _store_png(self.home, "image_a", b"aaa")
            _store_png(self.home, "image_b", b"bbb")
            reply.json(
                200,
                {
                    "data": [
                        {"id": "image_a", "b64_json": "YWFh"},
                        {"id": "image_b", "b64_json": "YmJi"},
                    ]
                },
            )
            return
        if prompt == "big":
            _store_png(self.home, "image_big", b"x" * (1024 * 1024 + 1))
            reply.json(200, {"data": [{"id": "image_big", "b64_json": "eA=="}]})
            return
        if prompt == "preempt":
            if "a" not in self.log:
                self.log.append("a")
                self.saw_a.set()
                state["preempt"].wait(5)
                reply.json(200, {"status": "in_progress", "stages": ["a"]})
                return
            self.log.append("b")
            _store_png(self.home, "image_preempt", b"pre")
            reply.json(
                200,
                {
                    "status": "completed",
                    "stages": ["a", "b"],
                    "data": [{"id": "image_preempt", "b64_json": "cHJl"}],
                },
            )
            return
        reply.json(400, {"error": {"message": "unknown prompt", "type": "invalid_request_error"}})

    def handle_video(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, body, selection, args, state
        self.video_called = True
        reply.json(
            200,
            {"id": "video_called", "status": "completed", "size": "1x1", "seconds": "1"},
        )

    def handle_music(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args, state
        reply.json(
            200,
            {
                "id": "music_job",
                "object": "audio.music",
                "status": "completed",
                "response_format": body.get("response_format") or "mp3",
            },
        )


def test_mcp_initialize_lists_tools_and_rejects_a_batch(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["image_home"] = tmp_path
    state["module"] = _QueueMedia(tmp_path)
    port = int(state["server"].server_address[1])
    try:
        status, body = _mcp(port, "server/discover", {})
        names_status, listed = _mcp(port, "tools/list", {})
        _refused_status, refused = _tool(
            port,
            "generate_image",
            {"prompt": "hold", "model": "example-model"},
            tasks=False,
        )
        batch_status, batch = _mcp_raw(
            port,
            b'[{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}]',
        )
    finally:
        _stop_background(state, thread)
    assert status == 200, body
    result = body["result"]
    assert "2026-07-28" in result["supportedVersions"]
    assert "io.modelcontextprotocol/tasks" in json.dumps(result["capabilities"])
    assert names_status == 200
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert names == {
        "list_media_models",
        "media_status",
        "media_cancel",
        "generate_image",
        "generate_video",
        "generate_music",
    }
    assert refused["error"]["code"] == -32021, refused
    assert not state.get("media_order")
    assert batch_status >= 400
    assert not isinstance(batch, list)


def test_mcp_queue_runs_the_first_image_and_cancel_skips_the_second(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    media = _QueueMedia(tmp_path)
    state["image_home"] = tmp_path
    state["module"] = media
    port = int(state["server"].server_address[1])
    try:
        _status, first = _tool(
            port, "generate_image", {"prompt": "hold", "model": "example-model"}
        )
        _status, second = _tool(
            port, "generate_image", {"prompt": "later", "model": "example-model"}
        )
        assert media.entered.wait(5)
        assert media.order == ["start"]
        first_id = first["result"]["taskId"]
        second_id = second["result"]["taskId"]
        assert first["result"]["status"] == "working"
        assert first["result"]["pollIntervalMs"] == 2000
        assert first_id != second_id
        _status, cancelled = _tool(port, "media_cancel", {"taskId": second_id})
        media.release.set()
        finished = _wait_task(port, first_id)
        skipped = _wait_task(port, second_id)
    finally:
        media.release.set()
        _stop_background(state, thread)
    assert media.order == ["start", "end"]
    assert finished["status"] == "completed"
    assert skipped["status"] == "cancelled"
    assert cancelled["result"]["structuredContent"]["status"] == "cancelled" or (
        cancelled["result"].get("status") == "cancelled"
    )


def test_mcp_cancel_during_a_batch_discards_the_file(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    media = _QueueMedia(tmp_path)
    state["image_home"] = tmp_path
    state["module"] = media
    port = int(state["server"].server_address[1])
    try:
        _status, created = _tool(
            port,
            "generate_image",
            {"prompt": "batch", "model": "example-model", "n": 2},
        )
        assert media.first_file.wait(5)
        task_id = created["result"]["taskId"]
        _tool(port, "media_cancel", {"taskId": task_id})
        media.hold_release.set()
        finished = _wait_task(port, task_id)
    finally:
        media.hold_release.set()
        _stop_background(state, thread)
    assert finished["status"] == "cancelled"
    assert not (tmp_path / "images" / "image_0.png").exists()
    assert not (tmp_path / "images" / "image_1.png").exists()


def test_mcp_image_results_are_links_and_resources_read_the_file(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["image_home"] = tmp_path
    state["module"] = _QueueMedia(tmp_path)
    port = int(state["server"].server_address[1])
    try:
        _status, pair = _tool(
            port,
            "generate_image",
            {"prompt": "pair", "model": "example-model", "n": 2},
        )
        pair_done = _wait_task(port, pair["result"]["taskId"])
        _status, big = _tool(
            port,
            "generate_image",
            {"prompt": "big", "model": "example-model", "include_image": True},
        )
        big_done = _wait_task(port, big["result"]["taskId"])
        _status, read = _mcp(
            port,
            "resources/read",
            {"uri": "favlo://images/image_a"},
            name="favlo://images/image_a",
        )
    finally:
        _stop_background(state, thread)
    blocks = pair_done["result"]["content"]
    assert [block["type"] for block in blocks].count("resource_link") == 2
    assert all(block["type"] != "image" for block in blocks)
    big_blocks = big_done["result"]["content"]
    assert any(block["type"] == "resource_link" for block in big_blocks)
    assert all(block["type"] != "image" for block in big_blocks)
    blob = read["result"]["contents"][0]["blob"]
    assert base64.b64decode(blob) == b"aaa"


def test_mcp_video_rejects_a_filesystem_path(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    media = _QueueMedia(tmp_path)
    state["module"] = media
    port = int(state["server"].server_address[1])
    try:
        _status, created = _tool(
            port,
            "generate_video",
            {"prompt": "gallop", "model": "example-model", "image": "/tmp/still.png"},
        )
        finished = _wait_task(port, created["result"]["taskId"])
    finally:
        _stop_background(state, thread)
    assert finished["result"]["isError"] is True
    assert "still" in json.dumps(finished)
    assert media.video_called is False


def test_mcp_chat_preempts_media_and_the_same_task_resumes(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    media = _QueueMedia(tmp_path)
    state["image_home"] = tmp_path
    state["module"] = media
    port = int(state["server"].server_address[1])
    chat: dict[str, bytes] = {}

    def post_chat() -> None:
        chat["body"] = _post_json(port, "/v1/chat/completions", {"messages": []})

    try:
        _status, created = _tool(
            port, "generate_image", {"prompt": "preempt", "model": "example-model"}
        )
        assert media.saw_a.wait(5)
        worker = threading.Thread(target=post_chat)
        worker.start()
        assert media.chat_entered.wait(5)
        _status, paused = _mcp(
            port, "tasks/get", _task_params(created["result"]["taskId"]), tasks=True
        )
        assert paused["result"]["stage"] == "paused"
        assert paused["result"]["status"] == "working"
        media.chat_release.set()
        worker.join(timeout=5)
        finished = _wait_task(port, created["result"]["taskId"])
    finally:
        media.chat_release.set()
        _stop_background(state, thread)
    assert media.log == ["a", "chat", "b"]
    assert finished["status"] == "completed"
    assert chat["body"].startswith(b"HTTP/1.1 200")


def test_media_http_routes_are_gone(monkeypatch, tmp_path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "example-model": {
                        "pipeline": "z-image-turbo",
                        "runtime": "metal41",
                        "alias": None,
                        "variants": {},
                        "context_length": 1536,
                    },
                    "text-model": {
                        "pipeline": "qwen36-35b-a3b",
                        "runtime": "metal41",
                        "alias": None,
                        "variants": {},
                        "context_length": 131072,
                    },
                },
            }
        )
    )
    monkeypatch.setattr("favlo.registry.registry_path", lambda root=None: registry)
    state, thread = _serve_background(_Runtime())
    port = int(state["server"].server_address[1])
    paths = (
        ("POST", "/v1/images/generations", {"prompt": "pair"}),
        ("POST", "/v1/images/edits", {"prompt": "pair"}),
        ("POST", "/v1/images/variations", {"prompt": "pair"}),
        ("GET", "/v1/images/image_a/content", None),
        ("POST", "/v1/videos", {"prompt": "gallop"}),
        ("POST", "/v1/videos/generations", {"prompt": "gallop"}),
        ("GET", "/v1/videos/video_test", None),
        ("GET", "/v1/videos/video_test/content", None),
        ("POST", "/v1/audio/music", {"prompt": "theme"}),
        ("POST", "/v1/audio/speech", {"input": "hello"}),
        ("GET", "/v1/audio/music/music_test", None),
        ("GET", "/v1/audio/music/music_test/content", None),
    )
    try:
        for method, route, payload in paths:
            status, _kind, _raw = _http(port, method, route, payload)
            assert status == 404, route
        models_status, _kind, models_raw = _http(port, "GET", "/v1/models")
        one_status, _kind, one_raw = _http(port, "GET", "/v1/models/text-model")
        media_status, _kind, _media_raw = _http(port, "GET", "/v1/models/example-model")
        release_status, _kind, release_raw = _http(
            port, "POST", "/v1/release", {"pid": os.getpid()}
        )
        resume_status, _kind, resume_raw = _http(port, "POST", "/v1/resume", {"pid": os.getpid()})
    finally:
        _stop_background(state, thread)
    assert models_status == 200
    rows = json.loads(models_raw)["data"]
    assert [row["id"] for row in rows] == ["text-model"]
    assert one_status == 200
    assert json.loads(one_raw)["pipeline"] == "qwen36-35b-a3b"
    assert media_status == 404
    assert release_status == 200
    assert b'"released": true' in release_raw
    assert resume_status == 200
    assert b'"resumed": true' in resume_raw


def test_language_harness_clients_read_chat_completions() -> None:
    from harness.benchmarks.serve import Request, complete
    from harness.coding.loop import _post
    from harness.coherence import _stream
    from openai import OpenAI

    state, thread = _serve_background(_Runtime())
    port = int(state["server"].server_address[1])
    messages = [{"role": "user", "content": "hello"}]
    try:
        text, reason, _tokens = complete(
            port,
            messages,
            8,
            Request(enable_thinking=False),
            model="text-model",
        )
        posted = _post(
            port,
            {
                "model": "text-model",
                "messages": messages,
                "stream": False,
                "enable_thinking": True,
            },
        )
        _reasoning, content, _led = _stream(
            port, messages, presence=0.0, window=20, model="text-model"
        )
        client = OpenAI(api_key="favlo", base_url=f"http://127.0.0.1:{port}/v1", timeout=30.0)
        completion = client.chat.completions.create(
            model="text-model",
            messages=messages,
            extra_body={"enable_thinking": False},
        )
    finally:
        _stop_background(state, thread)
    assert text
    assert reason
    assert posted["choices"][0]["message"]["content"]
    assert content or _reasoning
    assert completion.choices[0].message.content

