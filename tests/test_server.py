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

import pytest

import qwen38_flash_next.commands as commands
from favlo import listener
from favlo.pipelines import drop_runtime, install_runtime, open_runtime
from qwen38_flash_next.runtime import RequestClosed


@pytest.fixture(autouse=True)
def _registry_media_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "favlo.media_session.default_model",
        lambda kind, root=None: "example-model",
    )


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


def test_a_queued_chat_whose_client_left_does_not_run() -> None:
    runtime = _Runtime()
    runtime.hold = True
    state, thread = _serve_background(runtime)
    port = int(state["server"].server_address[1])
    holder = threading.Thread(target=_exchange, args=(port, "hold"))
    try:
        holder.start()
        assert runtime.entered.wait(3)
        sock = _open_request(port, "gone", stream=False)
        _wait_waiting(state, "chat")
        sock.close()
        deadline = time.monotonic() + 3
        while state["schedule"].waiting("chat") and time.monotonic() < deadline:
            time.sleep(0.02)
        assert state["schedule"].waiting("chat") == 0
        runtime.release.set()
        holder.join(5)
    finally:
        runtime.release.set()
        _stop_background(state, thread)
    assert [name for _kind, name in runtime.order] == ["hold", "hold"]


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
            listener._release(state, _args(), shutdown)
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

    def bind(body, bound_state, bound_args, kind="chat"):
        del body, bound_args, kind
        bound_state["runtime"] = runtime
        bound_state["record"] = "packed"
        return SimpleNamespace(requested_id="packed", enable_thinking=None, sampling=None)

    monkeypatch.setattr("favlo.pipelines.bind_request", bind)
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
        runtime.counts = {}
        runtime.proxy = SimpleNamespace()
        opened.append(runtime)
        return runtime

    def start(selection, args, schedule=None):
        del args, schedule
        return opener(selection, None)

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
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
    install_runtime(state, args, first)
    install_runtime(state, args, first)
    assert len(opened) == 1
    assert state["pipeline"] == "qwen38-flash-next"
    install_runtime(state, args, second)
    assert opened[0].closed
    assert state["runtime"] is opened[1]
    assert state["record"] == "dense"
    assert state["pipeline"] == "qwen38-27b"


def _embed_reply() -> SimpleNamespace:
    reply = SimpleNamespace(started=False, payload=None, model_id=None, status=0)

    def json(status: int, payload: dict) -> None:
        reply.started = True
        reply.status = status
        reply.payload = payload

    reply.json = json
    return reply


def _embed_state(resident: _Runtime) -> dict:
    from favlo.schedule import ensure

    state: dict = {
        "runtime": resident,
        "record": "qwen",
        "pipeline": "qwen36-35b-a3b",
        "module": resident,
        "lock": threading.Lock(),
    }
    ensure(state)
    return state


def test_an_embedding_request_leaves_the_resident_runtime(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    resident.counts = {}
    state = _embed_state(resident)
    opened: list[_Runtime] = []

    def handle_embed(reply, runtime, body, selection, args, request_state) -> None:
        del body, selection, args, request_state
        assert runtime is opened[0]
        assert resident.closed is False
        reply.json(200, {"object": "list", "data": []})

    def start(selection, args, schedule=None):
        del args, schedule
        child = _Runtime()
        child.counts = {}
        child.selection = selection
        child.proxy = SimpleNamespace(handle_embed=handle_embed)
        opened.append(child)
        return child

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, resident=None: _selection("bge-m3-int8", "bge-m3"),
    )
    monkeypatch.setattr(
        "favlo.pipelines.pipeline_module",
        lambda selection: SimpleNamespace(handle_embed=handle_embed),
    )
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(
        args,
        state,
        "embed",
        "application/json",
        b'{"model": "bge-m3-int8", "input": "bridge"}',
        reply,
    )
    assert reply.status == 200
    assert resident.closed is False
    assert state["runtime"] is resident
    assert state["record"] == "qwen"
    assert state["pipeline"] == "qwen36-35b-a3b"
    assert opened[0].closed is True


def test_an_embedding_child_exit_keeps_the_resident_runtime(monkeypatch) -> None:
    from favlo.http_app import _run_once
    from favlo.pipeline_worker import ChildExited

    resident = _Runtime()
    resident.counts = {}
    state = _embed_state(resident)
    opened: list[_Runtime] = []

    def handle_embed(reply, runtime, body, selection, args, request_state) -> None:
        del reply, runtime, body, selection, args, request_state
        raise ChildExited("embeddings process exited")

    def start(selection, args, schedule=None):
        del args, schedule
        child = _Runtime()
        child.counts = {}
        child.selection = selection
        child.proxy = SimpleNamespace(handle_embed=handle_embed)
        opened.append(child)
        return child

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, resident=None: _selection("bge-m3-int8", "bge-m3"),
    )
    monkeypatch.setattr(
        "favlo.pipelines.pipeline_module",
        lambda selection: SimpleNamespace(handle_embed=handle_embed),
    )
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(
        args,
        state,
        "embed",
        "application/json",
        b'{"model": "bge-m3-int8", "input": "bridge"}',
        reply,
    )
    assert reply.status == 500
    assert resident.closed is False
    assert state["runtime"] is resident
    assert opened[0].closed is True


def test_the_resident_embeddings_model_closes_when_the_request_returns(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    resident.counts = {}

    def handle_embed(reply, runtime, body, selection, args, request_state) -> None:
        del runtime, body, selection, args, request_state
        reply.json(200, {"object": "list", "data": []})

    resident.handle_embed = handle_embed
    state = _embed_state(resident)
    state["record"] = "bge-m3-int8"
    state["pipeline"] = "bge-m3"
    state["module"] = resident

    def start(selection, args, schedule=None):
        del selection, args, schedule
        raise AssertionError("a second embeddings process opened")

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, resident=None: _selection("bge-m3-int8", "bge-m3"),
    )
    monkeypatch.setattr(
        "favlo.pipelines.pipeline_module",
        lambda selection: SimpleNamespace(handle_embed=handle_embed),
    )
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(
        args,
        state,
        "embed",
        "application/json",
        b'{"model": "bge-m3-int8", "input": "bridge"}',
        reply,
    )
    assert reply.status == 200
    assert resident.closed is True
    assert state["runtime"] is None
    assert state["record"] is None


def test_an_embedding_request_uses_the_registry_default_when_model_is_omitted(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    state = _embed_state(resident)
    seen: list[str] = []

    def handle_embed(reply, runtime, body, selection, args, request_state) -> None:
        del runtime, body, selection, args, request_state
        reply.json(200, {"object": "list", "data": []})

    def start(selection, args, schedule=None):
        del args, schedule
        child = _Runtime()
        child.selection = selection
        child.proxy = SimpleNamespace(handle_embed=handle_embed)
        return child

    def resolve(model, resident=None):
        del resident
        seen.append(model)
        return _selection(model, "bge-m3")

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr("favlo.registry.resolve", resolve)
    monkeypatch.setattr("favlo.registry.embeddings_default", lambda root=None: "bge-m3-int8")
    monkeypatch.setattr(
        "favlo.pipelines.pipeline_module",
        lambda selection: SimpleNamespace(handle_embed=handle_embed),
    )
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(args, state, "embed", "application/json", b'{"input": "bridge"}', reply)
    assert reply.status == 200
    assert seen == ["bge-m3-int8"]
    assert resident.closed is False


def test_an_embedding_request_without_a_default_is_not_found(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    state = _embed_state(resident)

    def start(selection, args, schedule=None):
        del selection, args, schedule
        raise AssertionError("an embeddings process opened without a model")

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr("favlo.registry.embeddings_default", lambda root=None: "")
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(args, state, "embed", "application/json", b'{"input": "bridge"}', reply)
    assert reply.status == 404
    assert resident.closed is False


def test_an_embedding_request_refuses_a_model_that_does_not_embed(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    state = _embed_state(resident)

    def start(selection, args, schedule=None):
        del selection, args, schedule
        raise AssertionError("a model that does not embed was opened")

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, resident=None: _selection("qwen", "qwen36-35b-a3b"),
    )
    monkeypatch.setattr("favlo.pipelines.pipeline_module", lambda selection: SimpleNamespace())
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(
        args,
        state,
        "embed",
        "application/json",
        b'{"model": "qwen", "input": "bridge"}',
        reply,
    )
    assert reply.status == 404
    assert resident.closed is False
    assert state["runtime"] is resident


def test_an_image_request_does_not_enter_the_chat_runtime(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    resident.counts = {}
    state = _embed_state(resident)
    state["module"] = SimpleNamespace(handle_chat=lambda *_args: None)

    def start(selection, args, schedule=None):
        del selection, args, schedule
        raise AssertionError("the chat runtime was opened for an image")

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, resident=None: _selection("qwen", "qwen36-35b-a3b"),
    )
    monkeypatch.setattr("favlo.pipelines.pipeline_module", lambda selection: SimpleNamespace())
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    _run_once(
        args,
        state,
        "image",
        "application/json",
        b'{"model": "qwen", "prompt": "a horse"}',
        reply,
    )
    assert reply.status == 404
    assert resident.closed is False
    assert state["runtime"] is resident
    assert state["pipeline"] == "qwen36-35b-a3b"


def test_an_image_request_opens_the_image_runtime(monkeypatch) -> None:
    from favlo.http_app import _run_once

    resident = _Runtime()
    resident.counts = {}
    state = _embed_state(resident)
    opened: list[_Runtime] = []

    def handle_image(reply, runtime, body, selection, args, request_state) -> None:
        del body, args, request_state
        assert runtime is opened[0]
        assert selection.pipeline == "z-image-turbo"
        reply.json(200, {"data": [{"id": "image_ok"}]})

    def start(selection, args, schedule=None):
        del args, schedule
        child = _Runtime()
        child.counts = {}
        child.selection = selection
        child.proxy = SimpleNamespace(handle_image=handle_image)
        opened.append(child)
        return child

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    monkeypatch.setattr(
        "favlo.registry.resolve",
        lambda model, resident=None: _selection("z-image-turbo", "z-image-turbo"),
    )
    monkeypatch.setattr(
        "favlo.pipelines.pipeline_module",
        lambda selection: SimpleNamespace(handle_image=handle_image),
    )
    args = _args()
    args.lazy = True
    reply = _embed_reply()
    try:
        _run_once(
            args,
            state,
            "image",
            "application/json",
            b'{"model": "z-image-turbo", "prompt": "a horse"}',
            reply,
        )
        assert reply.status == 200
        assert resident.closed is True
        assert state["pipeline"] == "z-image-turbo"
        assert state["runtime"] is opened[0]
    finally:
        drop_runtime(state, args)


def _selection(folder: str, pipeline: str) -> SimpleNamespace:
    return SimpleNamespace(
        folder=folder,
        pipeline=pipeline,
        runtime="metal41",
        custom=False,
        alias=None,
        resident_percent=None,
        kv_cache="e4m3",
        path=folder,
        requested_id=folder,
    )


def test_open_runtime_passes_owner_when_accepted(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def opener(path, *, kv_cache, console, resident_percent, owner):
        del kv_cache, console, resident_percent
        seen["path"] = path
        seen["owner"] = owner
        return SimpleNamespace(_owner=owner)

    module = SimpleNamespace(open_runtime=opener)
    monkeypatch.setattr("favlo.pipelines.pipeline_module", lambda _selection: module)
    runtime = open_runtime(_selection("record", "z-image-turbo"), None, None)
    assert seen["path"] == "record"
    assert seen["owner"].startswith("z-image-turbo:")
    assert runtime._owner == seen["owner"]


def test_open_runtime_omits_owner_when_absent(monkeypatch) -> None:
    def opener(path, *, kv_cache, console, resident_percent):
        del kv_cache, console, resident_percent
        return SimpleNamespace(path=path)

    module = SimpleNamespace(open_runtime=opener)
    monkeypatch.setattr("favlo.pipelines.pipeline_module", lambda _selection: module)
    runtime = open_runtime(_selection("record", "wan22-i2v"), None, None)
    assert runtime.path == "record"


def test_a_failed_open_sweeps_spans_from_that_attempt(monkeypatch) -> None:
    from favlo.memory import memory

    held = memory.allocate(4096, owner="listener-held", zero=True)

    def opener(selection, console, max_memory=None):
        del selection, console, max_memory
        memory.allocate(4096, owner="listener-failed", zero=True)
        message = "open failed"
        raise RuntimeError(message)

    def start(selection, bound_args, schedule=None):
        del schedule
        return opener(selection, None, getattr(bound_args, "max_memory", None))

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    state: dict = {"runtime": None, "record": None, "pipeline": None, "module": None}
    try:
        try:
            install_runtime(state, _args(), _selection("rec", "z-image-turbo"))
        except RuntimeError as exc:
            assert str(exc) == "open failed"
        else:
            message = "open should fail"
            raise AssertionError(message)
        assert state["runtime"] is None
        assert "listener-failed" not in memory.owners()
        assert memory.nbytes(held) >= 16
    finally:
        memory.sweep("listener-held")
        memory.sweep("listener-failed")


def test_close_sweeps_the_owner_recorded_at_open(monkeypatch) -> None:
    from favlo.memory import memory

    owner = "listener-resident"

    def opener(selection, console, max_memory=None):
        del selection, console, max_memory
        memory.allocate(4096, owner=owner, zero=True)
        runtime = _Runtime()
        runtime._owner = owner
        runtime.counts = {owner: 4096}
        runtime.proxy = SimpleNamespace()
        return runtime

    def start(selection, bound_args, schedule=None):
        del schedule
        return opener(selection, None, getattr(bound_args, "max_memory", None))

    monkeypatch.setattr("favlo.pipelines.start_pipeline", start)
    state: dict = {"runtime": None, "record": None, "pipeline": None, "module": None}
    try:
        install_runtime(state, _args(), _selection("rec", "z-image-turbo"))
        assert owner in memory.owners()
        drop_runtime(state, _args())
        assert owner not in memory.owners()
        assert state["runtime"] is None
    finally:
        memory.sweep(owner)


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
            if state["cancel"].is_set():
                reply.json(200, {"status": "in_progress", "data": []})
                return
            _store_png(self.home, "image_0", b"batch")
            wrote = [{"id": "image_0", "b64_json": "YmF0Y2g="}]
            self.first_file.set()
            self.hold_release.wait(5)
            if state["cancel"].is_set():
                reply.json(200, {"status": "in_progress", "data": wrote})
                return
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

    def handle_voice(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args
        import wave
        from io import BytesIO

        import numpy as np

        from favlo.voice import store_wav

        tone = np.sin(np.linspace(0.0, 40.0, 4410, dtype=np.float32)) * np.float32(0.2)
        pcm = np.round(np.clip(tone, -1.0, 1.0) * 32767.0).astype("<i2")
        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(pcm.tobytes())
        voice_id = "voice_smoke"
        store_wav(voice_id, buffer.getvalue(), home=self.home, output_format="wav")
        state.setdefault("voice", {})[voice_id] = {"response_format": body.get("response_format") or "wav"}
        state["voice_home"] = self.home
        reply.json(
            200,
            {
                "id": voice_id,
                "object": "audio.voice",
                "status": "completed",
                "response_format": "wav",
            },
        )


def test_mcp_answers_after_the_pipeline_is_closed() -> None:
    from favlo.pipelines import pipelines

    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["record"] = "example-model"
    pipelines.add(runtime, ())
    port = int(state["server"].server_address[1])
    try:
        closed = _post_json(port, "/v1/pipelines/close", {})
        status, listed = _mcp(port, "tools/list", {})
    finally:
        _stop_background(state, thread)
    assert closed.startswith(b"HTTP/1.1 200")
    assert b'"unloaded": true' in closed
    assert runtime.closed
    assert state["runtime"] is None
    assert status == 200
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert "list_video_models_i2v" in names


def test_mcp_initialize_lists_tools_and_rejects_a_batch(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["image_home"] = tmp_path
    state["module"] = _QueueMedia(tmp_path)
    port = int(state["server"].server_address[1])
    try:
        status, body = _mcp(port, "server/discover", {})
        names_status, listed = _mcp(port, "tools/list", {})
        inline_status, inline = _tool(
            port,
            "generate_image",
            {"prompt": "pair"},
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
        "list_image_models",
        "list_video_models_i2v",
        "list_video_models_t2v",
        "list_music_models",
        "list_voice_models",
        "media_status",
        "media_cancel",
        "read_media",
        "remember",
        "recall",
        "forget",
        "set_image_model",
        "set_video_i2v_model",
        "set_video_t2v_model",
        "set_music_model",
        "set_voice_model",
        "generate_image",
        "generate_video_i2v",
        "generate_video_t2v",
        "generate_music",
        "generate_voice",
    }
    assert inline_status == 200, inline
    assert "error" not in inline
    assert inline["result"].get("resultType") == "complete"
    assert "taskId" not in inline["result"]
    assert not state.get("media_order")
    assert batch_status >= 400
    assert not isinstance(batch, list)


def test_legacy_initialize_is_rejected(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["image_home"] = tmp_path
    port = int(state["server"].server_address[1])
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "favlo-test", "version": "0"},
            },
        }
    ).encode()
    try:
        status, body = _mcp_raw(
            port,
            raw,
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "mcp-protocol-version": "2025-06-18",
                "mcp-method": "initialize",
            },
        )
        discover_status, discovered = _mcp(port, "server/discover", {})
    finally:
        _stop_background(state, thread)
    assert status == 200, body
    assert body["error"]["code"] == -32022
    assert body["id"] == 7
    assert discover_status == 200
    assert "2026-07-28" in discovered["result"]["supportedVersions"]


def test_a_queued_or_paused_task_stays_ongoing() -> None:
    from favlo.mcp_server import _working_fields

    state = {
        "media_tasks": {
            "run": {
                "started": True,
                "paused": False,
                "job": {"status": "in_progress", "progress": 40, "stage": "high", "eta": 12},
            },
            "wait": {"started": False, "paused": False, "job": None},
            "hold": {
                "started": True,
                "paused": True,
                "job": {"status": "in_progress", "progress": 40, "stage": "high", "eta": 12},
            },
            "fresh": {"started": True, "paused": False, "job": None},
        },
        "media_order": ["run", "wait", "hold", "fresh"],
        "videos": {"video_x": {"status": "in_progress", "progress": 3, "stage": "encode"}},
    }
    queued = _working_fields(state, "wait")
    paused = _working_fields(state, "hold")
    running = _working_fields(state, "run")
    fresh = _working_fields(state, "fresh")
    assert queued == {"position": 0, "statusMessage": "task is ongoing"}
    assert paused == {
        "statusMessage": "task is ongoing",
        "stage": "paused",
        "progress": 40,
        "eta": 12,
    }
    assert running == {"progress": 40, "eta": 12, "stage": "high"}
    assert fresh == {"progress": 3, "stage": "encode"}


def test_mcp_queue_runs_the_first_image_and_cancel_skips_the_second(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    media = _QueueMedia(tmp_path)
    state["image_home"] = tmp_path
    state["module"] = media
    port = int(state["server"].server_address[1])
    try:
        _status, first = _tool(
            port, "generate_image", {"prompt": "hold"}
        )
        _status, second = _tool(
            port, "generate_image", {"prompt": "later"}
        )
        assert media.entered.wait(5)
        assert media.order == ["start"]
        first_id = first["result"]["taskId"]
        second_id = second["result"]["taskId"]
        assert first["result"]["status"] == "working"
        assert first["result"]["pollIntervalMs"] == 2000
        assert first_id != second_id
        _status, queued = _mcp(port, "tasks/get", _task_params(second_id), tasks=True)
        assert queued["result"]["status"] == "working"
        assert queued["result"]["statusMessage"] == "task is ongoing"
        assert queued["result"]["position"] == 0
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
            {"prompt": "batch"},
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
            {"prompt": "pair"},
        )
        pair_done = _wait_task(port, pair["result"]["taskId"])
        _status, big = _tool(
            port,
            "generate_image",
            {"prompt": "big", "include_image": True},
        )
        big_done = _wait_task(port, big["result"]["taskId"])
        _status, read = _mcp(
            port,
            "resources/read",
            {"uri": "favlo://images/image_a.png"},
            name="favlo://images/image_a.png",
        )
    finally:
        _stop_background(state, thread)
    blocks = pair_done["result"]["content"]
    links = [block for block in blocks if block["type"] == "resource_link"]
    assert [block["uri"] for block in links] == [
        "favlo://images/image_a.png",
        "favlo://images/image_b.png",
    ]
    assert [block["name"] for block in links] == ["image_a.png", "image_b.png"]
    assert all(block["type"] != "image" for block in blocks)
    big_blocks = big_done["result"]["content"]
    assert any(block["type"] == "resource_link" for block in big_blocks)
    assert all(block["type"] != "image" for block in big_blocks)
    content = read["result"]["contents"][0]
    assert content["uri"] == "favlo://images/image_a.png"
    assert content["mimeType"] == "image/png"
    assert base64.b64decode(content["blob"]) == b"aaa"


def test_generate_video_schema_names_the_still_and_the_frame_default() -> None:
    import asyncio

    from favlo.mcp_server import build_mcp

    mcp = build_mcp(lambda *_args: {}, {}, lambda _text: [0.0] * 1024)
    tools = asyncio.run(mcp.list_tools(run_middleware=False))
    i2v = next(tool for tool in tools if tool.name == "generate_video_i2v")
    t2v = next(tool for tool in tools if tool.name == "generate_video_t2v")
    description = i2v.description or ""
    properties = i2v.parameters["properties"]
    assert "image_" in description or "image_" in properties["image"]["description"]
    assert "favlo://videos/" in description
    assert "duration" in properties
    assert "frames" not in properties
    assert "seconds" in properties["duration"]["description"]
    assert "4n+1" not in properties["duration"]["description"]
    assert "16 fps" not in description
    assert "image" in i2v.parameters["required"]
    assert "size" in i2v.parameters["required"]
    assert "list_video_models_i2v" in properties["size"]["description"]
    assert "image" not in t2v.parameters["properties"]
    assert "size" in t2v.parameters["required"]
    assert "list_video_models_t2v" in t2v.parameters["properties"]["size"]["description"]
    assert "xhigh" in properties["quality"]["description"]
    assert "sample negative" in properties["negative_prompt"]["description"]
    assert "[low, high]" in properties["guidance_scale"]["description"]
    assert "40" in properties["steps"]["description"]
    assert "model" not in properties
    assert "session" in properties
    assert "session" not in i2v.parameters["required"]
    image = next(tool for tool in tools if tool.name == "generate_image")
    assert "model" not in image.parameters["properties"]
    assert "n" not in image.parameters["properties"]
    assert "session" in image.parameters["properties"]
    assert "seed" not in image.parameters["required"]
    seed = image.parameters["properties"]["seed"]["description"]
    assert "random" in seed


def test_mcp_video_rejects_a_filesystem_path(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    media = _QueueMedia(tmp_path)
    state["module"] = media
    port = int(state["server"].server_address[1])
    try:
        _status, created = _tool(
            port,
            "generate_video_i2v",
            {
                "prompt": "gallop",
                "size": "576x320",
                "image": "/tmp/still.png",
            },
        )
        finished = _wait_task(port, created["result"]["taskId"])
    finally:
        _stop_background(state, thread)
    assert finished["result"]["isError"] is True
    assert "image_" in json.dumps(finished)
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
            port, "generate_image", {"prompt": "preempt"}
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
        assert paused["result"]["statusMessage"] == "task is ongoing"
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
                    "missing-model": {
                        "pipeline": "missing-pipeline",
                        "runtime": "metal41",
                        "alias": None,
                        "variants": {"default": {}},
                        "context_length": 1024,
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
        missing_status, _kind, _missing_raw = _http(port, "GET", "/v1/models/missing-model")
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
    assert missing_status == 404
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


def test_mcp_voice_writes_a_wav_and_the_link_reads_it(tmp_path) -> None:
    runtime = _Runtime()
    state, thread = _serve_background(runtime)
    state["voice_home"] = tmp_path
    state["module"] = _QueueMedia(tmp_path)
    port = int(state["server"].server_address[1])
    try:
        status, called = _tool(
            port,
            "generate_voice",
            {"input": "The small brown fox spoke one short line.", "response_format": "wav"},
            tasks=False,
        )
        _read_status, read = _mcp(
            port,
            "resources/read",
            {"uri": "favlo://voice/voice_smoke.wav"},
            name="favlo://voice/voice_smoke.wav",
        )
    finally:
        _stop_background(state, thread)
    assert status == 200, called
    content = called["result"]["content"]
    assert content[1]["uri"] == "favlo://voice/voice_smoke.wav"
    assert content[1]["name"] == "voice_smoke.wav"
    assert content[1]["mimeType"] == "audio/wav"
    blob = read["result"]["contents"][0]
    assert blob["uri"] == "favlo://voice/voice_smoke.wav"
    assert blob["mimeType"] == "audio/wav"
    assert blob["blob"]
    assert (tmp_path / "voice" / "voice_smoke.wav").is_file()


class _EmbedOrder:
    """Chat holds the lock. An embedding request runs before the next chat."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.hold_entered = threading.Event()
        self.hold_release = threading.Event()
        self.embed_saw_preempt = False

    def handle_chat(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args, state
        name = body["messages"][0]["content"]
        if name == "hold":
            self.hold_entered.set()
            self.hold_release.wait(5)
        self.order.append(f"chat:{name}")
        reply.json(200, {"id": name, "choices": []})

    def handle_embed(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args
        self.embed_saw_preempt = bool(state["preempt"].is_set())
        name = body["input"]
        if name == "hold":
            self.hold_entered.set()
            self.hold_release.wait(5)
        self.order.append(f"embed:{name}")
        reply.json(200, {"object": "list", "data": [], "model": "", "usage": {}})


def _embed_listener() -> tuple[_EmbedOrder, dict, threading.Thread]:
    module = _EmbedOrder()
    state: dict = {"runtime": module, "server": None, "module": module}
    thread = threading.Thread(
        target=listener._serve_loop,
        args=(_args(), state),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while state["server"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert state["server"] is not None
    return module, state, thread


def _wait_waiting(state: dict, kind: str) -> None:
    schedule = state["schedule"]
    deadline = time.monotonic() + 5
    while schedule.waiting(kind) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert schedule.waiting(kind) >= 1


def test_an_embedding_waits_for_the_open_chat_and_runs_before_the_next() -> None:
    module, state, thread = _embed_listener()
    port = int(state["server"].server_address[1])
    hold = threading.Thread(
        target=_post_json,
        args=(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hold"}]}),
    )
    embed = threading.Thread(
        target=_post_json,
        args=(port, "/v1/embeddings", {"input": "bridge", "model": "bge-m3-int8"}),
    )
    nxt = threading.Thread(
        target=_post_json,
        args=(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "next"}]}),
    )
    try:
        hold.start()
        assert module.hold_entered.wait(5)
        nxt.start()
        _wait_waiting(state, "chat")
        embed.start()
        _wait_waiting(state, "embed")
        assert module.order == []
        module.hold_release.set()
        hold.join(5)
        embed.join(5)
        nxt.join(5)
    finally:
        module.hold_release.set()
        _stop_background(state, thread)
    assert module.order == ["chat:hold", "embed:bridge", "chat:next"]
    assert module.embed_saw_preempt is False


def test_a_chat_waits_while_an_embedding_is_running() -> None:
    module, state, thread = _embed_listener()
    port = int(state["server"].server_address[1])
    embed = threading.Thread(
        target=_post_json,
        args=(port, "/v1/embeddings", {"input": "hold", "model": "bge-m3-int8"}),
    )
    chat = threading.Thread(
        target=_post_json,
        args=(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "next"}]}),
    )
    try:
        embed.start()
        assert module.hold_entered.wait(5)
        chat.start()
        time.sleep(0.05)
        assert module.order == []
        assert state["schedule"].running() == "embed"
        module.hold_release.set()
        embed.join(5)
        chat.join(5)
    finally:
        module.hold_release.set()
        _stop_background(state, thread)
    assert module.order == ["embed:hold", "chat:next"]
    assert module.embed_saw_preempt is False


class _ForegroundMedia:
    """One image call yields to the foreground queue, then finishes."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.ready = threading.Event()
        self.hold = threading.Event()

    def handle_image(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, body, selection, args
        if not self.order:
            self.ready.set()
            assert self.hold.wait(5)
            self.ready.clear()
            self.hold.clear()
        if state["preempt"].is_set():
            self.order.append("yield")
            reply.json(200, {"status": "in_progress"})
            return
        self.order.append("resume")
        reply.json(200, {"data": [{"id": "image_queue"}]})

    def handle_chat(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args, state
        self.order.append("chat:" + body["messages"][0]["content"])
        reply.json(200, {"id": "chat", "choices": []})

    def handle_embed(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args, state
        self.order.append("embed:" + str(body.get("input")))
        reply.json(200, {"object": "list", "data": [], "model": "", "usage": {}})


def _foreground_listener(module):
    state: dict = {"runtime": module, "server": None, "module": module}
    thread = threading.Thread(
        target=listener._serve_loop,
        args=(_args(), state),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while state["server"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert state["server"] is not None
    return state, thread


def test_the_foreground_queue_drains_before_media_resumes() -> None:
    module = _ForegroundMedia()
    state, thread = _foreground_listener(module)
    port = int(state["server"].server_address[1])
    image: dict[str, Any] = {}

    def post_image() -> None:
        image["result"] = _tool(
            port,
            "generate_image",
            {"prompt": "queue"},
            tasks=False,
        )

    embed = threading.Thread(
        target=_post_json,
        args=(port, "/v1/embeddings", {"input": "ahead", "model": "bge-m3-int8"}),
    )
    chat = threading.Thread(
        target=_post_json,
        args=(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "later"}]}),
    )
    worker = threading.Thread(target=post_image)
    try:
        worker.start()
        assert module.ready.wait(5)
        embed.start()
        _wait_waiting(state, "embed")
        chat.start()
        _wait_waiting(state, "chat")
        assert state["preempt"].is_set()
        assert module.order == []
        module.hold.set()
        worker.join(5)
        embed.join(5)
        chat.join(5)
    finally:
        module.hold.set()
        _stop_background(state, thread)
    assert module.order == ["yield", "embed:ahead", "chat:later", "resume"]
    assert "image_queue" in json.dumps(image["result"])


class _TwoSlices:
    """Each media slice waits, so a later chat still preempts the next slice."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.phase = 0
        self.ready = threading.Event()
        self.hold = threading.Event()

    def handle_image(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, body, selection, args
        self.phase += 1
        if self.phase < 3:
            self.ready.set()
            assert self.hold.wait(5)
            self.ready.clear()
            self.hold.clear()
            assert state["preempt"].is_set()
            self.order.append(f"yield{self.phase}")
            reply.json(200, {"status": "in_progress"})
            return
        self.order.append("resume")
        reply.json(200, {"data": [{"id": "image_queue"}]})

    def handle_chat(self, reply, runtime, body, selection, args, state) -> None:
        del runtime, selection, args, state
        self.order.append("chat:" + body["messages"][0]["content"])
        reply.json(200, {"id": "chat", "choices": []})


def test_a_later_chat_preempts_the_next_media_slice() -> None:
    module = _TwoSlices()
    state, thread = _foreground_listener(module)
    port = int(state["server"].server_address[1])

    def post_image() -> None:
        _tool(
            port,
            "generate_image",
            {"prompt": "queue"},
            tasks=False,
        )

    def post_chat(name: str) -> None:
        _post_json(
            port,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": name}]},
        )

    worker = threading.Thread(target=post_image)
    first = threading.Thread(target=post_chat, args=("one",))
    second = threading.Thread(target=post_chat, args=("two",))
    try:
        worker.start()
        assert module.ready.wait(5)
        module.ready.clear()
        first.start()
        _wait_waiting(state, "chat")
        module.hold.set()
        assert module.ready.wait(5)
        module.ready.clear()
        second.start()
        _wait_waiting(state, "chat")
        module.hold.set()
        worker.join(5)
        first.join(5)
        second.join(5)
    finally:
        module.hold.set()
        _stop_background(state, thread)
    assert module.order == ["yield1", "chat:one", "yield2", "chat:two", "resume"]

