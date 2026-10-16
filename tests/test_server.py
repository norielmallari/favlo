"""One request at a time, a dropped client, and signal shutdown."""

import json
import os
import socket
import threading
import time
from types import SimpleNamespace

import qwen38_flash_next.commands as commands
from qwen38_flash_next.runtime import RequestClosed

from madmetal import listener


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
    usage = commands._usage(runtime, 1187, 480, "qwen38-flash-next")
    timings = usage["timings"]
    assert usage["prompt_tokens"] == 1187
    assert timings["prompt_n"] == 287
    assert timings["predicted_n"] == 480
    assert timings["prompt_per_second"] == 143.5
    assert timings["predicted_per_second"] == 120.0
    idle = commands._usage(SimpleNamespace(), 10, 4, "qwen38-flash-next")
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

    def opener(selection, console):
        del console
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
        alias=None,
        resident_percent=None,
        kv_cache="e4m3",
        path="flash",
    )
    second = SimpleNamespace(
        folder="dense",
        pipeline="qwen38-27b",
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
