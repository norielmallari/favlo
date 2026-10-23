"""The one listener. A request names a pipeline, and that pipeline loads.

This module does not import a pipeline. ``open_runtime`` and ``handle_chat``
come from the entry file the registry row names. After the request closes,
a pipeline that defines ``idle_heartbeat`` runs that function here.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
import select
import signal
import socket
import sys
import threading
import traceback
import urllib.parse
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from favlo.console import VERSION, Console
from favlo.daemon import DAEMON_ENV, alive
from favlo.images import load_image, sweep_images
from favlo.logs import attach_process_log
from favlo.registry import model_objects, one_model, resolve
from favlo.videos import load_video, reported_quality, sweep_videos

_STREAM_WRITE_TIMEOUT = 2.0
_REQUEST_READ_TIMEOUT = 10.0
_RELEASE_WAIT = 25.0
_log_sink: Any = None


def bind_log(sink: Any) -> None:
    """Route listener lines through the console. ``None`` restores stderr."""
    global _log_sink
    _log_sink = sink


def _schedule_pipeline_heartbeat(module: Any, runtime: Any, selection: Any) -> None:
    """Run one pipeline lull in this process. The terminal does not call it."""
    pump = getattr(module, "idle_heartbeat", None) if module is not None else None
    if not callable(pump) or runtime is None:
        return
    threading.Thread(
        target=pump,
        args=(runtime, selection),
        name="favlo-pipeline-heartbeat",
        daemon=True,
    ).start()


def _log(message: str) -> None:
    sink = _log_sink
    if sink is not None:
        sink(message)
        return
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _report_failure(console: Any) -> None:
    """Write a traceback under the live rows, not through them."""
    text = traceback.format_exc()
    report = getattr(console, "report", None)
    if report is None:
        sys.stderr.write(text)
        sys.stderr.flush()
        return
    report(text)


class _SingleRequestServer(ThreadingHTTPServer):
    """Each connection has its own thread. One chat completion runs at a time.

    A connection that holds its socket open, such as an editor probing the
    port, does not hold the chat slot.
    """

    request_queue_size = 128
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A peer that closes mid-response is not a server failure."""
        if isinstance(sys.exception(), (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class _Shutdown:
    """SIGINT and SIGTERM stop the forward and leave the process able to release."""

    def __init__(self) -> None:
        self.runtime: Any = None
        self.busy: threading.Lock | None = None
        self.closing = False
        self.signaled = False

    def arm(self) -> None:
        self._previous = (
            signal.getsignal(signal.SIGINT),
            signal.getsignal(signal.SIGTERM),
        )
        signal.signal(signal.SIGINT, self.handle)
        signal.signal(signal.SIGTERM, self.handle)

    def disarm(self) -> None:
        previous = getattr(self, "_previous", None)
        if previous is None:
            return
        signal.signal(signal.SIGINT, previous[0])
        signal.signal(signal.SIGTERM, previous[1])
        self._previous = None

    def handle(self, _signum: int, _frame: Any) -> None:
        if self.closing:
            return
        self.signaled = True
        runtime = self.runtime
        if runtime is not None:
            runtime.halt()
        raise KeyboardInterrupt


def _release(runtime: Any, shutdown: _Shutdown, server: Any = None) -> None:
    """Close the listening socket and the runtime. A signal during this returns."""
    shutdown.closing = True
    try:
        if server is not None:
            try:
                server.server_close()
            except OSError:
                pass
        if runtime is not None:
            busy = shutdown.busy
            held = busy.acquire(timeout=_RELEASE_WAIT) if busy is not None else False
            try:
                runtime.close()
            finally:
                if held and busy is not None:
                    busy.release()
        if shutdown.signaled and (runtime is not None or server is not None):
            _log("released")
    finally:
        shutdown.disarm()


def _peer_closed(sock: socket.socket) -> bool:
    """True when the client has closed its side or the socket is already dead."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        peeked = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return False
    except OSError:
        return True
    return peeked == b""


def _pipeline_module(name: str) -> Any:
    from favlo.cli import load_pipeline

    return load_pipeline(name)


def _open_runtime(selection: Any, console: Any, max_memory: str | None) -> Any:
    """The pipeline named on the registry row. The previous runtime is already closed."""
    module = _pipeline_module(selection.pipeline)
    opener = getattr(module, "open_runtime", None)
    if not callable(opener):
        message = f"{selection.pipeline} does not open a runtime"
        raise RuntimeError(message)
    kwargs: dict[str, Any] = {
        "kv_cache": selection.kv_cache,
        "console": console,
        "resident_percent": selection.resident_percent,
    }
    if "max_memory" in inspect.signature(opener).parameters:
        kwargs["max_memory"] = max_memory
    return opener(selection.path, **kwargs)


def _install_runtime(state: dict[str, Any], args: Any, selection: Any) -> None:
    if (
        state.get("runtime") is not None
        and state.get("record") == selection.folder
        and state.get("pipeline") == selection.pipeline
    ):
        return
    old = state.get("runtime")
    state["runtime"] = None
    state["record"] = None
    state["pipeline"] = None
    state["module"] = None
    if old is not None:
        old.close()
    ui = getattr(args, "console_ui", None)
    if ui is not None:
        ui.pipeline = selection.pipeline
        ui.load_detail = selection.alias or selection.folder
    runtime = _open_runtime(selection, ui, getattr(args, "max_memory", None))
    state["runtime"] = runtime
    state["record"] = selection.folder
    state["pipeline"] = selection.pipeline
    state["module"] = _pipeline_module(selection.pipeline)
    shutdown = getattr(args, "shutdown", None)
    if shutdown is not None:
        shutdown.runtime = runtime
    if ui is not None and ui.keeps_lines:
        label = selection.alias or selection.folder
        _log(f"loaded {label} {selection.folder}")


def _smoke_held(state: dict[str, Any]) -> bool:
    """True when a live smoke pid still owns the GPU. A dead pid clears the hold."""
    holder = state.get("smoke_hold")
    if isinstance(holder, bool) or not isinstance(holder, int) or holder <= 0:
        return False
    if not alive(holder):
        state["smoke_hold"] = None
        return False
    return True


def _drop_runtime(state: dict[str, Any], args: Any) -> str | None:
    """Close the resident pipeline and leave the listener up."""
    old = state.get("runtime")
    folder = state.get("record")
    state["runtime"] = None
    state["record"] = None
    state["pipeline"] = None
    state["module"] = None
    shutdown = getattr(args, "shutdown", None)
    if shutdown is not None:
        shutdown.runtime = None
    if old is not None:
        old.close()
    if isinstance(folder, str) and folder:
        _log(f"released {folder}")
        return folder
    return None


def _bind_request(body: dict[str, Any], state: dict[str, Any], args: Any) -> Any:
    """Load the requested folder when serve was started with an empty process."""
    if not getattr(args, "lazy", False):
        return None
    model = body.get("model")
    if not isinstance(model, str) or not model:
        message = "model is required"
        raise LookupError(message)
    selection = resolve(model, resident=state.get("record"))
    _install_runtime(state, args, selection)
    runtime = state.get("runtime")
    state["requests"] = int(state.get("requests") or 0) + 1
    if runtime is not None:
        runtime._request_id = state["requests"]
    return selection


def serve(args: Any) -> int:
    sweep_videos()
    sweep_images()
    shutdown = _Shutdown()
    shutdown.arm()
    args.shutdown = shutdown
    state: dict[str, Any] = {
        "runtime": None,
        "server": None,
        "record": None,
        "module": None,
        "pipeline": None,
        "requests": 0,
    }
    try:
        return _run_until_stopped(args, state, shutdown)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        if not shutdown.closing:
            _release(state["runtime"], shutdown, state.get("server"))


def _run_until_stopped(args: Any, state: dict[str, Any], shutdown: _Shutdown) -> int:
    """Serve until SIGINT or SIGTERM. Then close the socket and the runtime."""
    shutdown.busy = state.setdefault("lock", threading.Lock())
    try:
        return _serve_loop(args, state)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        _release(state["runtime"], shutdown, state.get("server"))


def _request_pid(handler: BaseHTTPRequestHandler) -> int:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    body = json.loads(raw or b"{}")
    if not isinstance(body, dict):
        message = "pid is required"
        raise ValueError(message)
    pid = body.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        message = "pid is required"
        raise ValueError(message)
    return pid


def _video_body(content_type: str, raw: bytes) -> dict[str, Any]:
    """JSON object, or a multipart form whose file part is a PNG."""
    if "multipart/" not in content_type.lower():
        body = json.loads(raw or b"{}")
        if not isinstance(body, dict):
            message = "request body must be an object"
            raise ValueError(message)
        return body
    marker = _boundary(content_type)
    if marker is None:
        message = "multipart boundary is missing"
        raise ValueError(message)
    body: dict[str, Any] = {}
    for chunk in raw.split(b"--" + marker):
        if chunk in (b"", b"--", b"--\r\n", b"--\n"):
            continue
        chunk = chunk.removeprefix(b"\r\n").removeprefix(b"\n")
        if chunk.startswith(b"--"):
            continue
        header_blob, sep, payload = chunk.partition(b"\r\n\r\n")
        if not sep:
            header_blob, sep, payload = chunk.partition(b"\n\n")
        if not sep:
            continue
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        elif payload.endswith(b"\n"):
            payload = payload[:-1]
        headers = header_blob.decode("utf-8", errors="replace")
        name = _form_name(headers)
        if not name:
            continue
        filename = "filename=" in headers.lower()
        if filename or (name == "input_reference" and payload.startswith(b"\x89PNG")):
            body["image"] = base64.b64encode(payload).decode("ascii")
            continue
        body[name] = payload.decode("utf-8")
    return body


def _boundary(content_type: str) -> bytes | None:
    index = content_type.lower().find("boundary=")
    if index < 0:
        return None
    value = content_type[index + len("boundary=") :].split(";", 1)[0].strip().strip('"')
    if not value:
        return None
    return value.encode("utf-8")


def _form_name(headers: str) -> str:
    for line in headers.splitlines():
        if "name=" not in line.lower():
            continue
        _, _, tail = line.partition("name=")
        tail = tail.strip()
        if tail.startswith('"'):
            return tail[1:].split('"', 1)[0]
        return tail.split(";", 1)[0].strip()
    return ""


def _query_value(path: str, name: str) -> str | None:
    values = urllib.parse.parse_qs(urllib.parse.urlparse(path).query).get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _serve_loop(args: Any, state: dict[str, Any]) -> int:
    lock = state.setdefault("lock", threading.Lock())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = _REQUEST_READ_TIMEOUT

        def log_message(self, fmt: str, *log_args: Any) -> None:
            del fmt, log_args

        def do_GET(self) -> None:  # noqa: N802
            path = urllib.parse.unquote(self.path.split("?", 1)[0])
            if path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": model_objects(resident=state.get("record")),
                    },
                )
                return
            prefix = "/v1/models/"
            if path.startswith(prefix):
                try:
                    row = one_model(path[len(prefix) :], resident=state.get("record"))
                except LookupError as exc:
                    self._json(404, {"error": {"message": str(exc), "type": "not_found"}})
                    return
                self._json(200, row)
                return
            if path.startswith("/v1/images/"):
                self._image(path)
                return
            if path.startswith("/v1/videos/"):
                self._video(path)
                return
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0]
            if route == "/v1/release":
                self._release_for_smoke()
                return
            if route == "/v1/resume":
                self._resume_from_smoke()
                return
            if route not in {
                "/v1/chat/completions",
                "/v1/images/generations",
                "/v1/videos",
            }:
                self._json(404, {"error": {"message": "not found"}})
                return
            self._started = False
            self._gone = False
            self._model_id = ""
            selection = None
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                raw = self.rfile.read(length) if length else b"{}"
            except TimeoutError:
                self.close_connection = True
                return
            self.connection.settimeout(None)
            if not lock.acquire(blocking=False):
                _log("waiting for the open request")
                lock.acquire()
            opened = False
            try:
                if _smoke_held(state):
                    self._json(
                        503,
                        {
                            "error": {
                                "message": "the gpu is held for a pipeline smoke",
                                "type": "busy",
                            }
                        },
                    )
                    return
                _log("request open")
                opened = True
                try:
                    if route == "/v1/videos":
                        body = _video_body(self.headers.get("Content-Type", ""), raw)
                    else:
                        body = json.loads(raw or b"{}")
                except (json.JSONDecodeError, ValueError) as exc:
                    if route == "/v1/videos":
                        self._json(
                            400,
                            {
                                "error": {
                                    "message": str(exc),
                                    "type": "invalid_request_error",
                                }
                            },
                        )
                        return
                    raise
                selection = _bind_request(body, state, args)
                if selection is not None:
                    self._model_id = selection.requested_id
                runtime = state.get("runtime")
                module = state.get("module")
                if runtime is None or module is None:
                    self._json(
                        404,
                        {"error": {"message": "no model is loaded", "type": "not_found"}},
                    )
                    return
                if route == "/v1/images/generations":
                    image = getattr(module, "handle_image", None)
                    if not callable(image):
                        self._json(404, {"error": {"message": "not found"}})
                        return
                    image(self, runtime, body, selection, args, state)
                    return
                if route == "/v1/videos":
                    video = getattr(module, "handle_video", None)
                    if not callable(video):
                        self._json(404, {"error": {"message": "not found"}})
                        return
                    video(self, runtime, body, selection, args, state)
                    return
                chat = getattr(module, "handle_chat", None)
                if not callable(chat):
                    name = state.get("pipeline") or "pipeline"
                    self._json(
                        500,
                        {
                            "error": {
                                "message": f"{name} does not complete a chat",
                                "type": "RuntimeError",
                            }
                        },
                    )
                    return
                chat(self, runtime, body, selection, args, state)
            except KeyError as exc:
                self.close_connection = True
                if self._started:
                    return
                self._json(404, {"error": {"message": str(exc), "type": "not_found"}})
            except LookupError as exc:
                self.close_connection = True
                if self._started:
                    return
                self._json(404, {"error": {"message": str(exc), "type": "not_found"}})
            except Exception as exc:
                _report_failure(getattr(state.get("runtime"), "console", None))
                self.close_connection = True
                if self._started:
                    return
                self._json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})
            finally:
                if opened:
                    _log("request closed")
                module = state.get("module")
                runtime = state.get("runtime")
                lock.release()
                if opened:
                    _schedule_pipeline_heartbeat(module, runtime, selection)
                try:
                    self.connection.settimeout(_REQUEST_READ_TIMEOUT)
                except OSError:
                    self.close_connection = True

        def _release_for_smoke(self) -> None:
            try:
                pid = _request_pid(self)
            except (TimeoutError, ValueError, json.JSONDecodeError):
                self.close_connection = True
                self._json(400, {"error": {"message": "pid is required", "type": "bad_request"}})
                return
            if not lock.acquire(blocking=False):
                self._json(409, {"error": {"message": "a request is open", "type": "busy"}})
                return
            try:
                holder = state.get("smoke_hold")
                if _smoke_held(state) and holder != pid:
                    self._json(
                        409,
                        {"error": {"message": "a smoke hold is active", "type": "busy"}},
                    )
                    return
                try:
                    closed = _drop_runtime(state, args)
                except Exception as exc:
                    self._json(
                        500,
                        {"error": {"message": str(exc), "type": type(exc).__name__}},
                    )
                    return
                state["smoke_hold"] = pid
                self._json(200, {"released": True, "closed": closed})
            finally:
                lock.release()

        def _resume_from_smoke(self) -> None:
            try:
                pid = _request_pid(self)
            except (TimeoutError, ValueError, json.JSONDecodeError):
                self.close_connection = True
                self._json(400, {"error": {"message": "pid is required", "type": "bad_request"}})
                return
            if not lock.acquire(blocking=False):
                self._json(409, {"error": {"message": "a request is open", "type": "busy"}})
                return
            try:
                holder = state.get("smoke_hold")
                if _smoke_held(state) and holder != pid:
                    self._json(
                        409,
                        {"error": {"message": "a smoke hold is active", "type": "busy"}},
                    )
                    return
                state["smoke_hold"] = None
                self._json(200, {"resumed": True})
            finally:
                lock.release()

        def _client_stopped(self) -> bool:
            if self._gone:
                return True
            if _peer_closed(self.connection):
                self._gone = True
                return True
            return False

        def _event(self, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            self._send(b"data: " + blob + b"\n\n")

        def _send(self, data: bytes) -> None:
            try:
                if getattr(self, "_chunked", False):
                    if data:
                        data = f"{len(data):X}\r\n".encode() + data + b"\r\n"
                    else:
                        data = b"0\r\n\r\n"
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:
                self._gone = True
                message = "client closed"
                raise OSError(message) from None

        def _image(self, path: str) -> None:
            rest = path[len("/v1/images/") :]
            if not rest.endswith("/content"):
                self._json(404, {"error": {"message": "not found"}})
                return
            image_id = rest[: -len("/content")]
            if not image_id or "/" in image_id:
                self._json(404, {"error": {"message": "not found"}})
                return
            home = state.get("image_home")
            blob = load_image(image_id, home=home if isinstance(home, Path) else None)
            if blob is None:
                self._json(404, {"error": {"message": "image not found", "type": "not_found"}})
                return
            self._bytes(200, blob, "image/png")

        def _video(self, path: str) -> None:
            rest = path[len("/v1/videos/") :]
            content = rest.endswith("/content")
            video_id = rest[: -len("/content")] if content else rest
            if not video_id or "/" in video_id:
                self._json(404, {"error": {"message": "not found"}})
                return
            videos = state.get("videos")
            job = videos.get(video_id) if isinstance(videos, dict) else None
            if not isinstance(job, dict):
                self._json(404, {"error": {"message": "video not found", "type": "not_found"}})
                return
            if not content:
                self._json(200, job)
                return
            if job.get("status") != "completed":
                self._json(
                    400,
                    {
                        "error": {
                            "message": "video is not ready",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            home = state.get("video_home")
            root = home if isinstance(home, Path) else None
            raw = _query_value(self.path, "quality")
            chosen: str | None = None
            if raw is not None:
                try:
                    chosen = reported_quality(raw)
                except ValueError as exc:
                    self._json(
                        400,
                        {
                            "error": {
                                "message": str(exc),
                                "type": "invalid_request_error",
                                "param": "quality",
                            }
                        },
                    )
                    return
                if chosen == job.get("quality"):
                    chosen = None
            blob = load_video(video_id, home=root, quality=chosen)
            if blob is None:
                self._json(404, {"error": {"message": "video not found", "type": "not_found"}})
                return
            self._bytes(200, blob, "video/mp4")

        def _bytes(self, status: int, payload: bytes, content_type: str) -> None:
            previous = self.connection.gettimeout()
            self.connection.settimeout(_STREAM_WRITE_TIMEOUT)
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                _log("client closed")
            else:
                self.connection.settimeout(previous)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            blob = json.dumps(payload).encode()
            previous = self.connection.gettimeout()
            self.connection.settimeout(_STREAM_WRITE_TIMEOUT)
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
                self.wfile.flush()
            except OSError:
                self.close_connection = True
                _log("client closed")
            else:
                self.connection.settimeout(previous)

    server = _SingleRequestServer((args.host, args.port), Handler)
    state["server"] = server
    _log(f"listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


def serve_main(argv: Sequence[str] | None = None) -> int:
    """``favlo serve``. The daemon listens. A terminal runs the CLI."""
    parser = argparse.ArgumentParser(prog="favlo serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8129)
    parser.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    parser.add_argument("--transcript", type=Path, default=None)
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="start the server daemon and exit",
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help="maximum bytes for a pipeline that caps its runtime, such as 32GiB",
    )
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if os.environ.get(DAEMON_ENV) == "1":
        return _daemon_main(args)
    if args.non_interactive:
        return _start_daemon(args)
    return _interactive(args)


def _daemon_main(args: Any) -> int:
    """The listener process. Telemetry is JSON on stdout. Stdin is closed."""
    ui = Console("json", pipeline="", stream=sys.stdout)
    attach_process_log(ui)
    bind_log(ui.log)
    ui.live_decode = True
    ui.plate()
    args.lazy = True
    args.console_ui = ui
    args.record = None
    args.resident_percent = None
    try:
        return serve(args)
    finally:
        bind_log(None)
        ui.close()
        from favlo.daemon import release_pid

        release_pid(args.port, os.getpid())


def _start_daemon(args: Any) -> int:
    """Spawn the daemon when this address is free, print the listening line, and return."""
    from favlo.daemon import DaemonError, ensure

    try:
        ensure(
            args.host,
            args.port,
            max_memory=args.max_memory,
            transcript=args.transcript,
        )
    except DaemonError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"listening on http://{args.host}:{args.port}\n")
    sys.stdout.flush()
    return 0


def _interactive(args: Any) -> int:
    """First-run setup on a terminal, then chat against the daemon."""
    from favlo.cli import repo_root
    from favlo.daemon import DaemonError, ensure, follow_events, stop_pid, telemetry_path
    from favlo.setup import prepare
    from favlo.talk import Idle, start

    if sys.stdin.isatty():
        code = prepare(repo_root(), stdin_tty=True, console=args.console)
        if code != 0:
            return code
    ui = Console.from_env(args.console, pipeline="")
    ui.plate()
    try:
        try:
            pid, created = ensure(
                args.host,
                args.port,
                max_memory=args.max_memory,
                transcript=args.transcript,
            )
        except DaemonError as exc:
            ui.log(str(exc))
            return 1
        path = telemetry_path(args.port)
        offset = 0 if created or not path.is_file() else path.stat().st_size
        idle = Idle()
        stopped = threading.Event()

        def on_event(event: dict[str, Any]) -> None:
            if event.get("type") == "log" and event.get("line") == f"favlo {VERSION}":
                return
            ui.apply_event(event)

        follow_events(path, on_event, stopped, offset=offset, on_idle=ui.note_caught_up)

        def stop() -> None:
            stopped.set()
            stop_pid(pid)

        thread = start(
            args.host,
            args.port,
            sys.stdin,
            ui.say,
            idle,
            stop,
            draft=ui.draft,
            prompt=ui.prompt,
            hold=ui.hold,
        )
        try:
            thread.join()
        except KeyboardInterrupt:
            if created:
                stop()
            return 0
        finally:
            stopped.set()
        return 0
    finally:
        ui.close()
