"""The one listener. A request names a pipeline, and that pipeline loads.

This module does not import a pipeline. ``open_runtime`` and ``handle_chat``
come from the entry file the registry row names.
"""

from __future__ import annotations

import argparse
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
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from favlo.console import VERSION, Console
from favlo.daemon import DAEMON_ENV
from favlo.logs import attach_process_log
from favlo.registry import model_objects, one_model, resolve

_STREAM_WRITE_TIMEOUT = 2.0
_REQUEST_READ_TIMEOUT = 10.0
_log_sink: Any = None


def bind_log(sink: Any) -> None:
    """Route listener lines through the console. ``None`` restores stderr."""
    global _log_sink
    _log_sink = sink


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


class _SingleRequestServer(HTTPServer):
    """Accept the next connection only after the current request returns."""

    request_queue_size = 128

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A peer that closes mid-response is not a server failure."""
        if isinstance(sys.exception(), (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class _Shutdown:
    """SIGINT and SIGTERM stop the forward and leave the process able to release."""

    def __init__(self) -> None:
        self.runtime: Any = None
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
            runtime.close()
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
    try:
        return _serve_loop(args, state)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        _release(state["runtime"], shutdown, state.get("server"))


def _serve_loop(args: Any, state: dict[str, Any]) -> int:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = _REQUEST_READ_TIMEOUT

        def log_message(self, fmt: str, *log_args: Any) -> None:
            del fmt, log_args

        def do_GET(self) -> None:  # noqa: N802
            self.connection.settimeout(None)
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
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._json(404, {"error": {"message": "not found"}})
                return
            self._started = False
            self._gone = False
            self._model_id = ""
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                raw = self.rfile.read(length) if length else b"{}"
            except TimeoutError:
                self.close_connection = True
                return
            self.connection.settimeout(None)
            _log("request open")
            try:
                body = json.loads(raw or b"{}")
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
                with lock:
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
                _log("request closed")

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
            if event.get("type") == "log":
                line = event.get("line")
                if line == f"favlo {VERSION}":
                    return
                if line == "request open":
                    idle.enter()
                    return
                if line == "request closed":
                    idle.note_closed()
                    return
            ui.apply_event(event)

        follow_events(path, on_event, stopped, offset=offset)

        def stop() -> None:
            stopped.set()
            stop_pid(pid)

        def on_call() -> Callable[[], None]:
            progress = ui.begin("load", total=0, unit="bytes", detail="")

            def finish() -> None:
                if progress.closed or progress.request is not None:
                    return
                ui.end(progress, "")

            return finish

        thread = start(args.host, args.port, sys.stdin, ui.say, idle, stop, on_call=on_call)
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
