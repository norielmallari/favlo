"""The one listener. It accepts a request and writes the reply.

The pipeline manager chooses the runtime, admits the request, and sweeps
that runtime's owners after close and after a crash. This module does not
import a pipeline and does not open one. After the request closes, a pipeline
that defines ``idle_heartbeat`` runs that function here. A terminal runs this
process. ``--non-interactive`` starts a detached daemon and returns.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import threading
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.console import Console
from favlo.daemon import DAEMON_ENV, alive
from favlo.images import sweep_images
from favlo.logs import attach_process_log
from favlo.music import sweep_music
from favlo.pipelines import pipelines
from favlo.videos import sweep_videos
from favlo.voice import sweep_voice

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


def _release(state: dict[str, Any], args: Any, shutdown: _Shutdown) -> None:
    """Close the listening socket and every runtime. A signal during this returns."""
    shutdown.closing = True
    try:
        server = state.get("server")
        had_runtime = state.get("runtime") is not None or shutdown.runtime is not None
        if server is not None:
            try:
                server.server_close()
            except OSError:
                pass
        pipelines.close(state, args, wait=_RELEASE_WAIT)
        if shutdown.signaled and (server is not None or had_runtime):
            _log("released")
    finally:
        shutdown.disarm()



def _smoke_held(state: dict[str, Any]) -> bool:
    """True when a live smoke pid still owns the GPU. A dead pid clears the hold."""
    holder = state.get("smoke_hold")
    if isinstance(holder, bool) or not isinstance(holder, int) or holder <= 0:
        return False
    if not alive(holder):
        state["smoke_hold"] = None
        return False
    return True


def serve(args: Any) -> int:
    sweep_videos()
    sweep_images()
    sweep_music()
    sweep_voice()
    shutdown = _Shutdown()
    shutdown.arm()
    args.shutdown = shutdown
    memory_home = os.environ.get("FAVLO_MEMORY_HOME")
    state: dict[str, Any] = {
        "runtime": None,
        "server": None,
        "record": None,
        "module": None,
        "pipeline": None,
        "requests": 0,
    }
    if memory_home:
        state["memory_home"] = Path(memory_home)
    try:
        return _run_until_stopped(args, state, shutdown)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        if not shutdown.closing:
            _release(state, args, shutdown)


def _run_until_stopped(args: Any, state: dict[str, Any], shutdown: _Shutdown) -> int:
    """Serve until SIGINT or SIGTERM. Then close the socket and the runtime."""
    shutdown.busy = state.setdefault("lock", threading.Lock())
    try:
        return _serve_loop(args, state)
    except KeyboardInterrupt:
        _log("shutting down")
        return 0
    finally:
        _release(state, args, shutdown)


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


def _serve_loop(args: Any, state: dict[str, Any]) -> int:
    """Listen on one uvicorn server. Chat and MCP share that process."""
    from favlo.http_app import serve

    state.setdefault("lock", threading.Lock())
    return serve(args, state)


def serve_main(argv: Sequence[str] | None = None) -> int:
    """``favlo serve``. A terminal is the daemon. ``--non-interactive`` detaches it."""
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
    return _foreground(args)


def _listen(args: Any, ui: Console) -> int:
    """Run the listener on this process. Progress uses ``ui``."""
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


def _daemon_main(args: Any) -> int:
    """Detached listener. Telemetry is JSON on stdout. Stdin is closed."""
    ui = Console("json", pipeline="", stream=sys.stdout)
    return _listen(args, ui)


def _foreground(args: Any) -> int:
    """This process is the listener. A terminal paints progress on stderr."""
    from favlo.cli import repo_root
    from favlo.daemon import DaemonError, occupy
    from favlo.setup import prepare

    if sys.stdin.isatty():
        code = prepare(repo_root(), stdin_tty=True, console=args.console)
        if code != 0:
            return code
    try:
        occupy(args.host, args.port)
    except DaemonError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    ui = Console.from_env(args.console, pipeline="")
    return _listen(args, ui)


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
