"""The serve daemon.

A terminal ``favlo serve`` is this process and paints progress on stderr.
``favlo serve --non-interactive`` starts a detached daemon and returns. The
detached daemon writes JSON telemetry and does not read stdin.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home

DAEMON_ENV = "FAVLO_SERVE_DAEMON"
_READY_SECONDS = 60.0
_CLAIM = "0"


class DaemonError(RuntimeError):
    """The daemon did not become the listener on the requested address."""


def pid_path(port: int, home: Path | None = None) -> Path:
    root = favlo_home() if home is None else home
    return root / "serve" / f"{port}.pid"


def telemetry_path(port: int, home: Path | None = None) -> Path:
    root = favlo_home() if home is None else home
    return root / "logs" / f"serve-{port}.log"


def read_pid(port: int, home: Path | None = None) -> int | None:
    """The recorded pid, or None when the file is missing or not a live claim."""
    path = pid_path(port, home)
    try:
        text = path.read_text(encoding="utf-8").strip()
        pid = int(text)
    except (OSError, UnicodeError, ValueError):
        return None
    if pid <= 0:
        return None
    return pid


def alive(pid: int) -> bool:
    """True when ``pid`` is a running process. A zombie child counts as gone."""
    if pid <= 0:
        return False
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        waited = 0
    if waited == pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    """True when a connection to ``host:port`` is accepted."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def plain_line(raw: str) -> str | None:
    """One durable line from a telemetry record. Ticks are dropped."""
    text = raw.strip()
    if not text:
        return None
    if not text.startswith("{"):
        return text
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(event, dict):
        return None
    if event.get("type") == "log" and isinstance(event.get("line"), str):
        return event["line"]
    summary = event.get("summary")
    if event.get("type") == "end" and isinstance(summary, str) and summary:
        return summary
    return None


def plain_text(blob: str) -> str:
    """Durable lines from a telemetry blob."""
    lines = [line for raw in blob.splitlines() if (line := plain_line(raw))]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


class TracebackSkip:
    """A shutdown traceback stays in the log file and off the pretty console."""

    def __init__(self) -> None:
        self.open = False

    def drop(self, raw: str) -> bool:
        text = raw.strip()
        if not text:
            return self.open
        if text.startswith("{"):
            self.open = False
            return False
        if _traceback_header(text) or _traceback_marker(text) or (self.open and _exception_line(text)):
            self.open = True
            return True
        if self.open and _traceback_continuation(raw):
            return True
        self.open = False
        return False


def _traceback_header(text: str) -> bool:
    if text in {"Exception in ASGI application", "Traceback (most recent call last):"}:
        return True
    if text.startswith("During handling of the above exception"):
        return True
    if text.startswith("The above exception was the direct cause"):
        return True
    if text.startswith("Cancel ") and "running task" in text:
        return True
    if text == "ASGI callable returned without starting response.":
        return True
    return text.endswith("graceful shutdown exceeded")


def _traceback_marker(text: str) -> bool:
    if text.startswith("During handling of the above exception"):
        return True
    if text.startswith("The above exception was the direct cause"):
        return True
    if text.startswith('File "') or text.startswith("File '"):
        return True
    if set(text) <= {"^"}:
        return True
    return "CancelledError" in text


def _exception_line(text: str) -> bool:
    head = text.split(":", 1)[0]
    return head.endswith("Error") or head.endswith("Exception")


def _traceback_continuation(raw: str) -> bool:
    text = raw.strip()
    if not text or raw[:1].isspace():
        return True
    if text.startswith(("return ", "raise ", "yield ", "except ", "finally:", "else:", "try:")):
        return True
    return "await " in text or text == ")" or text.startswith(")")


def telemetry_event(raw: str) -> dict[str, Any] | None:
    """One telemetry object. A plain line is a log event. Ticks stay in the object."""
    text = raw.strip()
    if not text:
        return None
    if not text.startswith("{"):
        return {"type": "log", "line": text}
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return {"type": "log", "line": text}
    if isinstance(event, dict) and isinstance(event.get("type"), str):
        return event
    return None


def _follow(
    path: Path,
    stop: threading.Event,
    offset: int,
    take: Callable[[str], None],
    on_idle: Callable[[], None] | None = None,
) -> threading.Thread:
    def run() -> None:
        position = offset
        pending = b""
        while not stop.is_set():
            size = position
            if path.is_file():
                try:
                    size = path.stat().st_size
                except OSError:
                    size = position
                if size < position:
                    position = 0
                if size > position:
                    try:
                        with path.open("rb") as handle:
                            handle.seek(position)
                            chunk = handle.read(size - position)
                    except OSError:
                        chunk = b""
                    else:
                        position = size
                    pending += chunk
                    while b"\n" in pending:
                        raw, pending = pending.split(b"\n", 1)
                        take(raw.decode("utf-8", errors="replace"))
            if on_idle is not None and b"\n" not in pending and position >= size:
                if path.is_file():
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = position
                if position >= size:
                    on_idle()
            if stop.wait(0.05):
                return

    thread = threading.Thread(target=run, name="favlo-telemetry", daemon=True)
    thread.start()
    return thread


def follow(
    path: Path,
    on_line: Callable[[str], None],
    stop: threading.Event,
    *,
    offset: int = 0,
    on_idle: Callable[[], None] | None = None,
) -> threading.Thread:
    """Call ``on_line`` with each new durable line until ``stop`` is set."""

    def take(raw: str) -> None:
        line = plain_line(raw)
        if line:
            on_line(line)

    return _follow(path, stop, offset, take, on_idle)


def follow_events(
    path: Path,
    on_event: Callable[[dict[str, Any]], None],
    stop: threading.Event,
    *,
    offset: int = 0,
    on_idle: Callable[[], None] | None = None,
) -> threading.Thread:
    """Call ``on_event`` with each telemetry object, including ticks."""

    skip = TracebackSkip()

    def take(raw: str) -> None:
        if skip.drop(raw):
            return
        event = telemetry_event(raw)
        if event is not None:
            on_event(event)

    return _follow(path, stop, offset, take, on_idle)


def stop_pid(pid: int) -> None:
    """SIGTERM the daemon process group, then SIGKILL if it stays up."""
    if not alive(pid):
        return
    _signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if not alive(pid):
                return
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    if alive(pid):
        _signal(pid, signal.SIGKILL)


def release_pid(port: int, pid: int, home: Path | None = None) -> None:
    """Remove the pid file when it still names ``pid``."""
    path = pid_path(port, home)
    if read_pid(port, home) != pid:
        return
    try:
        path.unlink()
    except OSError:
        return


def occupy(host: str, port: int, *, home: Path | None = None) -> None:
    """Claim this process as the listener. Raise when that address is taken."""
    existing = read_pid(port, home)
    if existing is not None and alive(existing) and existing != os.getpid():
        raise DaemonError(f"http://{host}:{port} is already in use")
    if port_open(host, port):
        raise DaemonError(f"http://{host}:{port} is already in use")
    if _starting(port, home):
        raise DaemonError(f"http://{host}:{port} is already in use")
    _clear_pid(port, home)
    if not _claim(port, home):
        raise DaemonError(f"http://{host}:{port} is already in use")
    pid_path(port, home).write_text(f"{os.getpid()}\n", encoding="utf-8")


def ensure(
    host: str,
    port: int,
    *,
    max_memory: str | None = None,
    transcript: Path | None = None,
    home: Path | None = None,
    root: Path | None = None,
) -> tuple[int, bool]:
    """Return ``(pid, started)``. ``started`` is false when a daemon is already listening."""
    from favlo.cli import repo_root

    checkout = repo_root() if root is None else root
    for _attempt in range(100):
        existing = read_pid(port, home)
        if existing is not None and alive(existing):
            if _wait_port(host, port, _READY_SECONDS):
                return existing, False
            message = f"server process {existing} is not listening on {host}:{port}"
            raise DaemonError(message)
        if port_open(host, port):
            message = f"http://{host}:{port} is already in use"
            raise DaemonError(message)
        if _starting(port, home):
            time.sleep(0.1)
            continue
        _clear_pid(port, home)
        if not _claim(port, home):
            continue
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = _spawn(
                host,
                port,
                max_memory=max_memory,
                transcript=transcript,
                home=home,
                root=checkout,
            )
            pid_path(port, home).write_text(f"{proc.pid}\n", encoding="utf-8")
            if _wait_ready(proc, host, port, _READY_SECONDS):
                return proc.pid, True
            tail = _tail(port, home)
            stop_pid(proc.pid)
            release_pid(port, proc.pid, home)
            raise DaemonError(tail or "server did not start")
        except DaemonError:
            raise
        except OSError as exc:
            if proc is not None:
                stop_pid(proc.pid)
            _clear_pid(port, home)
            raise DaemonError(str(exc)) from exc
    message = f"server did not become ready on {host}:{port}"
    raise DaemonError(message)


def _signal(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        return
    except (PermissionError, OSError):
        try:
            os.kill(pid, signum)
        except OSError:
            return


def _wait_port(host: str, port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if port_open(host, port):
            return True
        time.sleep(0.05)
    return False


def _wait_ready(proc: subprocess.Popen[bytes], host: str, port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if port_open(host, port):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return port_open(host, port)


def _starting(port: int, home: Path | None) -> bool:
    path = pid_path(port, home)
    try:
        text = path.read_text(encoding="utf-8").strip()
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return text == _CLAIM and age < _READY_SECONDS


def _clear_pid(port: int, home: Path | None) -> None:
    path = pid_path(port, home)
    try:
        path.unlink()
    except OSError:
        return


def _claim(port: int, home: Path | None) -> bool:
    path = pid_path(port, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{_CLAIM}\n".encode())
    finally:
        os.close(fd)
    return True


def _spawn(
    host: str,
    port: int,
    *,
    max_memory: str | None,
    transcript: Path | None,
    home: Path | None,
    root: Path,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env[DAEMON_ENV] = "1"
    env["FAVLO_ROOT"] = str(root)
    env["FAVLO_SKIP_PLATE"] = "1"
    command = [
        sys.executable,
        "-m",
        "favlo",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--console",
        "json",
    ]
    if max_memory:
        command.extend(["--max-memory", max_memory])
    if transcript is not None:
        command.extend(["--transcript", str(transcript)])
    path = telemetry_path(port, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    env["FAVLO_LOG_FILE"] = str(path)
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=str(root),
        start_new_session=True,
    )


def _tail(port: int, home: Path | None) -> str:
    path = telemetry_path(port, home)
    try:
        blob = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = plain_text(blob).strip()
    if not text:
        text = blob.strip()
    lines = text.splitlines()
    return "\n".join(lines[-20:])
