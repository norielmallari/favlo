"""Take the GPU for a pipeline smoke without stopping favlo serve.

``acquire`` closes the resident model when the daemon is up and no request
is open, then writes ``artifacts/pipeline-test.lock``. ``release`` clears
that hold. The daemon process stays up. The next chat loads a model again.

``attach`` does not take that lock. It reuses ``127.0.0.1:8129``, starts the
daemon only when that port is closed, and stops the daemon only when this
call started it. When ``--pipeline`` is the resident model and that
pipeline's sources are newer than its worker, ``attach`` closes the model
before the command. The listener does the same check on the next request.

``run`` is the in-process GPU path. It holds the lock for the command.
A harness that calls the listener uses ``attach``.

Exit 0 means the lock is held, was cleared, or the attached command exited 0.
Exit 2 means a request is open. The lock was not taken.
Exit 3 means another live pid holds the lock.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from favlo.cli import repo_root
from favlo.daemon import alive

_HOST = "127.0.0.1"
_PORT = 8129
_CLOSE_TIMEOUT = 120.0


def lock_path(root: Path) -> Path:
    return root / "artifacts" / "pipeline-test.lock"


def owner_pid() -> int:
    """Process id of the shell that started this command.

    ``uv run`` and the Python interpreter sit between that shell and this
    module. Those parents exit when the module returns, so they are not the
    owner.
    """
    current = os.getpid()
    seen: set[int] = set()
    while current > 1 and current not in seen:
        seen.add(current)
        if not _launcher(current):
            return current
        parent = _parent(current)
        if parent <= 1:
            return current
        current = parent
    return os.getppid()


def hold(
    *,
    root: Path,
    pipeline: str,
    command: str,
    host: str = _HOST,
    port: int = _PORT,
    pid: int | None = None,
    wait: float = 5.0,
) -> bool:
    """Wait until this shell holds the lock.

    Return True when this call created the file. A lock this shell already
    holds stays as it is, and the return is False, so a nested command does
    not release the caller's hold.
    """
    owner = owner_pid() if pid is None else pid
    while True:
        holder = _read_pid(lock_path(root))
        if holder == owner and alive(holder):
            return False
        code = acquire(
            root=root,
            pipeline=pipeline,
            command=command,
            host=host,
            port=port,
            pid=owner,
        )
        if code == 0:
            return True
        if code not in {2, 3}:
            raise RuntimeError(f"lock status {code}")
        time.sleep(wait)


def acquire(
    *,
    root: Path,
    pipeline: str,
    command: str,
    host: str = _HOST,
    port: int = _PORT,
    pid: int | None = None,
) -> int:
    """Close an idle resident model, then hold the machine lock."""
    owner = owner_pid() if pid is None else pid
    status = _claim(root, pipeline, command, owner)
    if status != 0:
        return status
    code, body = _post(host, port, "/v1/release", {"pid": owner})
    if code == 0:
        print("serve is not listening")
        print("lock held")
        return 0
    if code == 200:
        closed = body.get("closed")
        if isinstance(closed, str) and closed:
            print(f"released {closed}")
        print("lock held")
        return 0
    _drop_claim(root, owner)
    if code == 409:
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        print(message if isinstance(message, str) and message else "a request is open")
        return 2
    print("release failed")
    return 1


def clear_hold(
    *,
    host: str = _HOST,
    port: int = _PORT,
    pid: int | None = None,
) -> int:
    """Clear the listener smoke hold. The lock file stays."""
    owner = owner_pid() if pid is None else pid
    code, body = _post(host, port, "/v1/resume", {"pid": owner})
    if code == 200:
        return 0
    if code == 409:
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        print(message if isinstance(message, str) and message else "a request is open")
        return 2
    if code == 0:
        print("serve is not listening")
        return 1
    print("resume failed")
    return 1


def release(
    *,
    root: Path,
    host: str = _HOST,
    port: int = _PORT,
    pid: int | None = None,
    stale: bool = False,
) -> int:
    """Clear the smoke hold and remove this shell's lock.

    A dead recorded pid is the same path. ``stale`` names that recovery.
    A live pid that is not this shell is left in place.
    """
    owner = owner_pid() if pid is None else pid
    path = lock_path(root)
    holder = _read_pid(path)
    if holder is not None and holder != owner and alive(holder):
        print(f"lock held by {holder}")
        return 3
    resume_pid = holder if holder is not None and not alive(holder) else owner
    code, _body = _post(host, port, "/v1/resume", {"pid": resume_pid})
    if code not in {0, 200}:
        if code == 409:
            print("a request is open")
            return 2
        print("resume failed")
        return 1
    if holder is None or holder == owner or not alive(holder):
        _drop_claim(root, owner if holder == owner else holder or owner)
    if code == 0:
        print("serve is not listening")
    recovered = stale and holder is not None and not alive(holder)
    print("stale lock released" if recovered else "lock released")
    return 0


@dataclass(frozen=True)
class Attached:
    """The listener a command is using. ``owned`` is true when this call started it."""

    host: str
    port: int
    owned: bool
    pid: int | None = None


def open_attached(
    *,
    root: Path,
    pipeline: str = "",
    host: str = _HOST,
    port: int = _PORT,
    max_memory: str | None = None,
) -> Attached:
    """Reuse the listener, or start one. Unload ``pipeline`` when its sources changed.

    This call does not take the machine lock. It waits until no live pid holds it.
    """
    from favlo.daemon import ensure, port_open

    wait_clear(root=root)
    if port_open(host, port):
        print(f"reuse {host}:{port}", flush=True)
        attached = Attached(host=host, port=port, owned=False)
    else:
        pid, started = ensure(host, port, max_memory=max_memory, root=root)
        if started:
            print(f"started {host}:{port}", flush=True)
            attached = Attached(host=host, port=port, owned=True, pid=pid)
        else:
            print(f"reuse {host}:{port}", flush=True)
            attached = Attached(host=host, port=port, owned=False, pid=pid)
    if pipeline and unload_if_changed(root=root, pipeline=pipeline, host=host, port=port):
        print(f"unloaded {pipeline}", flush=True)
    return attached


def unload_if_changed(
    *,
    root: Path,
    pipeline: str,
    host: str = _HOST,
    port: int = _PORT,
) -> bool:
    """Close the resident model when ``pipeline`` sources are newer than its worker.

    A stamp that names a different model stays until the next request names
    this one. With no stamp, the newest pipeline worker's start time is the
    clock. A request that is open is left running. The listener reloads on
    the next request for this pipeline.
    """
    from favlo.daemon import read_resident_stamp
    from favlo.pipelines import source_mtime
    from favlo.registry import resolve

    stamp = read_resident_stamp(port)
    if stamp is not None and stamp.get("pipeline") != pipeline:
        return False
    started = stamp.get("started") if stamp is not None else _worker_started()
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        return False
    try:
        selection = resolve(pipeline, root=root)
    except (LookupError, OSError, ValueError):
        return False
    if source_mtime(selection) <= float(started):
        return False
    code, body = _post(host, port, "/v1/pipelines/close", {})
    if code == 200:
        return True
    if code == 409:
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        print(message if isinstance(message, str) and message else "a request is open")
    elif code == 0:
        print("serve is not listening")
    else:
        print("unload failed")
    return False


def _worker_started() -> float | None:
    """Start time of the newest pipeline worker, or None when none is running.

    Used when the listener has not written a resident stamp. The clock is
    ``time.time()`` minus the elapsed time ``ps`` reports.
    """
    try:
        completed = subprocess.run(
            ["ps", "-axww", "-o", "etime=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    newest: float | None = None
    now = time.time()
    for line in completed.stdout.splitlines():
        if "favlo.pipeline_worker" not in line:
            continue
        elapsed = _etime_seconds(line.split(None, 1)[0])
        if elapsed is None:
            continue
        started = now - elapsed
        if newest is None or started > newest:
            newest = started
    return newest


def _etime_seconds(text: str) -> int | None:
    """Parse ``ps`` elapsed time ``[[dd-]hh:]mm:ss``."""
    days = 0
    clock = text.strip()
    if "-" in clock:
        day, clock = clock.split("-", 1)
        try:
            days = int(day)
        except ValueError:
            return None
    try:
        pieces = [int(part) for part in clock.split(":")]
    except ValueError:
        return None
    if len(pieces) == 2:
        hours, minutes, seconds = 0, pieces[0], pieces[1]
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def unload_model(*, host: str = _HOST, port: int = _PORT) -> str | None:
    """Close the resident model. The listener stays up.

    Return the closed folder, or None when nothing was closed.
    """
    code, body = _post(host, port, "/v1/pipelines/close", {})
    if code == 200:
        closed = body.get("closed")
        return closed if isinstance(closed, str) else ""
    if code == 409:
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        print(message if isinstance(message, str) and message else "a request is open")
        return None
    if code == 0:
        print("serve is not listening")
        return None
    print("unload failed")
    return None


def close_attached(attached: Attached) -> None:
    """Stop a listener this call started. A reused listener stays up."""
    if not attached.owned or attached.pid is None:
        return
    from favlo.daemon import stop_pid

    stop_pid(attached.pid)
    print(f"stopped {attached.host}:{attached.port}", flush=True)


def attach(
    *,
    root: Path,
    pipeline: str,
    command: str,
    argv: list[str],
    host: str = _HOST,
    port: int = _PORT,
    max_memory: str | None = None,
) -> int:
    """Reuse the listener, unload a changed ``pipeline``, run ``argv``, then leave it.

    ``command`` is recorded only in the log line. This path does not write the lock.
    """
    del command
    if not argv:
        print("a command is required")
        return 2
    attached = open_attached(
        root=root,
        pipeline=pipeline,
        host=host,
        port=port,
        max_memory=max_memory,
    )
    return _run_child(argv, finally_fn=lambda: close_attached(attached))


def wait_clear(
    *,
    root: Path,
    wait: float = 5.0,
) -> None:
    """Block until no live pid holds the machine lock. This call does not take it."""
    while True:
        holder = _read_pid(lock_path(root))
        if holder is None or not alive(holder):
            return
        print(f"lock held by {holder}", flush=True)
        time.sleep(wait)


def run(
    *,
    root: Path,
    pipeline: str,
    command: str,
    argv: list[str],
    host: str = _HOST,
    port: int = _PORT,
    wait: float = 5.0,
) -> int:
    """Hold the lock, run ``argv``, and release when this call created the claim."""
    if not argv:
        print("a command is required")
        return 2
    created = hold(
        root=root,
        pipeline=pipeline,
        command=command,
        host=host,
        port=port,
        wait=wait,
    )

    def _finish() -> None:
        if created:
            release(root=root, host=host, port=port)

    return _run_child(argv, finally_fn=_finish)


def _run_child(argv: list[str], *, finally_fn: Any) -> int:
    """Run ``argv`` and call ``finally_fn`` on every exit, including SIGINT."""
    child: subprocess.Popen[bytes] | None = None
    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def _on_signal(signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise KeyboardInterrupt

    code = 1
    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        child = subprocess.Popen(argv)
        code = int(child.wait())
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        code = 130
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            finally_fn()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return code


def _parent(pid: int) -> int:
    text = _ps("ppid=", pid)
    try:
        return int(text or "0")
    except ValueError:
        return 0


def _launcher(pid: int) -> bool:
    name = Path(_ps("comm=", pid)).name.lower()
    return name == "uv" or name.startswith("python")


def _ps(column: str, pid: int) -> str:
    try:
        completed = subprocess.run(
            ["ps", "-o", column, "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _claim(root: Path, pipeline: str, command: str, pid: int) -> int:
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pipeline": pipeline, "pid": pid, "command": command}
    for _ in range(3):
        holder = _read_pid(path)
        if holder == pid:
            return 0
        if holder is not None and alive(holder):
            print(f"lock held by {holder}")
            return 3
        if path.exists():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        try:
            os.write(fd, (json.dumps(payload) + "\n").encode())
        finally:
            os.close(fd)
        if _read_pid(path) != pid:
            return 1
        return 0
    print("lock held")
    return 3


def _drop_claim(root: Path, pid: int) -> None:
    path = lock_path(root)
    holder = _read_pid(path)
    if holder is not None and holder != pid:
        return
    path.unlink(missing_ok=True)


def _read_pid(path: Path) -> int | None:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    pid = body.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    return pid


def _post(host: str, port: int, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_CLOSE_TIMEOUT) as response:
            raw = response.read()
            body = json.loads(raw) if raw else {}
            return response.status, body if isinstance(body, dict) else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}
        return exc.code, body if isinstance(body, dict) else {}
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m favlo.smoke_lock")
    parser.add_argument("--host", default=_HOST)
    parser.add_argument("--port", type=int, default=_PORT)
    sub = parser.add_subparsers(dest="action", required=True)
    take = sub.add_parser("acquire")
    take.add_argument("--pipeline", required=True)
    take.add_argument("--command", required=True)
    take.add_argument("--wait", nargs="?", const=5.0, default=None, type=float)
    unlock = sub.add_parser("release")
    unlock.add_argument("--stale", action="store_true")
    wrapped = sub.add_parser("run")
    wrapped.add_argument("--pipeline", required=True)
    wrapped.add_argument("--command", required=True)
    wrapped.add_argument("--wait", type=float, default=5.0)
    wrapped.add_argument("argv", nargs=argparse.REMAINDER)
    joined = sub.add_parser("attach")
    joined.add_argument("--pipeline", required=True)
    joined.add_argument("--command", required=True)
    joined.add_argument("--max-memory", default=None)
    joined.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = repo_root()
    if args.action == "acquire":
        if args.wait is not None and args.wait > 0:
            hold(
                root=root,
                pipeline=args.pipeline,
                command=args.command,
                host=args.host,
                port=args.port,
                wait=args.wait,
            )
            return 0
        return acquire(
            root=root,
            pipeline=args.pipeline,
            command=args.command,
            host=args.host,
            port=args.port,
        )
    if args.action == "attach":
        command = list(args.argv)
        if command[:1] == ["--"]:
            command = command[1:]
        return attach(
            root=root,
            pipeline=args.pipeline,
            command=args.command,
            argv=command,
            host=args.host,
            port=args.port,
            max_memory=args.max_memory,
        )
    if args.action == "run":
        command = list(args.argv)
        if command[:1] == ["--"]:
            command = command[1:]
        return run(
            root=root,
            pipeline=args.pipeline,
            command=args.command,
            argv=command,
            host=args.host,
            port=args.port,
            wait=args.wait,
        )
    return release(root=root, host=args.host, port=args.port, stale=args.stale)


if __name__ == "__main__":
    sys.exit(main())
