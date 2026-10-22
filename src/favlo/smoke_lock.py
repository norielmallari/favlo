"""Take the GPU for a pipeline smoke without stopping favlo serve.

``acquire`` closes the resident model when the daemon is up and no request
is open, then writes ``artifacts/pipeline-test.lock``. ``release`` clears
that hold. The daemon process stays up. The next chat loads a model again.

Exit 0 means the lock is held, or was cleared.
Exit 2 means a request is open. The lock was not taken.
Exit 3 means another live pid holds the lock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


def release(
    *,
    root: Path,
    host: str = _HOST,
    port: int = _PORT,
    pid: int | None = None,
) -> int:
    """Clear the smoke hold and remove this shell's lock."""
    owner = owner_pid() if pid is None else pid
    path = lock_path(root)
    holder = _read_pid(path)
    if holder is not None and holder != owner and alive(holder):
        print(f"lock held by {holder}")
        return 3
    code, _body = _post(host, port, "/v1/resume", {"pid": owner})
    if code not in {0, 200}:
        if code == 409:
            print("a request is open")
            return 2
        print("resume failed")
        return 1
    if holder is None or holder == owner or not alive(holder):
        _drop_claim(root, owner if holder == owner else holder or owner)
    print("lock released")
    return 0


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
    sub.add_parser("release")
    args = parser.parse_args(argv)
    root = repo_root()
    if args.action == "acquire":
        return acquire(
            root=root,
            pipeline=args.pipeline,
            command=args.command,
            host=args.host,
            port=args.port,
        )
    return release(root=root, host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
