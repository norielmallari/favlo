"""Build one Python module through ``serve`` and grade it.

The model id is the pipeline name. The request sets ``enable_thinking`` true
and leaves every other sampling field unset. The hidden suite is the result.
Each request's server prefill and decode lines are the thinking series.

    uv run python -m harness.coding --model qwen38-27b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from harness.benchmarks.serve import Request, Server
from harness.coding.grade import grade
from harness.coding.loop import run_session
from harness.file_session import format_request, format_summary, parse_requests

_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _ROOT / "artifacts" / "pipeline-test.lock"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a Python file written through serve.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    args = parser.parse_args(argv)
    stem = args.model.partition(":")[0]
    workspace = _ROOT / "artifacts" / "coding" / stem
    if workspace.exists():
        for child in sorted(workspace.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    workspace.mkdir(parents=True, exist_ok=True)
    take_lock(stem)
    server = Server(args.max_memory)
    try:
        server.start()
        run_session(server.port, stem, Request(enable_thinking=True), workspace)
        _print_rates(server.log.text())
        result = grade(workspace)
    finally:
        server.stop()
        release_lock()
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        print("passed")
    return result.returncode


def _print_rates(log_text: str) -> None:
    requests = parse_requests(log_text)
    for index, request in enumerate(requests, start=1):
        print(format_request(index, request), flush=True)
    print(format_summary(requests, 0), flush=True)


def take_lock(pipeline: str) -> None:
    """Exclusive machine lock. A live holder is left alone."""
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pipeline": pipeline, "pid": os.getpid(), "command": "coding"}).encode()
    while True:
        if _LOCK.exists() and not _stale_lock():
            time.sleep(5)
            continue
        try:
            fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.write(fd, payload)
        os.close(fd)
        try:
            current = json.loads(_LOCK.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if current.get("pid") == os.getpid():
            return


def release_lock() -> None:
    if not _LOCK.exists():
        return
    try:
        current = json.loads(_LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if current.get("pid") == os.getpid():
        _LOCK.unlink()


def _stale_lock() -> bool:
    try:
        current = json.loads(_LOCK.read_text(encoding="utf-8"))
        pid = int(current["pid"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        _LOCK.unlink()
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
