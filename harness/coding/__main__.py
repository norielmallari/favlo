"""Build one Python module through ``serve`` and grade it.

The model id is the pipeline name. The request sets ``enable_thinking`` true
and leaves every other sampling field unset. The hidden suite is the result.
Each request's server prefill and decode lines are the thinking series.

    uv run python -m harness.coding --model qwen38-27b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from harness.benchmarks.serve import Request, Server
from harness.coding.grade import grade
from harness.coding.loop import run_session
from harness.file_session import format_request, format_summary, parse_requests

_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a Python file written through serve.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    args = parser.parse_args(argv)
    stem = args.model.partition(":")[0]
    from favlo.cli import repo_root
    from favlo.smoke_lock import hold, release

    root = repo_root()
    created = hold(root=root, pipeline=stem, command="coding")
    workspace = _ROOT / "artifacts" / "coding" / stem
    if workspace.exists():
        for child in sorted(workspace.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    workspace.mkdir(parents=True, exist_ok=True)
    server = Server(args.max_memory)
    try:
        server.start()
        run_session(server.port, stem, Request(enable_thinking=True), workspace)
        _print_rates(server.log)
        result = grade(workspace)
    finally:
        server.stop()
        if created:
            release(root=root)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        print("passed")
    return result.returncode


def _print_rates(log: object) -> None:
    """Print each request once its decode line is in the followed log."""
    text = getattr(log, "text", None)
    if not callable(text):
        return
    deadline = time.monotonic() + 2
    requests = parse_requests(text())
    while time.monotonic() < deadline:
        requests = parse_requests(text())
        if requests and requests[-1].decode:
            break
        time.sleep(0.05)
    for index, request in enumerate(requests, start=1):
        print(format_request(index, request), flush=True)
    print(format_summary(requests, 0), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
