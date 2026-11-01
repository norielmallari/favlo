"""Production readiness for one language pipeline.

The model id is the pipeline name. One Agent Framework agent sets
``enable_thinking`` true and leaves every sampling field unset. ``--mtp``
sends the draft flag. Omitted, that flag is off. ``--mtp-depth`` is the draft
length, three when omitted. ``--vision`` is off unless set. When it is set,
the first chat is the horse still, and a later request must reuse the tower.
Each model call prints the reasoning and the reply as the tokens arrive, then
the tool calls when that call ends. The framework runs every coding tool.
After the ledger, the same listener runs the chat contract.
Exit 0 means the hidden suite passed, each tool returned successfully, and
every contract check held.
Each request's server prefill and decode lines are the thinking series.

    uv run python -m harness.coding --model qwen38-27b
    uv run python -m harness.coding --model qwen38-27b --mtp on
    uv run python -m harness.coding --model qwen38-27b --vision
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from harness.benchmarks.serve import Request, Server
from harness.coding.compliance import run_contract, vision_problems
from harness.coding.grade import grade
from harness.coding.loop import run_session
from harness.coding.workspace import missing_tools
from harness.file_session import format_request, format_summary, parse_requests
from harness.image_attachment import HORSE

_ROOT = Path(__file__).resolve().parents[2]


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grade a Python file written through serve.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mtp", choices=("on", "off"), default="off")
    parser.add_argument("--mtp-depth", type=int, default=3)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="send the horse still as the first chat",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.mtp_depth < 1:
        parser.error("--mtp-depth must be a positive integer")
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
    vision: bytes | None = None
    if args.vision:
        if not HORSE.is_file():
            print(f"missing {HORSE.as_posix()}", flush=True)
            if created:
                release(root=root)
            return 1
        vision = HORSE.read_bytes()
    server = Server(args.max_memory)
    try:
        server.start()
        print(f"mtp {args.mtp} depth {args.mtp_depth}", flush=True)
        calls, problems = run_session(
            server.port,
            stem,
            Request(enable_thinking=True),
            workspace,
            mtp=args.mtp == "on",
            mtp_depth=args.mtp_depth,
            vision=vision,
        )
        if vision is not None:
            problems.extend(vision_problems(server.log.text()))
        base_url = f"http://127.0.0.1:{server.port}/v1"
        problems.extend(asyncio.run(run_contract(base_url, stem)))
        _print_rates(server.log)
        result = grade(workspace)
        absent = missing_tools(calls)
    finally:
        server.stop()
        if created:
            release(root=root)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode
    if absent:
        print("missing tools: " + ", ".join(absent))
        return 1
    if problems:
        return 1
    print("passed")
    return 0


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
