"""Production readiness for one language pipeline.

The model id is the pipeline name. ``--thinking on`` is the default and sets
``reasoning.effort`` to ``medium``. ``--thinking off`` sets it to ``none``.
Sampling stays on the selected model id. ``--mtp``
sends the draft flag. Omitted, that flag is off. ``--mtp-depth`` is the draft
length, three when omitted. ``--vision`` is off unless set. When it is set,
the first chat is the horse still, and a later request must reuse the tower.
Each model call prints the reasoning and the reply as the tokens arrive, then
the tool calls when that call ends. The framework runs the coding tools.
After the ledger, the same listener runs the chat contract.
Exit 0 means ``ledger.py`` ``collect`` returns 6, 0, and -6, and every
contract check held. Which tools the model called is not part of that result.
Each request's server prefill and decode lines are the thinking series. The
harness reuses ``127.0.0.1:8129``. After the first model turn it unloads the
resident model so the next turn loads the prefix from disk. Exit 1 when that
load is missing or a later turn does not extend it.

    uv run python -m harness.coding --model qwen38-27b
    uv run python -m harness.coding --model qwen38-27b --mtp on
    uv run python -m harness.coding --model qwen38-27b --vision --thinking on
    uv run python -m harness.coding --model qwen38-27b --vision --thinking off
    uv run python -m harness.coding --model qwen38-27b --vision --thinking off --store on
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
from harness.file_session import (
    format_request,
    format_summary,
    parse_requests,
    pipeline_log,
    printed_requests,
)
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
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument(
        "--store",
        choices=("on", "off"),
        default="off",
        help="one store value for the ledger and the contract",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.mtp_depth < 1:
        parser.error("--mtp-depth must be a positive integer")
    stem = args.model.partition(":")[0]
    from favlo.cli import repo_root

    root = repo_root()
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
            return 1
        vision = HORSE.read_bytes()
        # Drop a prior disk hit so this run's first tower logs ``vision patches``.
        # The same bytes are written again before the restart load.
        from favlo.logs import favlo_home

        cache = favlo_home() / "vision-cache" / stem
        if cache.is_dir():
            for child in cache.glob("*.npz"):
                child.unlink(missing_ok=True)
    server = Server(args.max_memory, pipeline=stem)
    try:
        from favlo.registry import resolve

        selection = resolve(args.model, root=root)
        server.start()
        vision_flag = "on" if args.vision else "off"
        store = args.store == "on"
        if vision is not None:
            from favlo.smoke_lock import unload_model

            unload_model()
        print(
            f"kv {selection.kv_cache} mtp {args.mtp} depth {args.mtp_depth} "
            f"thinking {args.thinking} vision {vision_flag} store {args.store}",
            flush=True,
        )
        _, problems = run_session(
            server,
            stem,
            Request(enable_thinking=args.thinking == "on"),
            workspace,
            mtp=args.mtp == "on",
            mtp_depth=args.mtp_depth,
            vision=vision,
            store=store,
        )
        if vision is not None:
            problems.extend(vision_problems(pipeline_log(server.log.text(), stem)))
        _print_rates(server.log, stem)
        base_url = f"http://127.0.0.1:{server.port}/v1"
        problems.extend(asyncio.run(run_contract(base_url, stem, store=store)))
        result = grade(workspace)
    finally:
        server.stop()
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        return result.returncode
    if problems:
        for note in problems:
            print(note, flush=True)
        return 1
    print("passed")
    return 0


def _print_rates(log: object, model: str) -> None:
    """Print this model's requests once the decode line is in the followed log."""
    text = getattr(log, "text", None)
    if not callable(text):
        return
    deadline = time.monotonic() + 2
    requests = printed_requests(parse_requests(pipeline_log(text(), model)))
    while time.monotonic() < deadline:
        requests = printed_requests(parse_requests(pipeline_log(text(), model)))
        if requests and requests[-1].decode:
            break
        time.sleep(0.05)
    for index, request in enumerate(requests, start=1):
        print(format_request(index, request), flush=True)
    print(format_summary(requests, 0), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
