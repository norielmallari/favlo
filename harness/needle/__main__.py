"""Find one passkey in a generated file.

The file is filler sentences plus one line, ``The special magic number is
<digits>.`` ``--tokens`` is the exact file length under the model tokenizer.
``--depth`` is 0 at the first line and 1 at the last. The default depth is
the middle. The agent has one tool, the coding ``read_file`` window.
``--mode page`` passes when the reply contains the number and one read
returned that line. ``--mode retrieve`` passes when the file was read to the
end and the last window starts after that line.

    uv run python -m harness.needle --model qwen36-35b-a3b --tokens 65536
    uv run python -m harness.needle --model qwen36-35b-a3b --tokens 65536 --mode retrieve --depth 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from harness.benchmarks.serve import Request, Server
from harness.file_session import (
    format_request,
    format_summary,
    parse_requests,
    pipeline_log,
    printed_requests,
)
from harness.needle.haystack import FILE_NAME, generate
from harness.needle.loop import run_needle

_ROOT = Path(__file__).resolve().parents[2]


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find a passkey in a generated haystack.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mtp", choices=("on", "off"), default="off")
    parser.add_argument("--mtp-depth", type=int, default=3)
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--mode", choices=("page", "retrieve"), default="page")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.depth < 0.0 or args.depth > 1.0:
        parser.error("--depth must be from 0 to 1")
    if args.mtp_depth < 1:
        parser.error("--mtp-depth must be a positive integer")
    stem = args.model.partition(":")[0]
    from favlo.cli import repo_root
    from favlo.registry import resolve

    root = repo_root()
    workspace = _ROOT / "artifacts" / "needle" / stem
    server: Server | None = None
    try:
        selection = resolve(args.model, root=root)
        haystack = generate(
            tokens=args.tokens,
            depth=args.depth,
            seed=args.seed,
            encode=_tokenizer(selection.path),
        )
        _prepare(workspace, haystack.text)
        print(
            f"haystack {haystack.tokens} tokens line {haystack.line} depth {args.depth}",
            flush=True,
        )
        server = Server(args.max_memory, pipeline=stem)
        server.start()
        print(
            f"kv {selection.kv_cache} mtp {args.mtp} depth {args.mtp_depth} "
            f"thinking {args.thinking} mode {args.mode}",
            flush=True,
        )
        problems = asyncio.run(
            run_needle(
                server,
                stem,
                Request(enable_thinking=args.thinking == "on"),
                workspace,
                haystack,
                mtp=args.mtp == "on",
                mtp_depth=args.mtp_depth,
                mode=args.mode,
            )
        )
        _print_rates(server.log, stem)
    finally:
        if server is not None:
            server.stop()
    if problems:
        for note in problems:
            print(note, flush=True)
        return 1
    print("passed")
    return 0


def _tokenizer(model_root: Path):
    from tokenizers import Tokenizer

    path = model_root / "tokenizer.json"
    if not path.is_file():
        message = f"missing tokenizer at {path.name}"
        raise FileNotFoundError(message)
    tokenizer = Tokenizer.from_file(str(path))

    def encode(text: str) -> int:
        return len(tokenizer.encode(text).ids)

    return encode


def _prepare(workspace: Path, text: str) -> None:
    if workspace.exists():
        for child in sorted(workspace.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / FILE_NAME).write_text(text, encoding="utf-8")


def _print_rates(log: object, model: str) -> None:
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
