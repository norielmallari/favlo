"""End-to-end rate of a cold synthetic prompt.

One Microsoft Agent Framework agent sends one user message and stops.
The message encodes to an exact prompt length under this model's chat
template. A short warmup runs first so the measured prompts are not the
load. Each measured prompt starts with its own label, so the prefill is a
cache miss. The printed end-to-end rate is the server's prefill time plus
its decode time.

    uv run python -m harness.synthetic --model qwen36-35b-a3b
    uv run python -m harness.synthetic --model qwen36-35b-a3b --tokens 512 --tokens 10000
"""

from __future__ import annotations

import argparse
import time

from harness.benchmarks.serve import Server
from harness.file_session import RequestRate, format_request, parse_requests, printed_requests
from harness.synthetic.loop import run_complete
from harness.synthetic.prompt import chat_encode, user_text
from harness.synthetic.rate import end_to_end, measured_problems

_DEFAULT_TOKENS = (512, 10000)
_WARMUP_TOKENS = 64
_WARMUP_DECODE = 16


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure a cold prompt end to end.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        action="append",
        default=None,
        help="prompt length; repeat for more than one. Default is 512 and 10000",
    )
    parser.add_argument("--decode", type=int, default=128, help="maximum sampled tokens")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mtp", choices=("on", "off"), default="off")
    parser.add_argument("--mtp-depth", type=int, default=3)
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--max-memory", default=None, help="serve memory cap, such as 16GiB")
    parser.add_argument(
        "--warmup",
        type=int,
        default=_WARMUP_TOKENS,
        help="cold prompt sent first so the measured lengths are not the load. 0 skips it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    lengths = tuple(args.tokens) if args.tokens else _DEFAULT_TOKENS
    if any(length < 1 for length in lengths):
        parser.error("--tokens must be at least 1")
    if args.decode < 16:
        parser.error("--decode must be at least 16")
    if args.mtp_depth < 1:
        parser.error("--mtp-depth must be a positive integer")
    if args.warmup < 0:
        parser.error("--warmup must be 0 or more")
    if len(set(lengths)) != len(lengths):
        parser.error("--tokens values must differ")
    if args.warmup in lengths:
        parser.error("--warmup must differ from every --tokens value")
    stem = args.model.partition(":")[0]
    from favlo.cli import repo_root
    from favlo.registry import resolve

    root = repo_root()
    selection = resolve(args.model, root=root)
    thinking = args.thinking == "on"
    encode = chat_encode(selection, thinking=thinking)
    prompts = [
        (
            length,
            user_text(
                tokens=length,
                label=f"Passage {length}.",
                encode=encode,
                seed=args.seed,
            ),
        )
        for length in lengths
    ]
    warmup = ""
    if args.warmup:
        warmup = user_text(
            tokens=args.warmup,
            label="Warmup passage.",
            encode=encode,
            seed=args.seed,
        )
    server = Server(args.max_memory, pipeline=stem)
    try:
        server.start()
        print(
            f"kv {selection.kv_cache} mtp {args.mtp} depth {args.mtp_depth} "
            f"thinking {args.thinking} decode {args.decode}",
            flush=True,
        )
        mark = _log_size()
        if warmup:
            run_complete(
                server,
                args.model,
                warmup,
                thinking=thinking,
                mtp=args.mtp == "on",
                mtp_depth=args.mtp_depth,
                decode=_WARMUP_DECODE,
            )
            warmed, mark = _finished_since(mark, args.warmup)
            if warmed:
                print(f"warmup {format_request(1, warmed)}", flush=True)
        problems: list[str] = []
        for index, (length, text) in enumerate(prompts, start=1):
            run_complete(
                server,
                args.model,
                text,
                thinking=thinking,
                mtp=args.mtp == "on",
                mtp_depth=args.mtp_depth,
                decode=args.decode,
            )
            request, mark = _finished_since(mark, length)
            if request is None:
                problems.append(f"prompt {length} produced no rate")
                continue
            print(format_request(index, request), flush=True)
            print(end_to_end(request), flush=True)
            problems.extend(measured_problems(request, length))
    finally:
        server.stop()
    if problems:
        for note in problems:
            print(note, flush=True)
        return 1
    print("passed")
    return 0


def _log_size() -> int:
    path = _log_file()
    if not path.is_file():
        return 0
    return path.stat().st_size


def _finished_since(mark: int, tokens: int) -> tuple[RequestRate | None, int]:
    """The finished request of ``tokens`` written after ``mark``.

    A rolled log is read from the start. A request of another length is not
    this prompt.
    """
    deadline = time.monotonic() + 2
    found, size = _read_finished(mark, tokens)
    while time.monotonic() < deadline and found is None:
        time.sleep(0.05)
        found, size = _read_finished(mark, tokens)
    return found, size


def _read_finished(mark: int, tokens: int) -> tuple[RequestRate | None, int]:
    path = _log_file()
    if not path.is_file():
        return None, 0
    size = path.stat().st_size
    offset = mark if size >= mark else 0
    text = path.read_bytes()[offset:].decode("utf-8", errors="replace")
    parsed = printed_requests(parse_requests(text))
    matched = [request for request in parsed if request.decode and request.prompt_tokens == tokens]
    if not matched:
        return None, size
    return matched[-1], size


def _log_file():
    from favlo.logs import favlo_home, log_path

    return log_path(favlo_home())


if __name__ == "__main__":
    raise SystemExit(main())
