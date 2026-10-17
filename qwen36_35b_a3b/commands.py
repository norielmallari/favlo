"""``madmetal --qwen36-35b-a3b`` commands. The listener stays ``madmetal serve``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from madmetal.registry import (
    add_model,
    choose_complete,
    format_listing,
    register_quantized,
    set_alias,
)
from qwen36_35b_a3b.chat import STOP, decode_text, encode_chat
from qwen36_35b_a3b.checks import format_report, run_checks
from qwen36_35b_a3b.convert import RECORD_NAME, quantize
from qwen36_35b_a3b.forward import Qwen36Runtime

PIPELINE = "qwen36-35b-a3b"
GROVE_PROMPT = (
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
)
_BF16 = Path("models/Qwen3.6-35B-A3B-bf16")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="madmetal --qwen36-35b-a3b")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize", help="write the Metal record from BF16 shards")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    sub.add_parser("bf16", help="run this package's shaders on stored BF16 values")
    complete = sub.add_parser("complete", help="complete from the packed record")
    complete.add_argument(
        "--kv-cache",
        choices=("e4m3", "e2m1", "f32"),
        default="e4m3",
        help="key/value width: fp8, fp4, or float32",
    )
    complete.add_argument(
        "--max-memory",
        default=None,
        help="maximum bytes for this runtime, such as 32GiB",
    )
    sub.add_parser("kernels", help="compile the shader library and run the dispatch checks")
    models = sub.add_parser("models", help="list or edit the flavor registry")
    models_sub = models.add_subparsers(dest="models_command")
    add = models_sub.add_parser("add", help="register a finished record directory")
    add.add_argument("folder")
    add.add_argument("--alias", default=None)
    add.add_argument("--source", default=None)
    alias = models_sub.add_parser("alias", help="change the alias of a folder")
    alias.add_argument("folder")
    alias.add_argument("alias")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "models":
        return _models(args)
    if args.command == "kernels":
        return _kernels()
    if args.command == "quantize":
        return _quantize(args)
    if args.command == "complete":
        return _complete(args)
    sys.stderr.write("qwen36-35b-a3b bf16 layer check is not this command\n")
    return 1


def _quantize(args: argparse.Namespace) -> int:
    src = args.src or _BF16
    output = args.output or Path("models") / RECORD_NAME
    try:
        quantize(src, output)
        register_quantized(
            output.name,
            source=src,
            alias="qwen36-35b-a3b",
            pipeline=PIPELINE,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{output}\n")
    return 0


def _complete(args: argparse.Namespace) -> int:
    try:
        path, _selection = choose_complete(model=None, record=None, pipeline=PIPELINE)
        runtime = Qwen36Runtime(
            path,
            kv_cache=args.kv_cache,
            max_memory=args.max_memory,
        )
        try:
            messages = [{"role": "user", "content": GROVE_PROMPT}]
            prompt = encode_chat(runtime.root, messages, None, False)
            tokens = runtime.generate(prompt, 128, 0.0, 1.0, 0, 0.0, STOP)
            text = decode_text(runtime.root, tokens)
        finally:
            runtime.close()
        body = {"prompt": GROVE_PROMPT, "text": text}
        target = path / "grove.json"
        target.write_text(json.dumps(body, indent=2) + "\n")
    except (OSError, RuntimeError, LookupError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(text + "\n")
    return 0


def _kernels() -> int:
    try:
        report = run_checks()
    except (OSError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(format_report(report) + "\n")
    return 0 if report["ok"] else 1


def _models(args: argparse.Namespace) -> int:
    command = getattr(args, "models_command", None)
    try:
        if command == "add":
            add_model(
                args.folder,
                alias=args.alias,
                source=args.source,
                pipeline=PIPELINE,
            )
        elif command == "alias":
            set_alias(args.folder, args.alias)
        listing = format_listing()
        if listing:
            sys.stdout.write(listing + "\n")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
