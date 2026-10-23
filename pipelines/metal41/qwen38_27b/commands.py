"""``favlo --qwen38-27b`` commands. The listener stays ``favlo serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.hf_cache import snapshot_dir
from favlo.registry import (
    add_model,
    choose_complete,
    format_listing,
    models_dir,
    register_quantized,
    set_alias,
)
from qwen38_27b.chat import STOP, decode_text, encode_chat
from qwen38_27b.checks import format_report, run_checks
from qwen38_27b.convert import quantize
from qwen38_27b.fetch import (
    CHECKOUT,
    REPO,
    download,
    list_files,
    remaining_download,
    shortfall,
)
from qwen38_27b.forward import Qwen38Runtime

PIPELINE = "qwen38-27b"
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)
_BF16_REPO = "Qwen/Qwen3.8-27B"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --qwen38-27b")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize", help="write the fp4 record, or a mixed feed-forward record")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    quant.add_argument(
        "--ffn-fp8",
        nargs="?",
        const="",
        default=None,
        help="percentage of language feed-forward layers to store as fp8; empty promotes none",
    )
    fetch = sub.add_parser("download", help="download the official BF16 checkout")
    fetch.add_argument("--check", action="store_true", help="report disk space and do not write")
    fetch.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    sub.add_parser("bf16", help="run this package's shaders on stored BF16 values")
    complete = sub.add_parser("complete", help="argmax completion from the packed record")
    complete.add_argument("--kv-cache", choices=("e4m3",), default="e4m3")
    complete.add_argument("--max-memory", default=None, help="maximum bytes, such as 32GiB")
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
    if args.command == "download":
        return _download(args)
    if args.command == "complete":
        return _complete(args)
    sys.stderr.write("qwen38-27b bf16 layer check is not this command\n")
    return 1


def _download(args: argparse.Namespace) -> int:
    """Fetch Qwen/Qwen3.8-27B into ``models/Qwen3.8-27B-bf16``. Quantize is separate."""
    dest = models_dir() / CHECKOUT
    if not args.check and (dest / "fetch_complete.json").is_file():
        sys.stdout.write(f"{dest}\n")
        return 0
    try:
        files = list_files()
        message = shortfall(remaining_download(dest, files), dest)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if message is not None:
        sys.stderr.write(message + "\n")
        return 1
    if args.check:
        sys.stdout.write("ok\n")
        return 0
    if (dest / "fetch_complete.json").is_file():
        sys.stdout.write(f"{dest}\n")
        return 0
    from favlo.console import Console

    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    if os.environ.get("FAVLO_SKIP_PLATE") != "1":
        ui.plate()
    bar = None
    try:
        total = sum(size for _name, size in files)
        bar = ui.begin("download", total=total, unit="bytes", detail=REPO)
        download(dest, files, lambda done, whole, _name: _move(bar, done, whole))
        ui.end(bar, f"downloaded {REPO}")
        bar = None
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        ui.report(f"{exc}\n")
        ui.log("the next download continues from the files already written")
        return 1
    finally:
        if bar is not None and not bar.closed:
            ui.end(bar, "")
        ui.close()
    sys.stdout.write(f"{dest}\n")
    return 0


def _move(progress: Any, done: int, total: int) -> None:
    progress.done = done
    progress.total = total


def installed_record() -> Path | None:
    """The one registered record for this pipeline. None when the registry has no row."""
    try:
        path, _selection = choose_complete(model=None, record=None, pipeline=PIPELINE)
    except LookupError:
        return None
    if not (path / "manifest.json").is_file():
        return None
    return path


def _percent(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        message = f"ffn fp8 percentage {text} is not a number"
        raise ValueError(message) from exc


def _quantize(args: argparse.Namespace) -> int:
    src = args.src or snapshot_dir(_BF16_REPO)
    try:
        output = quantize(src, args.output, ffn_fp8=_percent(args.ffn_fp8))
        register_quantized(
            output.name,
            source=src,
            alias="qwen38-27b",
            pipeline=PIPELINE,
            runtime="metal41",
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{output}\n")
    return 0


def _complete(args: argparse.Namespace) -> int:
    try:
        path, _selection = choose_complete(model=None, record=None, pipeline=PIPELINE)
        runtime = Qwen38Runtime(path, kv_cache=args.kv_cache, max_memory=args.max_memory)
        try:
            messages = [{"role": "user", "content": CARROT_PROMPT}]
            prompt = encode_chat(runtime.root, messages, None, False)
            tokens = runtime.generate(prompt, None, STOP)
            text = decode_text(runtime.root, tokens)
        finally:
            runtime.close()
        body = {"prompt": CARROT_PROMPT, "text": text}
        target = path / "carrots.json"
        target.write_text(json.dumps(body, indent=2) + "\n")
    except (OSError, RuntimeError, LookupError, ValueError, KeyError) as exc:
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
                runtime="metal41",
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
