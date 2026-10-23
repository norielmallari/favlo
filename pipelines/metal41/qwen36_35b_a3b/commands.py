"""``favlo --qwen36-35b-a3b`` commands. The listener stays ``favlo serve``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from favlo.console import Console, Progress
from favlo.registry import (
    add_model,
    choose_complete,
    format_listing,
    models_dir,
    register_quantized,
    set_alias,
)
from qwen36_35b_a3b.chat import STOP, decode_text, encode_chat, instruct_forbid
from qwen36_35b_a3b.checks import format_report, run_checks
from qwen36_35b_a3b.convert import quantize, record_name
from qwen36_35b_a3b.fetch import (
    REPO,
    download,
    list_files,
    record_bytes,
    remaining_download,
    shortfall,
)
from qwen36_35b_a3b.formats import promote_count
from qwen36_35b_a3b.forward import Qwen36Runtime

PIPELINE = "qwen36-35b-a3b"
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)
_BF16 = Path("models/Qwen3.6-35B-A3B-bf16")
_BF16_REPO = "Qwen/Qwen3.6-35B-A3B"


def _expert_budget(text: str) -> float:
    """Accept a routed-expert fp8 percentage from 0 to 100."""
    try:
        percent = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expert fp8 percentage is not a number") from exc
    try:
        promote_count(percent, 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return percent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --qwen36-35b-a3b")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize", help="write the Metal record from BF16 shards")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    quant.add_argument(
        "--expert-fp8",
        type=_expert_budget,
        default=20,
        help="percentage of routed experts stored as fp8, from 0 to 100",
    )
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
    setup = sub.add_parser("setup", help="download the official checkpoint and pack fp8-e20")
    setup.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
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
    if args.command == "setup":
        return _setup(args)
    if args.command == "complete":
        return _complete(args)
    sys.stderr.write("qwen36-35b-a3b bf16 layer check is not this command\n")
    return 1


def _setup(args: argparse.Namespace) -> int:
    """Download Qwen/Qwen3.6-35B-A3B, pack fp8-e20, and remove the checkout."""
    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    if os.environ.get("FAVLO_SKIP_PLATE") != "1":
        ui.plate()
    src = models_dir() / _BF16.name
    output = models_dir() / record_name(_BF16, 20)
    download_bar = None
    quantize_bar = None
    try:
        if _manifest_complete(output):
            _register(output, src)
            _remove_checkout(src)
            ui.log(f"record complete: {output}")
            return 0
        files = list_files()
        need_download = remaining_download(src, files)
        have_record = _tree_bytes(output)
        need_record = max(0, record_bytes() - have_record)
        message = shortfall(need_download, need_record, src, output)
        if message is not None:
            ui.log(message)
            return 1
        total = sum(size for _name, size in files)
        download_bar = ui.begin("download", total=total, unit="bytes", detail=REPO)
        download(
            src,
            files,
            lambda done, whole, _name: _move(download_bar, done, whole, REPO),
        )
        ui.end(download_bar, f"downloaded {REPO}")
        download_bar = None
        quantize_bar = ui.begin(
            "quantize",
            total=record_bytes(),
            unit="bytes",
            detail=output.name,
        )

        def _on_bytes(done: int, detail: str) -> None:
            assert quantize_bar is not None
            quantize_bar.done = done
            if done > quantize_bar.total:
                quantize_bar.total = done
            quantize_bar.detail = detail

        quantize(src, output, on_bytes=_on_bytes, expert_fp8=20)
        quantize_bar.done = quantize_bar.total
        ui.end(quantize_bar, f"record complete: {output}")
        quantize_bar = None
        _register(output, src)
        _remove_checkout(src)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        ui.report(f"{exc}\n")
        ui.log("the next serve continues from the files already written")
        return 1
    finally:
        if download_bar is not None and not download_bar.closed:
            ui.end(download_bar, "")
        if quantize_bar is not None and not quantize_bar.closed:
            ui.end(quantize_bar, "")
        ui.close()


def _move(progress: Progress, done: int, total: int, detail: str) -> None:
    progress.done = done
    progress.total = total
    progress.detail = detail


def _manifest_complete(output: Path) -> bool:
    path = output / "manifest.json"
    if not path.is_file():
        return False
    body = json.loads(path.read_text())
    return bool(body.get("complete"))


def _tree_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _register(output: Path, source: Path) -> None:
    register_quantized(
        output.name,
        source=source,
        alias="qwen36-35b-a3b",
        pipeline=PIPELINE,
        runtime="metal41",
    )


def _remove_checkout(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _quantize(args: argparse.Namespace) -> int:
    src = args.src or _BF16
    percent = float(args.expert_fp8)
    if args.output is not None:
        output = args.output
    else:
        output = Path("models") / record_name(Path(src), percent)
    try:
        quantize(src, output, expert_fp8=percent)
        register_quantized(
            output.name,
            source=src,
            alias="qwen36-35b-a3b",
            pipeline=PIPELINE,
            runtime="metal41",
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
            messages = [{"role": "user", "content": CARROT_PROMPT}]
            prompt = encode_chat(runtime.root, messages, None, False)
            tokens = runtime.generate(
                prompt,
                max(0, runtime.max_pos - len(prompt)),
                0.0,
                1.0,
                0,
                0.0,
                STOP,
                forbid=instruct_forbid(runtime.root),
            )
            text = decode_text(runtime.root, tokens)
        finally:
            runtime.close()
        body = {"prompt": CARROT_PROMPT, "text": text}
        target = path / "carrots.json"
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
