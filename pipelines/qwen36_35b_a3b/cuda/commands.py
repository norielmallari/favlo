"""``favlo --qwen36_35b_a3b`` on Windows. Download, then pack the int4 record."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.console import Console, publish_count
from favlo.registry import load_registry, models_dir, register_quantized, set_alias
from qwen36_35b_a3b.cuda.convert import checkout_dir, download, list_files, quantize
from qwen36_35b_a3b.cuda.layout import EXPERT_PERCENT, REPO, STEM, record_name

PIPELINE = STEM
PUBLIC_ID = "qwen36-35b-a3b"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"favlo --{PIPELINE}")
    sub = parser.add_subparsers(dest="command")
    fetch = sub.add_parser("download", help="download the official BF16 checkpoint")
    fetch.add_argument("--check", action="store_true")
    quant = sub.add_parser("quantize", help="pack the BF16 checkout on the CPU")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    quant.add_argument("--expert-int8", type=float, default=EXPERT_PERCENT)
    quant.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    setup = sub.add_parser("setup",
        help="download the official checkpoint and pack the int4 record")
    setup.add_argument("--src", type=Path, default=None)
    setup.add_argument("--keep-bf16", action="store_true")
    setup.add_argument("--expert-int8", type=float, default=EXPERT_PERCENT)
    setup.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    sub.add_parser("kernels", help="compile with the CUDA Toolkit and run the dispatch checks")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "download":
        return _download(args)
    if args.command == "quantize":
        return _quantize(args)
    if args.command == "setup":
        return _setup(args)
    if args.command == "kernels":
        return _kernels()
    parser.print_help()
    return 2


def _download(args: argparse.Namespace) -> int:
    dest = checkout_dir(models_dir())
    if args.check:
        index = (dest / "model.safetensors.index.json").is_file()
        if index and (dest / "tokenizer.json").is_file():
            sys.stdout.write("ok\n")
            return 0
        sys.stderr.write(f"missing checkout at {dest}\n")
        return 1
    files = list_files()
    download(dest, files, lambda _done, _whole, _name: None)
    sys.stdout.write(f"downloaded {REPO}\n")
    return 0


def _quantize(args: argparse.Namespace) -> int:
    src = Path(args.src) if args.src else checkout_dir(models_dir())
    folder = record_name(src.name, float(args.expert_int8))
    output = Path(args.output) if args.output else models_dir() / folder
    if not (src / "model.safetensors.index.json").is_file():
        sys.stderr.write(f"missing BF16 checkout at {src}\n")
        return 1
    ui = Console.from_env(args.console, pipeline=PIPELINE)
    bar = ui.begin("quantize", total=1, unit="files", detail=output.name)
    try:
        quantize(
            src,
            output,
            expert_int8=float(args.expert_int8),
            on_progress=lambda done, whole, name: _advance(bar, done, whole, name),
        )
        ui.end(bar, f"packed {output.name}")
        _register(src, output)
    finally:
        ui.close()
    return 0


def _setup(args: argparse.Namespace) -> int:
    src = Path(args.src) if args.src else checkout_dir(models_dir())
    if args.src is None and not (src / "model.safetensors.index.json").is_file():
        ui = Console.from_env(args.console, pipeline=PIPELINE)
        try:
            files = list_files()
            total = sum(size for _name, size in files)
            bar = ui.begin("download", total=max(total, 1), unit="bytes", detail=REPO)
            download(src, files, lambda done, whole, name: _advance(bar, done, whole, name))
            ui.end(bar, f"downloaded {REPO}")
        finally:
            ui.close()
    code = _quantize(args)
    if code == 0 and not args.keep_bf16 and src == checkout_dir(models_dir()):
        _remove(src)
    return code


def _kernels() -> int:
    try:
        from qwen36_35b_a3b.cuda.checks import run_kernels

        run_kernels()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


def _register(src: Path, output: Path) -> None:
    data = load_registry()
    for name, entry in data["models"].items():
        if name != output.name and entry.get("alias") == PUBLIC_ID:
            set_alias(name, None)
    register_quantized(
        output.name,
        source=src,
        alias=PUBLIC_ID,
        directory=output,
        pipeline=PIPELINE,
        runtime="cuda",
        kv_cache="int8",
    )


def _advance(bar: Any, done: int, whole: int, name: str) -> None:
    if bar is None:
        return
    bar.total = whole
    bar.detail = name
    publish_count(None, bar, done)


def _remove(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
