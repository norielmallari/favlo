"""``favlo --qwen35_4b`` on Windows. Download, then pack for this VRAM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from favlo.console import Console, publish_count
from favlo.qwen35_pack import Target, choose_target, quantize, record_folder, runtime_for
from favlo.registry import load_registry, models_dir, register_quantized, save_registry
from qwen35_4b.cuda.convert import checkout_dir, download, list_files
from qwen35_4b.cuda.layout import REPO, STEM

PIPELINE = STEM


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"favlo --{PIPELINE}")
    sub = parser.add_subparsers(dest="command")
    fetch = sub.add_parser("download", help="download the official BF16 checkpoint")
    fetch.add_argument("--check", action="store_true")
    quant = sub.add_parser("quantize", help="pack the BF16 checkout on the CPU")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    quant.add_argument("--plane", choices=("fp8", "int8", "int4"), default=None)
    quant.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    setup = sub.add_parser(
        "setup",
        help="download the official checkpoint and pack it for this VRAM",
    )
    setup.add_argument("--src", type=Path, default=None)
    setup.add_argument("--keep-bf16", action="store_true")
    setup.add_argument("--plane", choices=("fp8", "int8", "int4"), default=None)
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


def installed_record() -> Path | None:
    target = choose_target(None)
    path = models_dir() / record_folder(target.plane)
    manifest = path / "manifest.json"
    if not manifest.is_file() or not (path / "weights.bin").is_file():
        return None
    try:
        body = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if body.get("complete") is not True:
        return None
    if target.plane != "fp8" and body.get("weight") != target.plane:
        return None
    return path


def _download(args: argparse.Namespace) -> int:
    dest = checkout_dir(models_dir())
    if args.check:
        if _present(dest) and (dest / "tokenizer.json").is_file():
            sys.stdout.write("ok\n")
            return 0
        sys.stderr.write(f"missing checkout at {dest}\n")
        return 1
    files = list_files()
    download(dest, files, lambda _done, _whole, _name: None)
    sys.stdout.write(f"downloaded {REPO}\n")
    return 0


def _quantize(args: argparse.Namespace) -> int:
    target = choose_target(getattr(args, "plane", None))
    src = Path(args.src) if args.src else checkout_dir(models_dir())
    output = Path(args.output) if args.output else models_dir() / record_folder(target.plane)
    if not src.is_dir():
        sys.stderr.write(f"missing BF16 checkout at {src}\n")
        return 1
    ui = Console.from_env(args.console, pipeline=PIPELINE)
    bar = ui.begin("quantize", total=1, unit="files", detail=target.plane)
    try:
        quantize(
            src,
            output,
            target,
            on_progress=lambda done, whole, name: _advance(bar, done, whole, name),
        )
        ui.end(bar, f"packed {output.name}")
        _register(src, output, target)
    finally:
        ui.close()
    return 0


def _setup(args: argparse.Namespace) -> int:
    target = choose_target(args.plane)
    ui = Console.from_env(args.console, pipeline=PIPELINE)
    src = Path(args.src) if args.src else checkout_dir(models_dir())
    output = models_dir() / record_folder(target.plane)
    try:
        if args.src is None and installed_record() is not None:
            _register(src, output, target)
            if not args.keep_bf16:
                _remove_checkout(src, spare=output)
            ui.log(f"record complete: {output}")
            return 0
        if args.src is None:
            files = list_files()
            total = sum(size for _name, size in files)
            bar = ui.begin("download", total=max(total, 1), unit="bytes", detail=REPO)
            download(src, files, lambda done, whole, name: _advance(bar, done, whole, name))
            ui.end(bar, f"downloaded {REPO}")
        elif not _present(src):
            sys.stderr.write(f"missing BF16 checkout at {src}\n")
            return 1
    finally:
        ui.close()
    code = _quantize(
        argparse.Namespace(src=src, output=output, console=args.console, plane=args.plane)
    )
    if code != 0:
        return code
    if not args.keep_bf16 and src == checkout_dir(models_dir()):
        _remove_checkout(src, spare=output)
    return 0


def _kernels() -> int:
    try:
        from qwen35_4b.cuda.checks import run_kernels

        run_kernels()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


def _advance(progress: object, done: int, total: int, detail: str) -> None:
    progress.total = total
    progress.detail = detail
    publish_count(None, progress, done)


def _register(src: Path, output: Path, target: Target) -> None:
    """Register the record. The VRAM plane keeps the ``qwen35_4b`` alias."""
    host = choose_target(None)
    owns_alias = target.plane == host.plane
    if owns_alias:
        _release_alias(PIPELINE, keep=output.name)
    register_quantized(
        output.name,
        source=src,
        alias=PIPELINE if owns_alias else None,
        directory=output,
        pipeline=PIPELINE,
        runtime=runtime_for(target.plane),
        max_prompt_tokens=target.context,
        kv_cache="fp8_e4m3" if target.plane == "fp8" else "int8",
    )


def _release_alias(alias: str, keep: str) -> None:
    data = load_registry()
    changed = False
    for name, entry in data.get("models", {}).items():
        if name == keep or not isinstance(entry, dict):
            continue
        if entry.get("alias") == alias:
            entry.pop("alias", None)
            changed = True
    if changed:
        save_registry(data)


def _present(dest: Path) -> bool:
    single = (dest / "model.safetensors").is_file()
    index = (dest / "model.safetensors.index.json").is_file()
    return single or index


def _remove_checkout(path: Path, *, spare: Path | None = None) -> None:
    root = models_dir().resolve()
    try:
        target = path.expanduser().resolve()
        spared = None if spare is None else spare.expanduser().resolve()
    except OSError:
        return
    if target == root or root not in target.parents:
        return
    if spared is not None and (target == spared or target in spared.parents):
        return
    if path.is_symlink() or not path.is_dir():
        return
    shutil.rmtree(target)
