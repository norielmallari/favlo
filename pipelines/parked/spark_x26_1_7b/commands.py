"""``favlo --spark-x26-1-7b`` commands."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.registry import load_registry, models_dir, register_quantized, set_alias
from spark_x26_1_7b.layout import FOLDER, STEM

PIPELINE = STEM
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"favlo --{PIPELINE}")
    sub = parser.add_subparsers(dest="command")
    fetch = sub.add_parser("download", help="download the official BF16 checkout")
    fetch.add_argument("--check", action="store_true")
    quant = sub.add_parser("quantize", help="write the fp8 record")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    setup = sub.add_parser("setup", help="download the official checkpoint and pack fp8")
    setup.add_argument(
        "--keep-bf16",
        action="store_true",
        help="leave the BF16 checkout under models/ after a successful quantize",
    )
    setup.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    sub.add_parser("kernels", help="compile the shader library and run the dispatch checks")
    complete = sub.add_parser("complete", help="argmax completion from the packed record")
    complete.add_argument("--prompt", default=CARROT_PROMPT)
    complete.add_argument("--max-tokens", type=int, default=64)
    complete.add_argument("--max-memory", default=None)
    models = sub.add_parser("models", help="list or edit the flavor registry")
    models_sub = models.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list registered Spark records")
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
    if args.command == "download":
        return _download(args)
    if args.command == "quantize":
        return _quantize(args)
    if args.command == "setup":
        return _setup(args)
    if args.command == "kernels":
        return _kernels()
    if args.command == "complete":
        return _complete(args)
    if args.command == "models":
        return _models(args)
    parser.print_help()
    return 2


def installed_record() -> Path | None:
    path = models_dir() / FOLDER
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        body = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if body.get("complete") is not True or not (path / "weights.bin").is_file():
        return None
    return path


def _download(args: argparse.Namespace) -> int:
    from spark_x26_1_7b.convert import checkout_dir, download, list_files

    dest = checkout_dir(models_dir())
    try:
        files = list_files()
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if args.check:
        sys.stdout.write("ok\n")
        return 0
    if (dest / "fetch_complete.json").is_file():
        sys.stdout.write(f"{dest}\n")
        return 0
    try:
        download(
            dest,
            files,
            lambda done, total, name: sys.stdout.write(f"{name} {done}/{total}\n"),
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{dest}\n")
    return 0


def _quantize(args: argparse.Namespace) -> int:
    from spark_x26_1_7b.convert import checkout_dir, quantize, record_dir

    src = Path(args.src) if args.src is not None else checkout_dir(models_dir())
    dest = Path(args.output) if args.output is not None else record_dir(models_dir())
    try:
        output = quantize(src, dest)
        _register_record(src, output)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{output}\n")
    return 0


def _setup(args: argparse.Namespace) -> int:
    """Download the official checkpoint, pack it, and remove the checkout."""
    from favlo.console import Console
    from spark_x26_1_7b.convert import checkout_dir, download, list_files, quantize, record_dir

    ui = Console.from_env(getattr(args, "console", None), pipeline=PIPELINE)
    if os.environ.get("FAVLO_SKIP_PLATE") != "1":
        ui.plate()
    src = checkout_dir(models_dir())
    output = record_dir(models_dir())
    download_bar = None
    try:
        if installed_record() is not None:
            _register_record(src, output)
            if not args.keep_bf16:
                _remove_checkout(src, spare=output)
            ui.log(f"record complete: {output}")
            return 0
        files = list_files()
        total = sum(size for _name, size in files)
        download_bar = ui.begin(
            "download",
            total=max(total, 1),
            unit="bytes",
            detail="XHToken/Spark-X2.6-1.7B",
        )
        download(src, files, lambda done, whole, name: _advance(download_bar, done, whole, name))
        ui.end(download_bar, "downloaded XHToken/Spark-X2.6-1.7B")
        download_bar = None
        quantize(src, output)
        _register_record(src, output)
        if not args.keep_bf16:
            _remove_checkout(src, spare=output)
        ui.log(f"record complete: {output}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        ui.report(f"{exc}\n")
        ui.log("the next serve continues from the files already written")
        return 1
    finally:
        if download_bar is not None and not download_bar.closed:
            ui.end(download_bar, "")
        ui.close()


def _advance(progress: Any, done: int, total: int, detail: str) -> None:
    progress.done = done
    progress.total = total
    progress.detail = detail


def _register_record(src: Path, output: Path) -> None:
    _free_pipeline_alias()
    register_quantized(
        output.name,
        source=src,
        alias=PIPELINE,
        directory=output,
        pipeline=PIPELINE,
        runtime="metal41",
        kv_cache="f16",
    )


def _remove_checkout(path: Path, *, spare: Path | None = None) -> None:
    """Remove a BF16 checkout that lives under ``models/``."""
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
    if path.is_symlink() or not target.is_dir():
        return
    shutil.rmtree(target)


def _free_pipeline_alias() -> None:
    """Drop this pipeline's alias from an older record so the new folder can take it."""
    data = load_registry()
    for name, entry in data["models"].items():
        if not isinstance(entry, dict):
            continue
        if entry.get("alias") != PIPELINE or entry.get("pipeline") != PIPELINE:
            continue
        if name == FOLDER:
            continue
        set_alias(name, None)


def _kernels() -> int:
    from spark_x26_1_7b.checks import run

    try:
        run()
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write("ok\n")
    return 0


def _complete(args: argparse.Namespace) -> int:
    from spark_x26_1_7b.chat import STOP, decode_text, encode_chat
    from spark_x26_1_7b.forward import SparkRuntime

    root = installed_record()
    if root is None:
        sys.stderr.write("quantized record is not installed\n")
        return 1
    runtime = SparkRuntime(root, max_memory=args.max_memory)
    try:
        prompt = encode_chat(
            runtime.root,
            [{"role": "user", "content": args.prompt}],
            None,
            False,
            None,
        )
        tokens = runtime.generate(
            prompt,
            args.max_tokens,
            STOP,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    finally:
        runtime.close()
    sys.stdout.write(decode_text(root, tokens) + "\n")
    return 0


def _models(args: argparse.Namespace) -> int:
    if args.models_command == "add":
        folder = str(args.folder)
        source = Path(args.source) if args.source else models_dir() / folder
        try:
            register_quantized(
                folder,
                source=source,
                alias=args.alias,
                directory=models_dir() / folder,
                pipeline=PIPELINE,
                runtime="metal41",
                kv_cache="f16",
            )
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        return 0
    if args.models_command == "alias":
        try:
            set_alias(args.folder, args.alias)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        return 0
    data = load_registry()
    for folder, entry in data.get("models", {}).items():
        if entry.get("pipeline") == PIPELINE:
            sys.stdout.write(f"{folder}\n")
    return 0
