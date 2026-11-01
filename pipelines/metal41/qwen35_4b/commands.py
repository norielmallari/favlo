"""``favlo --qwen35-4b`` commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from favlo.console import Console
from favlo.registry import load_registry, models_dir, register_quantized, set_alias
from qwen35_4b.convert import checkout_dir, quantize, record_dir
from qwen35_4b.layout import FOLDER, REPO, STEM
from qwen35_4b.sampling import CONTEXT_LENGTH

PIPELINE = STEM
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"favlo --{PIPELINE}")
    sub = parser.add_subparsers(dest="command")
    fetch = sub.add_parser("download", help="download the official BF16 checkpoint")
    fetch.add_argument("--check", action="store_true")
    quant = sub.add_parser("quantize", help="write the fp8 record")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    quant.add_argument("--console", choices=("auto", "pretty", "plain", "json"), default="auto")
    setup = sub.add_parser("setup", help="download the official checkpoint and pack it")
    setup.add_argument("--src", type=Path, default=None)
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
    models_sub.add_parser("list", help="list registered Qwen3.5-4B records")
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


def _checkout_present(dest: Path) -> bool:
    return (dest / "model.safetensors").is_file() or (
        dest / "model.safetensors.index.json"
    ).is_file()


def _download(args: argparse.Namespace) -> int:
    from qwen35_4b.convert import download, list_files

    dest = checkout_dir(models_dir())
    if args.check:
        if _checkout_present(dest) and (dest / "tokenizer.json").is_file():
            sys.stdout.write("ok\n")
            return 0
        sys.stderr.write(f"missing BF16 checkout at {dest}\n")
        return 1
    if (dest / "fetch_complete.json").is_file() or _checkout_present(dest):
        sys.stdout.write(f"present {dest}\n")
        return 0
    try:
        files = list_files()
        download(dest, files, lambda done, total, name: sys.stdout.write(f"{name} {done}/{total}\n"))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{dest}\n")
    return 0


def _quantize(args: argparse.Namespace) -> int:
    src = Path(args.src) if args.src else checkout_dir(models_dir())
    dest = Path(args.output) if args.output else record_dir(models_dir())
    ui = Console.from_env(args.console, pipeline=PIPELINE)
    bar = ui.begin("quantize", total=1, unit="files", detail=src.name)
    try:
        quantize(src, dest)
        bar.done = 1
        ui.end(bar, f"record complete: {dest}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        ui.end(bar, "")
        sys.stderr.write(f"{exc}\n")
        return 1
    finally:
        ui.close()
    register_quantized(
        dest.name,
        source=src,
        alias=PIPELINE,
        directory=dest,
        pipeline=PIPELINE,
        runtime="metal41",
        max_prompt_tokens=CONTEXT_LENGTH,
        kv_cache="f16",
    )
    return 0


def _setup(args: argparse.Namespace) -> int:
    """Download the official checkpoint, pack it, and remove the checkout."""
    from qwen35_4b.convert import download, list_files

    ui = Console.from_env(args.console, pipeline=PIPELINE)
    src = Path(args.src) if args.src else checkout_dir(models_dir())
    output = record_dir(models_dir())
    download_bar = None
    try:
        if args.src is None and installed_record() is not None:
            _register(src, output)
            if not args.keep_bf16:
                _remove_checkout(src, spare=output)
            ui.log(f"record complete: {output}")
            return 0
        if not _checkout_present(src):
            if args.src is not None:
                sys.stderr.write(f"missing BF16 checkout at {src}\n")
                return 1
            files = list_files()
            total = sum(size for _name, size in files)
            download_bar = ui.begin(
                "download",
                total=max(total, 1),
                unit="bytes",
                detail=REPO,
            )
            download(
                src,
                files,
                lambda done, whole, name: _advance(download_bar, done, whole, name),
            )
            ui.end(download_bar, f"downloaded {REPO}")
            download_bar = None
        code = _quantize(argparse.Namespace(src=src, output=None, console=args.console))
        if code != 0:
            return code
        if not args.keep_bf16 and src == checkout_dir(models_dir()):
            _remove_checkout(src, spare=output)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if download_bar is not None:
            ui.end(download_bar, "")
        sys.stderr.write(f"{exc}\n")
        sys.stderr.write("the next serve continues from the files already written\n")
        return 1
    finally:
        ui.close()


def _advance(progress: Any, done: int, total: int, detail: str) -> None:
    progress.done = done
    progress.total = total
    progress.detail = detail


def _register(src: Path, output: Path) -> None:
    register_quantized(
        output.name,
        source=src,
        alias=PIPELINE,
        directory=output,
        pipeline=PIPELINE,
        runtime="metal41",
        max_prompt_tokens=CONTEXT_LENGTH,
        kv_cache="f16",
    )


def _remove_checkout(path: Path, *, spare: Path | None = None) -> None:
    """Remove a BF16 checkout that lives under ``models/``."""
    import shutil

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


def _kernels() -> int:
    try:
        from qwen35_4b.checks import run_kernels

        run_kernels()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write("ok\n")
    return 0


def _complete(args: argparse.Namespace) -> int:
    path = installed_record()
    if path is None:
        sys.stderr.write(f"missing packed record {FOLDER}\n")
        return 1
    try:
        from qwen35_4b.runtime import open_runtime

        runtime = open_runtime(path, max_memory=args.max_memory)
        try:
            from qwen35_4b.chat import encode_prompt

            prompt = encode_prompt(runtime.root, args.prompt)
            tokens = runtime.generate(prompt, args.max_tokens, {248044}, temperature=0.0)
            from qwen35_4b.chat import decode_text

            sys.stdout.write(decode_text(runtime.root, tokens) + "\n")
        finally:
            runtime.close()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


def _models(args: argparse.Namespace) -> int:
    registry = load_registry()
    if args.models_command == "list":
        models = registry.get("models", {})
        if isinstance(models, dict):
            rows = [
                {"folder": folder, **row}
                for folder, row in models.items()
                if isinstance(row, dict) and row.get("pipeline") == PIPELINE
            ]
        else:
            rows = []
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.models_command == "add":
        folder = args.folder
        path = models_dir() / folder
        if not path.is_dir():
            sys.stderr.write(f"missing record directory {path}\n")
            return 1
        source = Path(args.source) if args.source else path
        register_quantized(
            folder,
            source=source,
            alias=args.alias,
            directory=path,
            pipeline=PIPELINE,
            runtime="metal41",
            max_prompt_tokens=CONTEXT_LENGTH,
            kv_cache="f16",
        )
        return 0
    if args.models_command == "alias":
        set_alias(args.folder, args.alias)
        return 0
    return 2
