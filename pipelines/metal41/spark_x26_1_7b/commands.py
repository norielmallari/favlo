"""``favlo --spark-x26-1-7b`` commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from favlo.registry import load_registry, models_dir, register_quantized, set_alias

from spark_x26_1_7b.layout import FOLDER, REPO, STEM

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
    quant = sub.add_parser("quantize", help="write the fp4 record")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
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
        download(dest, files, lambda done, total, name: sys.stdout.write(f"{name} {done}/{total}\n"))
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
        register_quantized(
            output.name,
            source=src,
            alias=PIPELINE,
            directory=output,
            pipeline=PIPELINE,
            runtime="metal41",
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    sys.stdout.write(f"{output}\n")
    return 0


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
