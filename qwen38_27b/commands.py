"""``madmetal --qwen38-27b`` commands. The listener stays ``madmetal serve``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from madmetal.registry import add_model, format_listing, set_alias

PIPELINE = "qwen38-27b"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="madmetal --qwen38-27b")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize", help="write the Metal record from BF16 shards")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--output", type=Path, default=None)
    sub.add_parser("bf16", help="run this package's shaders on stored BF16 values")
    sub.add_parser("complete", help="complete from the packed record")
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
    sys.stderr.write("qwen38-27b forward is not available for this record\n")
    return 1


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
