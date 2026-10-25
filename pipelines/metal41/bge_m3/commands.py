"""CLI for quantize and embed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bge_m3.quantize import RECORD


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --bge-m3")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kernels", help="dispatch the int8 matmul and one attention head")
    quant = sub.add_parser("quantize", help="pack the checkout to int8")
    quant.add_argument("--src", type=Path, required=True)
    quant.add_argument("--out", type=Path)
    embed = sub.add_parser("embed", help="print one dense vector from the packed record")
    embed.add_argument("--text", required=True)
    embed.add_argument("--record", type=Path)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "kernels":
        from bge_m3.gpu import check_kernels

        check_kernels()
        return 0
    if args.command == "quantize":
        return _quantize(args.src, args.out)
    if args.command == "embed":
        return _embed(args.text, args.record)
    return 2


def _quantize(source: Path, out: Path | None) -> int:
    from bge_m3.quantize import quantize
    from favlo.cli import repo_root

    dest = out or repo_root() / "models" / RECORD
    quantize(source, dest)
    print(dest, flush=True)
    return 0


def _embed(text: str, record: Path | None) -> int:
    from bge_m3.runtime import Runtime
    from favlo.cli import repo_root

    path = record or repo_root() / "models" / RECORD
    if not (path / "manifest.json").is_file():
        print(f"{path} is not a packed record", file=sys.stderr)
        return 1
    runtime = Runtime(path)
    try:
        result = runtime.embed(text)
    finally:
        runtime.close()
    print(json.dumps(result.vector), flush=True)
    return 0
