"""setup and quantize for the shared umT5 int8 pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from umt5_xxl.metal41.source import pack_dir, repo_id
from umt5_xxl.metal41.weights import FORMATS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --umt5_xxl")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="download the BF16 encoder and tokenizer")
    setup.add_argument(
        "--umt5",
        type=_umt5_repo,
        default=None,
        help="umT5 repository, org/name or a huggingface.co link",
    )
    quant = sub.add_parser("quantize", help="pack int8 and copy the tokenizer into that pack")
    quant.add_argument(
        "--format",
        choices=FORMATS,
        required=True,
        help="wan is the official zip .pth; hf is a Hugging Face .safetensors encoder",
    )
    quant.add_argument("--src", type=Path, default=None, help="BF16 encoder file")
    quant.add_argument("--out", type=Path, default=None, help="packed directory")
    quant.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="directory that holds tokenizer.json when it is not beside --src",
    )
    quant.add_argument(
        "--keep-source",
        action="store_true",
        help="leave the BF16 checkout after packing",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        return _setup(args.umt5)
    if args.command == "quantize":
        return _quantize(args)
    return 2


def _setup(repo: str | None) -> int:
    from umt5_xxl.metal41.source import download

    try:
        dest = download(repo=repo)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest, flush=True)
    return 0


def _quantize(args: argparse.Namespace) -> int:
    from umt5_xxl.metal41.quantize import quantize

    try:
        dest = args.out if args.out is not None else pack_dir()
        worst = quantize(
            src=args.src,
            dest=dest,
            encoder_format=args.format,
            tokenizer_src=args.tokenizer,
            drop_source=not args.keep_source,
        )
    except (OSError, ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest, flush=True)
    print(f"int8 umt5 worst {float(worst):.6f}", flush=True)
    return 0


def _umt5_repo(text: str) -> str:
    return repo_id(text)
