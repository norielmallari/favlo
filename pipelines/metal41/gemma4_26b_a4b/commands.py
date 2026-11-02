"""Commands for ``favlo --gemma4-26b-a4b``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from favlo.registry import models_dir, register_quantized
from gemma4_26b_a4b.quantize import SOURCE_DIR, estimate, quantize, record_name
from gemma4_26b_a4b.sampling import DEFAULT_VARIANT

PIPELINE = "gemma4-26b-a4b"
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)


def _folder(text: str) -> str:
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        message = "folder name must be a single path segment"
        raise argparse.ArgumentTypeError(message)
    return text


def _print_plan() -> None:
    totals = estimate()
    gib = 1024**3
    for key in ("resident", "scratch", "half", "experts"):
        print(f"{key} {totals[key]} {totals[key] / gib:.3f}GiB")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --gemma4-26b-a4b")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print packed sizes from the checkpoint shapes")
    quant = sub.add_parser("quantize", help="pack a local bfloat16 checkout")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--name", type=_folder, default=None)
    sub.add_parser("kernels", help="compile the shader library and check matmul and the prefix pack")
    complete = sub.add_parser("complete", help="sample one carrot reply from the packed record")
    complete.add_argument("--name", type=_folder, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "plan":
            _print_plan()
            return 0
        if args.command == "quantize":
            return _quantize(args.src, args.name)
        if args.command == "kernels":
            from gemma4_26b_a4b.session import check_matmul

            check_matmul()
            return 0
        if args.command == "complete":
            return _complete(args.name)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 2


def _quantize(src: Path | None, name: str | None) -> int:
    source = src if src is not None else models_dir() / SOURCE_DIR
    if not source.is_dir():
        label = f"models/{SOURCE_DIR}" if src is None else source.name
        sys.stderr.write(f"missing checkpoint: {label}\n")
        return 2
    folder = name if name is not None else record_name()
    dest = models_dir() / folder
    quantize(source, dest)
    register_quantized(
        folder,
        source=source,
        alias=PIPELINE,
        directory=dest,
        pipeline=PIPELINE,
        runtime="metal41",
        kv_cache="half",
    )
    print(f"wrote {folder}", flush=True)
    return 0


def _complete(name: str | None) -> int:
    from gemma4_26b_a4b.runtime import open_runtime

    folder = name if name is not None else record_name()
    path = models_dir() / folder
    if not (path / "manifest.json").is_file():
        sys.stderr.write(f"missing record: {folder}\n")
        return 2
    runtime = open_runtime(
        path,
        kv_cache="half",
        console=None,
        resident_percent=None,
        owner="gemma4-26b-a4b:complete",
    )
    try:
        text = runtime.complete_text([{"role": "user", "content": CARROT_PROMPT}], thinking=False)
    finally:
        runtime.close()
    sys.stdout.write(text + "\n")
    return 0


__all__ = ["CARROT_PROMPT", "DEFAULT_VARIANT", "main"]
