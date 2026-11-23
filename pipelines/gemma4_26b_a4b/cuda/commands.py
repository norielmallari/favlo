"""Commands for ``favlo --gemma4_26b_a4b`` on CUDA."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from favlo.registry import models_dir, register_quantized
from gemma4_26b_a4b.cuda.convert import quantize, record_name
from gemma4_26b_a4b.cuda.layout import FOLDER, resident_bytes
from gemma4_26b_a4b.cuda.sampling import DEFAULT_VARIANT

PIPELINE = "gemma4_26b_a4b"
CARROT_PROMPT = (
    "Amanda and Celestia have 15 carrots. They collect more carrots for Favlo the Horse. "
    "After collecting, they have 21 carrots. How many carrots did they collect?"
)


def _folder(text: str) -> str:
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        message = "folder name must be a single path segment"
        raise argparse.ArgumentTypeError(message)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --gemma4_26b_a4b")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the resident int8 size")
    quant = sub.add_parser("quantize", help="pack the bfloat16 snapshot")
    quant.add_argument("--src", type=Path, default=None)
    quant.add_argument("--name", type=_folder, default=None)
    sub.add_parser("kernels", help="compile the CUDA step and run its checks")
    complete = sub.add_parser("complete", help="sample one carrot reply")
    complete.add_argument("--name", type=_folder, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "plan":
            print(f"resident {resident_bytes()}")
            return 0
        if args.command == "quantize":
            return _quantize(args.src, args.name)
        if args.command == "kernels":
            from gemma4_26b_a4b.cuda.checks import run_kernels

            run_kernels()
            return 0
        if args.command == "complete":
            return _complete(args.name)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 2


def _quantize(src: Path | None, name: str | None) -> int:
    source = src if src is not None else _snapshot()
    if source is None or not source.is_dir():
        sys.stderr.write("missing checkpoint\n")
        return 2
    folder = name if name is not None else record_name()
    dest = models_dir() / folder
    quantize(source, dest)
    _claim_alias(folder)
    register_quantized(
        folder,
        source=source,
        alias=PIPELINE,
        directory=dest,
        pipeline=PIPELINE,
        runtime="cuda",
        kv_cache="half",
    )
    print(f"wrote {folder}", flush=True)
    return 0


def _complete(name: str | None) -> int:
    from gemma4_26b_a4b.cuda.runtime import open_runtime

    folder = name if name is not None else record_name()
    path = models_dir() / folder
    if not (path / "manifest.json").is_file():
        sys.stderr.write(f"missing record: {folder}\n")
        return 2
    runtime = open_runtime(path, kv_cache="half", console=None, owner="gemma4_26b_a4b:complete")
    try:
        text = runtime.complete_text([{"role": "user", "content": CARROT_PROMPT}], thinking=False)
    finally:
        runtime.close()
    sys.stdout.write(text + "\n")
    return 0


def _snapshot() -> Path | None:
    root = Path(r"D:\AIModels\hub\models--google--gemma-4-26B-A4B-it\snapshots")
    if not root.is_dir():
        return models_dir() / "gemma-4-26B-A4B-it"
    snaps = [path for path in root.iterdir() if path.is_dir()]
    if not snaps:
        return None
    return snaps[-1]


def _claim_alias(folder: str) -> None:
    """Move the public alias onto this record when another folder still holds it."""
    from favlo.registry import load_registry, save_registry

    data = load_registry()
    changed = False
    for name, entry in data["models"].items():
        if name == folder or not isinstance(entry, dict):
            continue
        if entry.get("alias") == PIPELINE:
            entry.pop("alias", None)
            changed = True
    if changed:
        save_registry(data)


__all__ = ["CARROT_PROMPT", "DEFAULT_VARIANT", "FOLDER", "main"]
