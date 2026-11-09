"""CLI for setup, kernels, and quantize."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --z-image-base")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="download Tongyi-MAI/Z-Image from the repository")
    sub.add_parser("kernels", help="compile the shaders and check the guidance mix")
    quant = sub.add_parser("quantize", help="pack safetensors weights")
    quant.add_argument("--weights", "-w", type=Path, help="one file or a directory of shards")
    quant.add_argument("--source", type=Path, help="checkout; same as --weights when omitted")
    quant.add_argument("--tokenizer-source", type=Path, help="tree with tokenizer/")
    quant.add_argument("--out", type=Path, help="packed directory")
    quant.add_argument("--name", default=None, help="folder under models/; the registry id")
    quant.add_argument("--fp4", action="store_true", help="pack fp4 instead of int8")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        from z_image_base.source import download

        print(download(), flush=True)
        return 0
    if args.command == "kernels":
        return kernels()
    if args.command == "quantize":
        return _quantize(
            args.weights,
            args.source,
            args.tokenizer_source,
            args.out,
            args.name,
            fp4=args.fp4,
        )
    return 2


_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _quantize(
    weights: Path | None,
    source: Path | None,
    tokenizer_source: Path | None,
    out: Path | None,
    name: str | None,
    *,
    fp4: bool = False,
) -> int:
    from z_image.record import quantize

    from favlo.registry import register_quantized
    from z_image_base.source import source_dir

    origin = weights or source or source_dir()
    kind = "fp4" if fp4 else "int8"
    try:
        dest = _record_dir(origin, name, out, fp4=fp4)
        quantize(
            origin,
            dest,
            tokenizer_source=tokenizer_source,
            kind=kind,
            source_repo="Tongyi-MAI/Z-Image",
            format_id="z-image-base-fp4" if fp4 else "z-image-base-int8",
        )
        register_quantized(
            dest.name,
            source=origin,
            alias=None,
            directory=dest,
            pipeline="z-image-base",
            runtime="metal41",
            max_prompt_tokens=1536,
        )
        _store_sampler(dest.name)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest, flush=True)
    return 0


def _record_dir(origin: Path, name: str | None, out: Path | None, *, fp4: bool = False) -> Path:
    """Packed directory. A custom file is ``<stem>-int8`` or ``<stem>-fp4``."""
    from favlo.registry import models_dir

    if name is not None and out is not None:
        message = "--name and --out name the same directory"
        raise ValueError(message)
    if name is not None:
        _check_folder(name)
        return models_dir() / name
    if out is not None:
        return out
    stem = origin.stem if origin.suffix == ".safetensors" else origin.name
    suffix = "-fp4" if fp4 else "-int8"
    if not stem.endswith(suffix):
        stem = f"{stem}{suffix}"
    _check_folder(stem)
    return models_dir() / stem


def _store_sampler(folder: str) -> None:
    """Keep a row's sampler fields. A new row receives the pipeline defaults."""
    from favlo.registry import load_registry, save_registry

    from z_image_base.schedule import (
        CFG_NORMALIZATION,
        CFG_TRUNCATION,
        GUIDANCE,
        SHIFT,
        STEPS,
    )

    data = load_registry()
    entry = data["models"].get(folder)
    if not isinstance(entry, dict):
        return
    stored = {
        "steps": STEPS,
        "guidance": GUIDANCE,
        "shift": SHIFT,
        "cfg_truncation": CFG_TRUNCATION,
        "cfg_normalization": CFG_NORMALIZATION,
    }
    changed = False
    for key, value in stored.items():
        if key not in entry:
            entry[key] = value
            changed = True
    if changed:
        save_registry(data)


def _check_folder(name: str) -> None:
    if _FOLDER.fullmatch(name) is None:
        message = f"{name} is not a folder name; pass --name"
        raise ValueError(message)


def kernels() -> int:
    from z_image.metal import Gpu

    gpu = Gpu()
    try:
        if not _cfg(gpu):
            print("kernels cfg mismatch", flush=True)
            return 1
    finally:
        gpu.close()
    print("kernels ok", flush=True)
    return 0


def _cfg(gpu: Any) -> bool:
    """The guidance mix and the optional L2 cap, against the host formula."""
    rng = np.random.default_rng(3)
    count = 1000
    pos = rng.normal(size=count).astype(np.float32)
    neg = rng.normal(size=count).astype(np.float32)
    pos_b = gpu.upload(pos)
    if not _cfg_case(gpu, pos, neg, pos_b, 4.0, 1.0, count):
        return False
    return _cfg_case(gpu, pos, neg, pos_b, 4.0, 0.0, count)


def _cfg_case(
    gpu: Any,
    pos: np.ndarray,
    neg: np.ndarray,
    pos_b: Any,
    scale: float,
    limit: float,
    count: int,
) -> bool:
    neg_b = gpu.upload(np.array(neg, copy=True))
    gpu.cfg_apply(pos_b, neg_b, scale, limit, count)
    got = gpu.read_f32(neg_b, count)
    pred = pos + scale * (pos - neg)
    if limit > 0:
        pred_norm = float(np.linalg.norm(pred))
        cap = float(np.linalg.norm(pos)) * limit
        if pred_norm > cap and pred_norm > 0:
            pred = pred * (cap / pred_norm)
    return float(np.max(np.abs(got - pred))) < 1e-4
