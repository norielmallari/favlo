"""CLI for setup, kernels, quantize, and generate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from z_image_turbo.formats import TILE_M, TILE_N, dequant_fp4, pack_fp4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --z-image-turbo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="download Tongyi-MAI/Z-Image-Turbo from the repository")
    sub.add_parser("kernels", help="dispatch the fp4 matmul and the attention shader")
    quant = sub.add_parser("quantize", help="pack the checkpoint to fp4")
    quant.add_argument("--source", type=Path)
    quant.add_argument("--out", type=Path)
    gen = sub.add_parser("generate", help="write one PNG")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--steps", type=int, default=8)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--record", type=Path)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        from z_image_turbo.source import download

        print(download(), flush=True)
        return 0
    if args.command == "kernels":
        return _kernels()
    if args.command == "quantize":
        return _quantize(args.source, args.out)
    if args.command == "generate":
        return _generate(args)
    return 2


def _quantize(source: Path | None, out: Path | None) -> int:
    from favlo.registry import register_quantized
    from z_image_turbo.record import quantize
    from z_image_turbo.source import source_dir

    origin = source or source_dir()
    dest = out or origin.parent / "Z-Image-Turbo-fp4"
    quantize(origin, dest)
    register_quantized(
        dest.name,
        source=origin,
        alias=None,
        pipeline="z-image-turbo",
        max_prompt_tokens=1536,
    )
    print(dest, flush=True)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from z_image_turbo.runtime import Runtime
    from z_image_turbo.source import source_dir

    record = args.record or source_dir().parent / "Z-Image-Turbo-fp4"
    runtime = Runtime(record)
    try:
        png, digest, timing = runtime.generate(
            args.prompt, args.width, args.height, args.steps, args.seed
        )
    finally:
        runtime.close()
    args.out.write_bytes(png)
    print(f"latent {digest}", flush=True)
    print(f"wrote {args.out} steps {len(timing['steps'])}", flush=True)
    return 0


def _kernels() -> int:
    from z_image_turbo.metal import Gpu

    gpu = Gpu()
    try:
        return _kernel_checks(gpu)
    finally:
        gpu.close()


def _kernel_checks(gpu: Any) -> int:
    from z_image_turbo.metal import linear

    rng = np.random.default_rng(0)
    weight = rng.normal(size=(32, 256)).astype(np.float32)
    packed, scales, stored = pack_fp4(weight)
    act = rng.normal(size=(TILE_M, stored)).astype(np.float32)
    dest = gpu.empty(TILE_M * TILE_N * 4)
    out = gpu.empty(TILE_M * TILE_N * 4)
    linear(
        gpu,
        gpu.upload(act),
        gpu.upload(packed),
        gpu.upload(scales),
        dest,
        out,
        TILE_M,
        32,
        stored,
    )
    got = gpu.read_f32(dest, TILE_M * 32).reshape(TILE_M, 32)
    ref = act.astype(np.float16).astype(np.float32) @ dequant_fp4(packed, scales, stored).T
    if not _close(got, ref):
        print("kernels matmul mismatch", flush=True)
        return 1
    short_w = rng.normal(size=(32, 144)).astype(np.float32)
    packed, scales, stored = pack_fp4(short_w)
    act = np.zeros((TILE_M, stored), dtype=np.float32)
    act[:, :144] = rng.normal(size=(TILE_M, 144)).astype(np.float32)
    dest = gpu.empty(TILE_M * TILE_N * 4)
    linear(
        gpu,
        gpu.upload(act),
        gpu.upload(packed),
        gpu.upload(scales),
        dest,
        out,
        TILE_M,
        32,
        stored,
    )
    got = gpu.read_f32(dest, TILE_M * 32).reshape(TILE_M, 32)
    ref = act.astype(np.float16).astype(np.float32) @ dequant_fp4(packed, scales, stored).T
    if not _close(got, ref):
        print("kernels conv-K mismatch", flush=True)
        return 1
    if not _attention(gpu):
        print("kernels attention mismatch", flush=True)
        return 1
    print("kernels ok", flush=True)
    return 0


def _attention(gpu: object) -> bool:
    seq = 4
    heads = 2
    dim = 128
    rows = TILE_M
    rng = np.random.default_rng(1)
    q = np.zeros((rows, heads * dim), dtype=np.float32)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    q[:seq] = rng.normal(size=(seq, heads * dim)).astype(np.float32)
    k[:seq] = rng.normal(size=(seq, heads * dim)).astype(np.float32)
    v[:seq] = rng.normal(size=(seq, heads * dim)).astype(np.float32)
    dest = gpu.empty(rows * heads * dim * 4)
    gpu.attend(
        gpu.upload(q), gpu.upload(k), gpu.upload(v), dest, rows, seq, heads, heads, dim, True
    )
    got = gpu.read_f32(dest, seq * heads * dim).reshape(seq, heads, dim)
    ref = _attend_numpy(
        q[:seq].reshape(seq, heads, dim).astype(np.float32),
        k[:seq].reshape(seq, heads, dim).astype(np.float32),
        v[:seq].reshape(seq, heads, dim).astype(np.float32),
    )
    return _close(got, ref)


def _attend_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    seq, heads, dim = q.shape
    scale = dim**-0.5
    out = np.empty_like(q)
    allowed = np.arange(seq)[None, :] <= np.arange(seq)[:, None]
    for head in range(heads):
        scores = np.where(allowed, q[:, head] @ k[:, head].T * scale, -1.0e9)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        out[:, head] = weights @ v[:, head]
    return out


def _close(got: np.ndarray, ref: np.ndarray) -> bool:
    err = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref))) + 1.0
    print(f"kernels max abs {err:.5f} ref {scale - 1:.5f}", flush=True)
    return err <= 0.05 * scale
