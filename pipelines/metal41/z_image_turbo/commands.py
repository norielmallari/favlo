"""CLI for setup, kernels, quantize, and generate."""

from __future__ import annotations

import argparse
import re
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
    quant = sub.add_parser("quantize", help="pack safetensors weights to fp4")
    quant.add_argument(
        "--weights",
        "-w",
        type=Path,
        help="one .safetensors file or a directory of shards",
    )
    quant.add_argument(
        "--source",
        type=Path,
        help="Diffusers checkout; same as --weights when omitted",
    )
    quant.add_argument(
        "--tokenizer-source",
        type=Path,
        help="tree with tokenizer/ when weights are not a full checkout",
    )
    quant.add_argument("--out", type=Path, help="packed directory")
    quant.add_argument(
        "--name",
        default=None,
        help="folder under models/; this name is the registry id",
    )
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
        return kernels()
    if args.command == "quantize":
        return _quantize(args.weights, args.source, args.tokenizer_source, args.out, args.name)
    if args.command == "generate":
        return _generate(args)
    return 2


_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _quantize(
    weights: Path | None,
    source: Path | None,
    tokenizer_source: Path | None,
    out: Path | None,
    name: str | None,
) -> int:
    from favlo.registry import register_quantized
    from z_image_turbo.record import quantize
    from z_image_turbo.source import source_dir

    origin = weights or source or source_dir()
    try:
        dest = _record_dir(origin, name, out)
        quantize(origin, dest, tokenizer_source=tokenizer_source)
        register_quantized(
            dest.name,
            source=origin,
            alias=None,
            directory=dest,
            pipeline="z-image-turbo",
            runtime="metal41",
            max_prompt_tokens=1536,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest, flush=True)
    return 0


def _record_dir(origin: Path, name: str | None, out: Path | None) -> Path:
    """Packed directory. A custom file is ``<stem>-fp4`` under ``models/``."""
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
    if not stem.endswith("-fp4"):
        stem = f"{stem}-fp4"
    _check_folder(stem)
    return models_dir() / stem


def _check_folder(name: str) -> None:
    if _FOLDER.fullmatch(name) is None:
        message = f"{name} is not a folder name; pass --name"
        raise ValueError(message)


def _generate(args: argparse.Namespace) -> int:
    from z_image_turbo.png import encode_png
    from z_image_turbo.runtime import Runtime
    from z_image_turbo.source import source_dir

    record = args.record or source_dir().parent / "Z-Image-Turbo-fp4"
    runtime = Runtime(record)
    try:
        pixels, digest, timing = runtime.generate(
            args.prompt, args.width, args.height, args.steps, args.seed
        )
    finally:
        runtime.close()
    args.out.write_bytes(encode_png(pixels))
    print(f"latent {digest}", flush=True)
    print(f"wrote {args.out} steps {len(timing['steps'])}", flush=True)
    return 0


def kernels() -> int:
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
    if not _layout(gpu):
        print("kernels layout mismatch", flush=True)
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


def _layout(gpu: object) -> bool:
    from z_image_turbo.runtime import _patchify, _unpatchify

    latent = np.zeros((16, 4, 2), dtype=np.float32)
    latent[3, 0, 1] = 7.0
    tokens = gpu.empty(2 * 256 * 4)
    gpu.patchify(gpu.upload(latent), tokens, 16, 2, 1, 256)
    got = gpu.read_f32(tokens, 2 * 256).reshape(2, 256)
    ref = _patchify(latent)
    if not np.array_equal(got[:, :64], ref) or not np.all(got[:, 64:] == 0):
        return False
    packed = gpu.upload(np.ascontiguousarray(ref))
    restored = gpu.empty(16 * 4 * 2 * 4)
    gpu.unpatchify(packed, restored, 16, 4, 2, 64)
    back = gpu.read_f32(restored, 16 * 4 * 2).reshape(16, 4, 2)
    if not np.array_equal(back, _unpatchify(ref, 4, 2)):
        return False
    base = gpu.upload(np.ones(4, dtype=np.float32))
    gpu.euler(base, gpu.upload(np.full(4, 2.0, dtype=np.float32)), 0.5, 4)
    if not np.allclose(gpu.read_f32(base, 4), 2.0):
        return False
    wave = gpu.empty(256 * 4)
    gpu.sinusoid(wave, 250.0, 256)
    half = np.arange(128, dtype=np.float32)
    freq = np.exp(-np.log(np.float32(10000.0)) * half / np.float32(128.0))
    angles = np.float32(250.0) * freq
    expect = np.concatenate([np.cos(angles), np.sin(angles)])
    if float(np.max(np.abs(gpu.read_f32(wave, 256) - expect))) > 1e-4:
        return False
    scaled = gpu.upload(np.array([0.3611, 0.0], dtype=np.float32))
    gpu.affine(scaled, 0.3611, 0.1159, 2)
    affine = gpu.read_f32(scaled, 2)
    if abs(float(affine[0]) - 1.1159) > 1e-4 or abs(float(affine[1]) - 0.1159) > 1e-4:
        return False
    pos = gpu.empty(4 * 3 * 4)
    gpu.text_positions(pos, 4, 2, 1)
    temporal = gpu.read_i32(pos, 12).reshape(4, 3)[:, 0]
    if not np.array_equal(temporal, np.array([1, 2, 0, 0], dtype=np.int32)):
        return False
    grid = np.zeros((3, 2, 2), dtype=np.float32)
    grid[0, 0, 0] = 1.0
    grid[1, 0, 1] = -1.0
    raw = gpu.empty(12)
    gpu.clip_u8(gpu.upload(grid), raw, 2, 2)
    picture = gpu.read_u8(raw, 12).reshape(2, 2, 3)
    if int(picture[0, 0, 0]) != 255 or int(picture[0, 1, 1]) != 0:
        return False
    src = gpu.upload(np.arange(8, dtype=np.float32).reshape(2, 4))
    dst = gpu.empty(8 * 4)
    gpu.copy_rows(src, dst, 4, 4, 4, 0, 0, 2)
    copied = gpu.read_f32(dst, 8).reshape(2, 4)
    if not np.array_equal(copied, np.arange(8, dtype=np.float32).reshape(2, 4)):
        return False
    token = gpu.upload(np.array([1, 2, 3, 4], dtype=np.float16))
    rows = gpu.empty(8 * 4)
    gpu.fill_half_rows(token, rows, 4, 1, 2)
    filled = gpu.read_f32(rows, 8).reshape(2, 4)
    return bool(np.allclose(filled[1], [1, 2, 3, 4]) and np.all(filled[0] == 0))


def _close(got: np.ndarray, ref: np.ndarray) -> bool:
    err = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref))) + 1.0
    print(f"kernels max abs {err:.5f} ref {scale - 1:.5f}", flush=True)
    return err <= 0.05 * scale
