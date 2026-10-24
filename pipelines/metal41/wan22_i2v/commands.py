"""CLI for quantize, kernels, and generate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from wan22_i2v.formats import TILE_M, dequant_fp4, dequant_fp8, pack_linear
from wan22_i2v.png import decode_png
from wan22_i2v.schedule import OFFICIAL, P320_HEIGHT, P320_WIDTH, SHORT_FRAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --wan22-i2v")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="download the VAE, umT5, and the two experts")
    sub.add_parser("kernels", help="dispatch the fp4 and fp8 matmul")
    quant = sub.add_parser("quantize", help="pack the high-noise and low-noise experts")
    quant.add_argument("--high", type=Path, required=True)
    quant.add_argument("--low", type=Path, required=True)
    quant.add_argument("--config", type=Path)
    quant.add_argument("--out", type=Path)
    gen = sub.add_parser("generate", help="write one AVI")
    gen.add_argument("--record", type=Path, required=True)
    gen.add_argument("--image", type=Path, required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--frames", type=int, default=SHORT_FRAMES)
    gen.add_argument("--width", type=int, default=P320_WIDTH)
    gen.add_argument("--height", type=int, default=P320_HEIGHT)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        from wan22_i2v.source import dit_dir, download, expert_paths, umt5_dir, vae_dir

        download()
        high, low = expert_paths()
        print(vae_dir(), flush=True)
        print(umt5_dir(), flush=True)
        print(dit_dir(), flush=True)
        print(high, flush=True)
        print(low, flush=True)
        return 0
    if args.command == "kernels":
        return kernels()
    if args.command == "quantize":
        return _quantize(args)
    if args.command == "generate":
        return _generate(args)
    return 2


def kernels() -> int:
    _matmul("fp4")
    _matmul("fp8")
    _attend()
    _spatial()
    _patch()
    print("fp4 ok", flush=True)
    print("fp8 ok", flush=True)
    print("attend ok", flush=True)
    print("spatial ok", flush=True)
    print("patch ok", flush=True)
    return 0


def _quantize(args: argparse.Namespace) -> int:
    from favlo.cli import repo_root
    from favlo.registry import register_quantized
    from wan22_i2v.record import quantize

    if args.config is None:
        config = OFFICIAL
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    dest = args.out or repo_root() / "models" / "Wan2.2-I2V-A14B-fp4"
    manifest = quantize(args.high, args.low, dest, config)
    register_quantized(
        dest.name,
        source=args.high,
        alias=None,
        pipeline="wan22-i2v",
        runtime="metal41",
        max_prompt_tokens=int(config["text_len"]),
    )
    print(dest, flush=True)
    print(f"high fp8 blocks {manifest['high_promoted']}", flush=True)
    print(f"low fp8 blocks {manifest['low_promoted']}", flush=True)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from wan22_i2v.runtime import Runtime

    image = decode_png(args.image.read_bytes())
    runtime = Runtime(args.record)
    try:
        avi, timing = runtime.generate(
            args.prompt, image, args.frames, args.width, args.height, args.seed
        )
    finally:
        runtime.close()
    args.out.write_bytes(avi)
    print(f"wrote {args.out} frames {args.frames} steps {len(timing['steps'])}", flush=True)
    return 0


def _matmul(kind: str) -> None:
    from wan22_i2v.metal import Gpu

    rows = TILE_M
    cols = 32
    inner = 256
    weight = np.ones((cols, inner), dtype=np.float32)
    pack = pack_linear(weight, np.zeros(cols, dtype=np.float32), kind)
    activation = np.ones((rows, inner), dtype=np.float32)
    gathered = np.zeros((rows, pack.stored_k), dtype=np.float32)
    for column in range(pack.stored_k):
        source = int(pack.perm[column])
        if source < inner:
            gathered[:, column] = activation[:, source]
    if kind == "fp4":
        restored = dequant_fp4(pack.packed, pack.scales, pack.stored_k)
    else:
        restored = dequant_fp8(pack.packed, pack.scales, pack.stored_k)
    expect = gathered @ restored[:cols].T
    gpu = Gpu()
    try:
        src = gpu.upload(activation)
        dest = gpu.empty(rows * pack.padded_n * 4)
        half = gpu.empty(rows * pack.stored_k * 2)
        uploaded = {
            "packed": gpu.upload(pack.packed),
            "scales": gpu.upload(pack.scales),
            "bias": gpu.upload(pack.bias),
            "perm": gpu.upload(pack.perm),
        }
        gpu.project(src, inner, rows, pack, dest, uploaded, half)
        got = gpu.read_f32(dest, rows * cols).reshape(rows, cols)
    finally:
        gpu.close()
    if not np.allclose(got, expect, atol=1e-2):
        message = f"{kind} matmul mismatch"
        raise RuntimeError(message)


def _attend() -> None:
    from wan22_i2v.metal import Gpu

    shapes = ((5, 7, 2, 128), (8, 8, 1, 128), (9, 4, 2, 128), (3, 3, 1, 64))
    rng = np.random.default_rng(0)
    gpu = Gpu()
    try:
        for q_len, kv_len, heads, dim in shapes:
            query = rng.standard_normal((q_len, heads, dim)).astype(np.float32)
            key = rng.standard_normal((kv_len, heads, dim)).astype(np.float32)
            value = rng.standard_normal((kv_len, heads, dim)).astype(np.float32)
            scale = float(dim) ** -0.5
            scores = np.einsum("qhd,khd->qkh", query, key) * scale
            scores -= scores.max(axis=1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=1, keepdims=True)
            expect = np.einsum("qkh,khd->qhd", weights, value)
            dest = gpu.empty(q_len * heads * dim * 4)
            gpu.attend(
                gpu.upload(query),
                gpu.upload(key),
                gpu.upload(value),
                dest,
                q_len,
                kv_len,
                heads,
                dim,
            )
            got = gpu.read_f32(dest, q_len * heads * dim).reshape(q_len, heads, dim)
            if not np.allclose(got, expect, atol=2e-3, rtol=2e-3):
                message = f"attend mismatch {q_len}x{kv_len} heads {heads} dim {dim}"
                raise RuntimeError(message)
    finally:
        gpu.close()


def _spatial() -> None:
    from wan22_i2v.metal import Gpu

    shapes = ((32, 1, 2, 3), (384, 1, 2, 2), (48, 1, 2, 2))
    rng = np.random.default_rng(1)
    gpu = Gpu()
    try:
        for channels, time, height, width in shapes:
            spatial = height * width
            qkv = rng.standard_normal((3 * channels, time, height, width)).astype(np.float32)
            scale = float(channels) ** -0.5
            expect = np.zeros((channels, time, height, width), dtype=np.float32)
            for t in range(time):
                query = qkv[:channels, t].reshape(channels, spatial)
                key = qkv[channels : 2 * channels, t].reshape(channels, spatial)
                value = qkv[2 * channels :, t].reshape(channels, spatial)
                scores = (query.T @ key) * scale
                scores -= scores.max(axis=1, keepdims=True)
                weights = np.exp(scores)
                weights /= weights.sum(axis=1, keepdims=True)
                expect[:, t] = (value @ weights.T).reshape(channels, height, width)
            dest = gpu.empty(channels * time * spatial * 4)
            gpu.spatial_attend(
                gpu.upload(qkv), dest, channels, time, height, width
            )
            got = gpu.read_f32(dest, channels * time * spatial).reshape(
                channels, time, height, width
            )
            if not np.allclose(got, expect, atol=2e-3, rtol=2e-3):
                message = f"spatial mismatch {channels} {time}x{height}x{width}"
                raise RuntimeError(message)
    finally:
        gpu.close()


def _patch() -> None:
    from wan22_i2v.metal import Gpu

    rng = np.random.default_rng(2)
    cases = ((4, 2, 4, 4, 70), (36, 1, 2, 2, 64))
    gpu = Gpu()
    try:
        for channels, time, height, width, dim in cases:
            volume = rng.standard_normal((channels, time, height, width)).astype(np.float32)
            weight = rng.standard_normal((dim, channels, 2, 2)).astype(np.float16)
            bias = rng.standard_normal(dim).astype(np.float16)
            grid_h = height // 2
            grid_w = width // 2
            spatial = grid_h * grid_w
            tokens = time * spatial
            expect = np.zeros((tokens, dim), dtype=np.float32)
            for token in range(tokens):
                frame = token // spatial
                rem = token - frame * spatial
                ph = rem // grid_w
                pw = rem - ph * grid_w
                for out in range(dim):
                    acc = float(bias[out])
                    for channel in range(channels):
                        for kh in range(2):
                            for kw in range(2):
                                value = volume[channel, frame, ph * 2 + kh, pw * 2 + kw]
                                acc += float(value) * float(weight[out, channel, kh, kw])
                    expect[token, out] = acc
            dest = gpu.empty(tokens * dim * 4)
            gpu.patch(
                gpu.upload(volume),
                gpu.upload(weight),
                gpu.upload(bias),
                dest,
                channels,
                time,
                height,
                width,
                dim,
            )
            got = gpu.read_f32(dest, tokens * dim).reshape(tokens, dim)
            if not np.allclose(got, expect, atol=2e-2, rtol=2e-2):
                message = f"patch mismatch {channels} {time}x{height}x{width} dim {dim}"
                raise RuntimeError(message)
    finally:
        gpu.close()
