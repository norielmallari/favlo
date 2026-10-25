"""CLI for quantize, kernels, and generate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from wan22_i2v.formats import TILE_M, dequant_fp4, dequant_fp8, pack_linear
from wan22_i2v.png import decode_png
from wan22_i2v.schedule import (
    OFFICIAL,
    P320_HEIGHT,
    P320_WIDTH,
    PROMOTE_PERCENT,
    SHORT_FRAMES,
    STEPS,
)
from wan22_i2v.source import repo_id

_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RECORD = "Wan2.2-I2V-A14B-fp4"
_INT8 = "Wan2.2-I2V-A14B-int8"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="favlo --wan22-i2v")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="download the VAE, umT5, and the two experts")
    setup.add_argument(
        "--umt5",
        type=_umt5_repo,
        default=None,
        help="umT5 repository, org/name or a huggingface.co link",
    )
    sub.add_parser("kernels", help="dispatch the fp4, fp8, and int8 matmul")
    quant = sub.add_parser("quantize", help="pack the high-noise and low-noise experts")
    quant.add_argument("--high", type=Path, required=True)
    quant.add_argument("--low", type=Path, required=True)
    quant.add_argument("--config", type=Path)
    quant.add_argument("--out", type=Path, help="packed directory")
    quant.add_argument("--name", type=_folder_name, default=None, help="folder under models/")
    quant.add_argument("--high-dir", type=Path, default=None, help="quantized high-noise directory")
    quant.add_argument("--low-dir", type=Path, default=None, help="quantized low-noise directory")
    quant.add_argument(
        "--fp8-percent",
        type=int,
        default=PROMOTE_PERCENT,
        help="share of blocks stored as fp8",
    )
    quant.add_argument(
        "--int8",
        action="store_true",
        help="pack every DiT linear as per-channel int8",
    )
    text = sub.add_parser("quantize-text", help="pack the umT5 encoder")
    text.add_argument("--out", type=Path, help="packed umT5 directory")
    text.add_argument("--name", type=_folder_name, default=None, help="packed model folder")
    text.add_argument(
        "--umt5",
        type=_umt5_repo,
        default=None,
        help="umT5 repository already under models/",
    )
    gen = sub.add_parser("generate", help="write one AVI")
    gen.add_argument("--record", type=Path, required=True)
    gen.add_argument("--image", type=Path, required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--frames", type=int, default=SHORT_FRAMES)
    gen.add_argument("--width", type=int, default=P320_WIDTH)
    gen.add_argument("--height", type=int, default=P320_HEIGHT)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--negative-prompt", default="")
    gen.add_argument("--guidance-scale", type=float, nargs="+")
    gen.add_argument("--steps", type=int, default=STEPS)
    gen.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "setup":
        from wan22_i2v.source import dit_dir, download, expert_paths, umt5_dir, vae_dir

        download(umt5=args.umt5)
        high, low = expert_paths()
        print(vae_dir(), flush=True)
        print(umt5_dir(repo=args.umt5), flush=True)
        print(dit_dir(), flush=True)
        print(high, flush=True)
        print(low, flush=True)
        return 0
    if args.command == "kernels":
        return kernels()
    if args.command == "quantize":
        return _quantize(args)
    if args.command == "quantize-text":
        return _quantize_text(args)
    if args.command == "generate":
        return _generate(args)
    return 2


def _guidance(values: list[float] | None) -> tuple[float, float]:
    if not values:
        return (1.0, 1.0)
    if len(values) == 1:
        return (values[0], values[0])
    if len(values) == 2:
        return (values[0], values[1])
    message = "guidance-scale is one number or a low, high pair"
    raise ValueError(message)


def kernels() -> int:
    _matmul("fp4")
    _matmul("fp8")
    _matmul_i8()
    _attend()
    _spatial()
    _patch()
    print("fp4 ok", flush=True)
    print("fp8 ok", flush=True)
    print("int8 ok", flush=True)
    print("attend ok", flush=True)
    print("spatial ok", flush=True)
    print("patch ok", flush=True)
    return 0


def _umt5_repo(text: str) -> str:
    try:
        return repo_id(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _folder_name(text: str) -> str:
    if _FOLDER.fullmatch(text) is None:
        message = "folder name is not a single name"
        raise argparse.ArgumentTypeError(message)
    return text


def _record_dir(name: str | None, out: Path | None, percent: int = PROMOTE_PERCENT) -> Path:
    from wan22_i2v.source import models_dir

    if name is not None and out is not None:
        message = "--name and --out name the same directory"
        raise ValueError(message)
    if name is not None:
        return models_dir() / name
    if out is not None:
        return out
    if percent != PROMOTE_PERCENT:
        return models_dir() / f"Wan2.2-I2V-A14B-fp8-e{percent}"
    return models_dir() / _RECORD


def _parts(high_dir: Path, low_dir: Path, *, int8: bool) -> dict[str, Path]:
    """Expert directories, and the int8 encoder a clip reads."""
    parts = {"high": high_dir, "low": low_dir}
    if int8:
        from wan22_i2v.umt5_pack import shared_pack_dir

        parts["umt5"] = shared_pack_dir()
    return parts


def _quantize(args: argparse.Namespace) -> int:
    from favlo.registry import register_quantized
    from wan22_i2v.record import quantize

    try:
        if args.config is None:
            config = OFFICIAL
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
        percent = int(args.fp8_percent)
        if args.int8 and args.name is None and args.out is None:
            from wan22_i2v.source import models_dir

            dest = models_dir() / _INT8
        else:
            dest = _record_dir(args.name, args.out, percent)
        high_dir = args.high_dir or dest / "high"
        low_dir = args.low_dir or dest / "low"
        if args.int8:
            from wan22_i2v.record import quantize_int8

            manifest = quantize_int8(
                args.high,
                args.low,
                dest,
                config,
                high_dir=high_dir,
                low_dir=low_dir,
            )
        else:
            manifest = quantize(
                args.high,
                args.low,
                dest,
                config,
                high_dir=high_dir,
                low_dir=low_dir,
                percent=percent,
            )
        register_quantized(
            dest.name,
            source=args.high,
            alias=None,
            directory=dest,
            pipeline="wan22-i2v",
            runtime="metal41",
            max_prompt_tokens=int(config["text_len"]),
            parts=_parts(high_dir, low_dir, int8=bool(args.int8)),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest, flush=True)
    if args.int8:
        print(f"int8 high worst {manifest['high_worst']:.6f}", flush=True)
        print(f"int8 low worst {manifest['low_worst']:.6f}", flush=True)
    else:
        print(f"high fp8 blocks {manifest['high_promoted']}", flush=True)
        print(f"low fp8 blocks {manifest['low_promoted']}", flush=True)
    return 0


def _quantize_text(args: argparse.Namespace) -> int:
    from favlo.registry import set_part
    from wan22_i2v.text import encoder_path
    from wan22_i2v.umt5_pack import quantize_umt5

    try:
        from wan22_i2v.umt5_pack import shared_pack_dir

        folder = args.name or _INT8
        dest = args.out if args.out is not None else shared_pack_dir()
        src = None if args.umt5 is None else encoder_path(repo=args.umt5)
        worst = quantize_umt5(src=src, dest=dest)
        set_part(folder, "umt5", dest)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    print(dest, flush=True)
    print(f"int8 umt5 worst {float(worst):.6f}", flush=True)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from wan22_i2v.runtime import Runtime

    image = decode_png(args.image.read_bytes())
    try:
        guidance = _guidance(args.guidance_scale)
    except ValueError as exc:
        print(exc, file=sys.stderr, flush=True)
        return 2
    from favlo.registry import parts_for_directory

    runtime = Runtime(args.record, parts=parts_for_directory(args.record))
    try:
        avi, timing = runtime.generate(
            args.prompt,
            image,
            args.frames,
            args.width,
            args.height,
            args.seed,
            negative_prompt=args.negative_prompt,
            guidance=guidance,
            steps=args.steps,
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


def _matmul_i8() -> None:
    from wan22_i2v.metal import Gpu

    rows = TILE_M
    cols = 128
    inner = 256
    rng = np.random.default_rng(0)
    weight = rng.normal(0.0, 0.02, size=(cols, inner)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, size=(cols,)).astype(np.float32)
    activation = rng.normal(0.0, 0.5, size=(rows, inner)).astype(np.float32)
    pack = pack_linear(weight, bias, "int8")
    peak = np.max(np.abs(activation), axis=1)
    row_scale = np.where(peak > 0, peak / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    inverse = np.where(peak > 0, np.float32(127.0) / peak, np.float32(0.0)).astype(np.float32)
    codes = np.clip(np.rint(activation * inverse[:, None]), -127, 127).astype(np.int32)
    product = codes @ pack.packed[:cols, :inner].astype(np.int32).T
    expect = product.astype(np.float32) * row_scale[:, None] * pack.scales[:cols].astype(np.float32)
    expect += bias.astype(np.float32)
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
    if not np.allclose(got, expect, rtol=1e-4, atol=1e-4):
        message = "int8 matmul mismatch"
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
