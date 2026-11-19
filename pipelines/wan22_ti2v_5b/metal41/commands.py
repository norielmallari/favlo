"""setup, quantize, kernels, and ceilings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from favlo.cli import repo_root
from favlo.smoke_lock import hold, release
from wan22_ti2v_5b.metal41.gpu import Gpu, _Dispatch, _pad
from wan22_ti2v_5b.metal41.quant import pack_fp4, pack_fp8, pack_int8, quantize_activation
from wan22_ti2v_5b.metal41.quantize import quantize
from wan22_ti2v_5b.metal41.schedule import require_knots, timesteps
from wan22_ti2v_5b.metal41.spec import DIM, LAYERS
from wan22_ti2v_5b.metal41.stream import _FLASH_K, _FLASH_Q, launch_flash

_PIPELINE = "wan22_ti2v_5b"
_STUDENT = "quanhaol/Wan2.2-TI2V-5B-Turbo"
_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"


def main(argv: list[str] | None = None) -> int:
    from umt5_xxl.metal41.commands import attach_umt5_commands, dispatch_umt5

    parser = argparse.ArgumentParser(prog=f"favlo --{_PIPELINE}")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="download the VAE")
    setup.add_argument("--student-dir", type=Path, default=None)
    umt5 = sub.add_parser("umt5", help="download and pack the shared umT5 encoder")
    attach_umt5_commands(umt5)
    quant = sub.add_parser("quantize", help="pack a local DiT checkpoint")
    quant.add_argument("--weights", type=Path, required=True)
    quant.add_argument("--copy", default=None)
    sub.add_parser("kernels", help="compile and check matmul2d")
    sub.add_parser("ceilings", help="time the isolated kernels")
    args = parser.parse_args(argv)
    if args.command == "setup":
        return _setup(args.student_dir)
    if args.command == "umt5":
        return dispatch_umt5(args)
    if args.command == "quantize":
        quantize(args.weights, args.copy, repo_root())
        return 0
    if args.command == "kernels":
        return _locked(_kernels, "kernels")
    if args.command == "ceilings":
        return _locked(_ceilings, "ceilings")
    return 2


def _setup(student_dir: Path | None) -> int:
    from huggingface_hub import hf_hub_download

    from favlo.registry import models_dir

    dest = models_dir(repo_root()) / "Wan-AI" / "Wan2.2-TI2V-5B"
    dest.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=_VAE_REPO, filename="Wan2.2_VAE.pth", local_dir=str(dest))
    print("vae Wan2.2_VAE.pth", flush=True)
    if student_dir is not None:
        student_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=_STUDENT, filename="model.pt", local_dir=str(student_dir))
        print("dit model.pt", flush=True)
    return 0


def _locked(fn, command: str) -> int:
    root = repo_root()
    created = hold(root=root, pipeline=_PIPELINE, command=command)
    try:
        return fn()
    finally:
        if created:
            release(root=root)


def _kernels() -> int:
    print("timesteps " + " ".join(f"{value:.6f}" for value in timesteps()), flush=True)
    require_knots(timesteps())
    gpu = Gpu("wan22_ti2v_5b-kernels")
    try:
        _check_kind(gpu, "fp4")
        _check_kind(gpu, "fp8")
        _check_int8(gpu)
        _check_conv(gpu)
        from wan22_ti2v_5b.metal41.stream import check_stream
        from wan22_ti2v_5b.metal41.vae import check_activate, check_fused_conv

        check_stream(gpu)
        check_activate(gpu)
        check_fused_conv(gpu)
    finally:
        gpu.close()
    print("kernels ok", flush=True)
    return 0


def _check_kind(gpu: Gpu, kind: str) -> None:
    rng = np.random.default_rng(0 if kind == "fp4" else 1)
    weight = rng.standard_normal((32, 256)).astype(np.float32) * np.float32(0.05)
    left = rng.standard_normal((64, 256)).astype(np.float32)
    if kind == "fp4":
        packed, scales, dequant = pack_fp4(weight)
    else:
        packed, scales, dequant = pack_fp8(weight)
    got = gpu.linear(left, packed, scales, kind=kind)
    ref = left.astype(np.float16).astype(np.float32) @ dequant.T
    error = float(np.max(np.abs(got - ref)))
    print(f"{kind} max abs {error:.6e}", flush=True)
    if error > 1e-3:
        message = f"{kind} matmul drifted"
        raise RuntimeError(message)


def _check_int8(gpu: Gpu) -> None:
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((128, 256)).astype(np.float32) * np.float32(0.05)
    left = rng.standard_normal((64, 256)).astype(np.float32)
    bias = rng.standard_normal(128).astype(np.float32)
    packed, scales, _dequant = pack_int8(weight)
    got = gpu.linear(left, packed, scales, kind="int8", bias=bias)
    rounded = left.astype(np.float16).astype(np.float32)
    codes, row_scale = quantize_activation(rounded)
    product = codes.astype(np.int32) @ packed.astype(np.int32).T
    ref = product.astype(np.float32) * row_scale[:, None] * scales[None, :]
    ref = ref + bias.astype(np.float16).astype(np.float32)
    error = float(np.max(np.abs(got - ref)))
    print(f"int8 max abs {error:.6e}", flush=True)
    if error > 1e-3:
        message = "int8 matmul drifted"
        raise RuntimeError(message)


def _check_conv(gpu: Gpu) -> None:
    source = np.ones((1, 1, 2, 2), dtype=np.float32)
    weight = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    out = gpu.conv3d(source, weight, bias, stride=(1, 1, 1), padding=(0, 0, 0))
    if out.shape != (1, 1, 2, 2) or float(np.max(np.abs(out - 1))) > 1e-5:
        message = "conv3d drifted"
        raise RuntimeError(message)
    rng = np.random.default_rng(3)
    wide_source = rng.standard_normal((4, 2, 48, 48)).astype(np.float32)
    wide_weight = rng.standard_normal((32, 4, 3, 3, 3)).astype(np.float32) * np.float32(0.02)
    wide_bias = rng.standard_normal(32).astype(np.float32)
    wide = gpu.conv3d(wide_source, wide_weight, wide_bias, stride=(1, 1, 1), padding=(1, 1, 1))
    reference = _conv_reference(wide_source, wide_weight, wide_bias)
    error = float(np.max(np.abs(wide - reference)))
    print(f"conv3d max abs {error:.6e}", flush=True)
    if wide.shape != reference.shape or error > 2e-2:
        message = "conv3d drifted"
        raise RuntimeError(message)


def _conv_reference(source: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Same-pad stride-1 convolution. The wide kernel check compares against this."""
    in_c, frames, height, width = source.shape
    out_c, _, kt, kh, kw = weight.shape
    out = np.broadcast_to(bias.reshape(out_c, 1, 1, 1), (out_c, frames, height, width)).copy()
    for ic in range(in_c):
        for kt_i in range(kt):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    t_out, t_in = _same_span(kt_i - 1, frames)
                    y_out, y_in = _same_span(kh_i - 1, height)
                    x_out, x_in = _same_span(kw_i - 1, width)
                    patch = source[ic, t_in, y_in, x_in]
                    out[:, t_out, y_out, x_out] += (
                        weight[:, ic, kt_i, kh_i, kw_i].reshape(out_c, 1, 1, 1) * patch
                    )
    return out


def _same_span(offset: int, length: int) -> tuple[slice, slice]:
    """Output and input slices where the output reads ``index + offset``."""
    if offset >= 0:
        return slice(0, length - offset), slice(offset, length)
    return slice(-offset, length), slice(0, length + offset)


def _ceilings() -> int:
    """Isolated kernel times. The print is this run."""
    gpu = Gpu("wan22_ti2v_5b-ceilings")
    try:
        calls = LAYERS * 4
        square = _time_i8(gpu, 576, 3072, 3072)
        up = _time_i8(gpu, 576, 14336, 3072)
        down = _time_i8(gpu, 576, 3072, 14336)
        self_attn = _time_flash(gpu, 540, 540)
        cross_attn = _time_flash(gpu, 540, 512)
        native_square = _time_i8(gpu, 27328, 3072, 3072)
        native_up = _time_i8(gpu, 27328, 14336, 3072)
        native_down = _time_i8(gpu, 27328, 3072, 14336)
        native_self = _time_flash(gpu, 27280, 27280)
        native_cross = _time_flash(gpu, 27280, 512)
    finally:
        gpu.close()
    smoke = calls * (8 * square + up + down + self_attn + cross_attn)
    native_linear = calls * (8 * native_square + native_up + native_down)
    native_attn = calls * (native_self + native_cross)
    print(f"smoke int8 square {square:.6f}", flush=True)
    print(f"smoke int8 up {up:.6f}", flush=True)
    print(f"smoke int8 down {down:.6f}", flush=True)
    print(f"smoke self {self_attn:.6f}", flush=True)
    print(f"smoke cross {cross_attn:.6f}", flush=True)
    print(f"smoke dit {smoke:.3f}", flush=True)
    print(f"native square {native_square:.6f}", flush=True)
    print(f"native up {native_up:.6f}", flush=True)
    print(f"native down {native_down:.6f}", flush=True)
    print(f"native linear {native_linear:.3f}", flush=True)
    print(f"native self {native_self:.6f}", flush=True)
    print(f"native cross {native_cross:.6f}", flush=True)
    print(f"native attention {native_attn:.3f}", flush=True)
    flops = 2.0 * 27328 * 3072 * 14336
    print(f"native down tops {(flops / native_down) / 1e12:.2f}", flush=True)
    return 0


def _time_i8(gpu: Gpu, rows: int, out_n: int, cols: int) -> float:
    rng = np.random.default_rng(rows + out_n + cols)
    left = rng.standard_normal((rows, cols)).astype(np.float32)
    weight = rng.standard_normal((out_n, cols)).astype(np.float32) * np.float32(0.02)
    packed, scales, _dequant = pack_int8(weight)
    codes, row_scale = quantize_activation(left)
    bias = np.zeros(out_n, dtype=np.float16)
    left_h = gpu.upload(codes)
    weight_h = gpu.upload(packed)
    row_h = gpu.upload(row_scale)
    scale_h = gpu.upload(scales)
    bias_h = gpu.upload(bias)
    dest = gpu.zeros(rows * out_n * 4)
    dims = np.array([rows, out_n, cols, cols, cols], dtype=np.int32)
    shape = np.array([rows, out_n, out_n], dtype=np.int32)

    def once() -> None:
        gpu._gemm("gemm_i8", (left_h, weight_h, dest), dims, scales=False, tile_n=128)
        gpu._launch("scale_i8", (dest, row_h, scale_h, bias_h), shape.tobytes(), rows * out_n)

    seconds = _median(once)
    for handle in (left_h, weight_h, row_h, scale_h, bias_h, dest):
        gpu.release(handle)
    return seconds


def _time_flash(gpu: Gpu, q_len: int, kv_len: int) -> float:
    q_cap = max(_pad(q_len, 64), _pad(q_len, _FLASH_Q))
    kv_cap = max(_pad(kv_len, 64), _pad(kv_len, _FLASH_K))
    rng = np.random.default_rng(q_len + kv_len)
    query = gpu.upload(rng.standard_normal((q_cap, DIM)).astype(np.float16))
    key = gpu.upload(rng.standard_normal((kv_cap, DIM)).astype(np.float16))
    value = gpu.upload(rng.standard_normal((kv_cap, DIM)).astype(np.float16))
    dest = gpu.zeros(q_cap * DIM * 4)

    def once() -> None:
        group = _Dispatch(gpu)
        launch_flash(group, query, key, value, dest, q_len, kv_len, q_cap, kv_cap)
        group.finish("attend_flash")

    seconds = _median(once)
    for handle in (query, key, value, dest):
        gpu.release(handle)
    return seconds


def _median(once) -> float:
    once()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        once()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[1]

