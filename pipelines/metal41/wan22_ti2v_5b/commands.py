"""setup, quantize, kernels, smoke, and generate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from favlo.cli import repo_root
from favlo.logs import favlo_home
from favlo.registry import parts_for_directory
from favlo.smoke_lock import hold, release
from wan22_ti2v_5b.gpu import Gpu
from wan22_ti2v_5b.quant import pack_fp4, pack_fp8, pack_int8, quantize_activation
from wan22_ti2v_5b.quantize import quantize
from wan22_ti2v_5b.request import VideoRequest
from wan22_ti2v_5b.runtime import open_runtime
from wan22_ti2v_5b.schedule import require_knots, timesteps

_PIPELINE = "wan22-ti2v-5b"
_STUDENT = "quanhaol/Wan2.2-TI2V-5B-Turbo"
_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"favlo --{_PIPELINE}")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="download the VAE")
    setup.add_argument("--student-dir", type=Path, default=None)
    quant = sub.add_parser("quantize", help="pack a local DiT checkpoint")
    quant.add_argument("--weights", type=Path, required=True)
    quant.add_argument("--copy", default=None)
    sub.add_parser("kernels", help="compile and check matmul2d")
    sub.add_parser("ceilings", help="time the isolated kernels")
    sub.add_parser("smoke", help="832x480 text clip and still clip")
    generate = sub.add_parser("generate", help="write one mp4")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--size", default="1280x704")
    generate.add_argument("--frames", type=int, default=9)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--image", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "setup":
        return _setup(args.student_dir)
    if args.command == "quantize":
        quantize(args.weights, args.copy, repo_root())
        return 0
    if args.command == "kernels":
        return _locked(_kernels, "kernels")
    if args.command == "ceilings":
        return _locked(_ceilings, "ceilings")
    if args.command == "smoke":
        return _locked(_smoke, "smoke")
    if args.command == "generate":
        return _locked(lambda: _generate(args), "generate")
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
    gpu = Gpu("wan22-ti2v-5b-kernels")
    try:
        _check_kind(gpu, "fp4")
        _check_kind(gpu, "fp8")
        _check_int8(gpu)
        _check_conv(gpu)
        from wan22_ti2v_5b.stream import check_stream
        from wan22_ti2v_5b.vae import check_activate, check_fused_conv

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
    codes, row_scale = quantize_activation(left)
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
    gpu = Gpu("wan22-ti2v-5b-ceilings")
    try:
        square = _time_i8(gpu, 576, 3072, 3072)
        up = _time_i8(gpu, 576, 14336, 3072)
        down = _time_i8(gpu, 576, 3072, 14336)
        qk = _time_f16(gpu, 576, 576, 128, transpose=True)
        mix = _time_f16(gpu, 576, 128, 576, transpose=False)
        cross_qk = _time_f16(gpu, 576, 512, 128, transpose=True)
        cross_mix = _time_f16(gpu, 576, 128, 512, transpose=False)
        native_square = _time_i8(gpu, 27328, 3072, 3072)
        native_up = _time_i8(gpu, 27328, 14336, 3072)
        native_down = _time_i8(gpu, 27328, 3072, 14336)
        native_qk = _time_f16(gpu, 64, 27296, 128, transpose=True)
        native_av = _time_f16(gpu, 64, 128, 27296, transpose=False)
    finally:
        gpu.close()
    smoke = (
        960 * square
        + 120 * up
        + 120 * down
        + 2880 * qk
        + 2880 * mix
        + 2880 * cross_qk
        + 2880 * cross_mix
    )
    native_linear = 960 * native_square + 120 * native_up + 120 * native_down
    tiles = (27280 + 63) // 64
    native_attn = tiles * 24 * 30 * 4 * (native_qk + native_av)
    print(f"smoke int8 square {square:.6f}", flush=True)
    print(f"smoke int8 up {up:.6f}", flush=True)
    print(f"smoke int8 down {down:.6f}", flush=True)
    print(f"smoke qk {qk:.6f}", flush=True)
    print(f"smoke av {mix:.6f}", flush=True)
    print(f"smoke cross qk {cross_qk:.6f}", flush=True)
    print(f"smoke cross av {cross_mix:.6f}", flush=True)
    print(f"smoke dit {smoke:.3f}", flush=True)
    print(f"native square {native_square:.6f}", flush=True)
    print(f"native up {native_up:.6f}", flush=True)
    print(f"native down {native_down:.6f}", flush=True)
    print(f"native linear {native_linear:.3f}", flush=True)
    print(f"native qk tile {native_qk:.6f}", flush=True)
    print(f"native av tile {native_av:.6f}", flush=True)
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


def _time_f16(gpu: Gpu, rows: int, cols: int, k_dim: int, *, transpose: bool) -> float:
    rng = np.random.default_rng(rows + cols + k_dim)
    left = rng.standard_normal((rows, k_dim)).astype(np.float16)
    if transpose:
        right = rng.standard_normal((cols, k_dim)).astype(np.float16)
        name = "gemm_f16_tn"
        stride_b = k_dim
    else:
        right = np.zeros((k_dim, cols), dtype=np.float16)
        right[:, :] = rng.standard_normal((k_dim, cols)).astype(np.float16)
        name = "gemm_f16_nn"
        stride_b = cols
    left_h = gpu.upload(np.ascontiguousarray(left))
    right_h = gpu.upload(np.ascontiguousarray(right))
    dest = gpu.zeros(rows * cols * 4)
    dims = np.array([rows, cols, k_dim, k_dim, stride_b], dtype=np.int32)

    def once() -> None:
        gpu._gemm(name, (left_h, right_h, dest), dims, scales=False)

    seconds = _median(once)
    gpu.release(left_h)
    gpu.release(right_h)
    gpu.release(dest)
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


def _smoke() -> int:
    card = json.loads((repo_root() / "docs" / _PIPELINE / "smoke.json").read_text(encoding="utf-8"))
    record = repo_root() / "models" / "Wan2.2-TI2V-5B-int8"
    runtime = open_runtime(
        record,
        "fp8",
        None,
        None,
        owner="wan22-ti2v-5b-smoke",
        parts=parts_for_directory(record),
    )
    home = favlo_home()
    try:
        text = _one(runtime, card, image=None, home=home)
        still_card = dict(card)
        still_card["width"] = int(card["i2v_width"])
        still_card["height"] = int(card["i2v_height"])
        still = _load_image(
            repo_root() / "docs" / _PIPELINE / "still.png",
            int(still_card["height"]),
            int(still_card["width"]),
        )
        image = _one(runtime, still_card, image=still, home=home)
    finally:
        runtime.close()
    _mp4(home, text["id"], int(card["frames"]))
    _mp4(home, image["id"], int(card["frames"]))
    _drop_student(record)
    print("smoke ok", flush=True)
    return 0


def _one(runtime, card: dict, image: np.ndarray | None, home: Path) -> dict:
    request = VideoRequest(
        prompt=str(card["prompt"]),
        negative_prompt="",
        width=int(card["width"]),
        height=int(card["height"]),
        frames=int(card["frames"]),
        seed=int(card["seed"]),
        quality="high",
        image=image,
    )
    videos: dict = {}
    payload, paused = runtime.clip.sample(
        request,
        video_id="video_smoke_" + ("i2v" if image is not None else "t2v"),
        home=home,
        videos=videos,
        stop=None,
    )
    if paused is not None or payload.get("status") != "completed":
        message = "smoke clip did not finish"
        raise RuntimeError(message)
    print(payload["id"], payload["size"], payload["seconds"], flush=True)
    return payload


def _load_image(path: Path, height: int, width: int) -> np.ndarray:
    """PNG to ``[-1, 1]`` HWC, resized to the clip when the still is another size."""
    from wan22_ti2v_5b.png import decode_png
    from wan22_ti2v_5b.request import _resize

    image = decode_png(path.read_bytes())
    if image.shape[0] != height or image.shape[1] != width:
        image = _resize(image, width, height)
    return image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)


def _mp4(home: Path, video_id: str, frames: int) -> None:
    path = home / "videos" / f"{video_id}.mp4"
    if not path.is_file() or path.stat().st_size < 1000:
        message = "mp4 is missing"
        raise RuntimeError(message)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != str(frames):
        message = "mp4 frame count does not match"
        raise RuntimeError(message)


def _drop_student(record: Path) -> None:
    source = record / "source.json"
    if not source.is_file():
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    weights = Path(str(payload.get("weights", "")))
    if weights.is_file() and weights.name == "model.pt":
        weights.unlink()
        print("removed student checkpoint", flush=True)


def _generate(args) -> int:
    record = repo_root() / "models" / "Wan2.2-TI2V-5B-int8"
    runtime = open_runtime(
        record,
        "fp8",
        None,
        None,
        owner="wan22-ti2v-5b-generate",
        parts=parts_for_directory(record),
    )
    try:
        width = int(args.size.lower().split("x")[0])
        height = int(args.size.lower().split("x")[1])
        image = None
        if args.image is not None:
            image = _load_image(args.image, height, width)
        card = {
            "prompt": args.prompt,
            "width": width,
            "height": height,
            "frames": args.frames,
            "seed": args.seed,
        }
        _one(runtime, card, image, favlo_home())
    finally:
        runtime.close()
    return 0
