"""Isolated kernel times for one 9-frame 576×320 clip.

Each sample is one command buffer. The first dispatch is discarded. The
recorded time is the median of the next five, in milliseconds.
"""

from __future__ import annotations

import struct

import numpy as np

from wan22_i2v.metal import Gpu

_GIB = 2 * 1024**3


def median_ms(gpu: Gpu, launch) -> float:
    launch()
    gpu.gpu_ms()
    samples = [0.0] * 5
    for index in range(5):
        launch()
        samples[index] = gpu.gpu_ms()
    samples.sort()
    return samples[2]


def run() -> None:
    gpu = Gpu()
    try:
        _bandwidth(gpu)
        _matmuls(gpu)
        _attend(gpu)
        _pointwise(gpu)
        _vae(gpu)
        _text(gpu)
    finally:
        gpu.close()


def _bandwidth(gpu: Gpu) -> None:
    count = _GIB // 4
    source = gpu.empty(count * 4)
    dest = gpu.empty(count * 4)
    ms = median_ms(gpu, lambda: gpu.copy_f32(source, 0, dest, count))
    traffic = 2 * _GIB
    gbps = (traffic / (ms / 1000.0)) / 1e9
    print(f"copy_f32 2GiB {ms:.3f} ms {gbps:.2f} GB/s", flush=True)


def _matmuls(gpu: Gpu) -> None:
    shapes = (
        ("fp4", 2176, 5120, 5120),
        ("fp8", 2176, 5120, 5120),
        ("fp4", 2176, 13824, 5120),
        ("fp8", 2176, 13824, 5120),
        ("fp4", 2176, 5120, 13824),
        ("fp8", 2176, 5120, 13824),
        ("fp4", 512, 5120, 5120),
        ("fp8", 512, 5120, 5120),
        ("fp4", 512, 5120, 4096),
        ("fp4", 64, 5120, 256),
        ("fp4", 64, 5120, 5120),
        ("fp4", 64, 30720, 5120),
        ("fp4", 2176, 64, 5120),
    )
    for kind, rows, cols, inner in shapes:
        weight_bytes = cols * inner // 2 if kind == "fp4" else cols * inner
        act = gpu.empty(rows * inner * 2)
        weight = gpu.empty(weight_bytes)
        scales = gpu.empty(cols * (inner // 32))
        dest = gpu.empty(rows * cols * 4)
        ms = median_ms(
            gpu,
            lambda act=act, weight=weight, scales=scales, dest=dest, rows=rows, cols=cols, inner=inner, kind=kind: gpu.matmul(
                kind, act, weight, scales, dest, rows, cols, inner
            ),
        )
        print(f"matmul_{kind} {rows}x{cols} K{inner} {ms:.3f}", flush=True)


def _attend(gpu: Gpu) -> None:
    for q_len, kv_len, heads, dim in ((2160, 2160, 40, 128), (2160, 512, 40, 128)):
        q = gpu.empty(q_len * heads * dim * 4)
        k = gpu.empty(kv_len * heads * dim * 4)
        v = gpu.empty(kv_len * heads * dim * 4)
        dest = gpu.empty(q_len * heads * dim * 4)
        ms = median_ms(
            gpu,
            lambda q=q, k=k, v=v, dest=dest, q_len=q_len, kv_len=kv_len, heads=heads, dim=dim: gpu.attend(
                q, k, v, dest, q_len, kv_len, heads, dim
            ),
        )
        print(f"attend {q_len}x{kv_len} {heads}x{dim} {ms:.3f}", flush=True)


def _pointwise(gpu: Gpu) -> None:
    rows, cols = 2176, 5120
    src = gpu.empty(rows * cols * 4)
    other = gpu.empty(rows * cols * 4)
    out = gpu.empty(rows * cols * 4)
    half = gpu.empty(rows * cols * 2)
    perm = gpu.upload(np.arange(cols, dtype=np.uint32))
    factor = gpu.empty(rows * 4)
    bias = gpu.empty(cols * 4)
    shift = gpu.empty(cols * 4)
    scale = gpu.empty(cols * 4)
    gate = gpu.empty(cols * 4)
    weight = gpu.empty(cols * 4)
    wide = gpu.empty(rows * 13824 * 4)
    packed = [struct.pack("I", value) for value in (rows, cols, cols, cols)]
    jobs = (
        (
            "scale_gather",
            lambda: gpu._dispatch(
                "scale_gather",
                [src, perm, half, factor],
                packed,
                (rows, 1, 1),
                (128, 1, 1),
            ),
        ),
        ("rescale_rows", lambda: gpu._span("rescale_rows", [src, factor], rows, cols, [])),
        ("add_bias", lambda: gpu._span("add_bias", [src, bias], rows, cols, [struct.pack("I", cols)])),
        ("layernorm", lambda: gpu.layernorm(src, out, rows, cols, 1e-6)),
        ("modulate", lambda: gpu.modulate(src, shift, scale, rows, cols, cols)),
        ("rmsnorm", lambda: gpu.rmsnorm(src, weight, out, rows, cols, 1e-6)),
        ("gate_residual", lambda: gpu.gate(src, other, gate, rows, cols, cols)),
        ("gelu_tanh", lambda: gpu.pointwise("gelu_tanh", wide, rows, 13824, 13824)),
    )
    for name, launch in jobs:
        ms = median_ms(gpu, launch)
        print(f"{name} {ms:.3f}", flush=True)
    rope_count = 2160 * 40 * 128
    values = gpu.empty(rope_count * 4)
    ms = median_ms(
        gpu,
        lambda: gpu.rope(values, 2160, 40, 128, (3, 40, 72), (44, 42, 42)),
    )
    print(f"rope3d {ms:.3f}", flush=True)


def _vae(gpu: Gpu) -> None:
    shapes = (
        (96, 5, 320, 576),
        (96, 6, 320, 576),
        (192, 6, 160, 288),
        (192, 5, 160, 288),
        (96, 1, 320, 576),
        (384, 3, 80, 144),
        (384, 4, 80, 144),
    )
    for cin, tin, hin, win in shapes:
        cout = cin
        source = gpu.empty(cin * tin * hin * win * 4)
        weight = gpu.empty(cout * cin * 27 * 4)
        bias = gpu.empty(cout * 4)

        def launch(
            source=source,
            weight=weight,
            bias=bias,
            cin=cin,
            tin=tin,
            hin=hin,
            win=win,
            cout=cout,
        ) -> None:
            dest, _shape = gpu.conv_into(
                source,
                (cin, tin, hin, win),
                weight,
                (cout, cin, 3, 3, 3),
                bias,
                (1, 1, 1),
                (2, 1, 1),
            )
            launch.dest = dest

        launch.dest = None
        launch()
        gpu.gpu_ms()
        gpu.release(launch.dest)
        samples = []
        for _ in range(5):
            launch()
            samples.append(gpu.gpu_ms())
            gpu.release(launch.dest)
        samples.sort()
        print(f"conv {cin}x{tin}x{hin}x{win} {samples[2]:.3f}", flush=True)
    channels, time, height, width = 384, 3, 40, 72
    plane = time * height * width
    qkv = gpu.empty(3 * channels * plane * 4)
    dest = gpu.empty(channels * plane * 4)
    ms = median_ms(
        gpu,
        lambda: gpu.spatial_attend(qkv, dest, channels, time, height, width),
    )
    print(f"spatial_attend {channels} {time}x{height}x{width} {ms:.3f}", flush=True)


def _text(gpu: Gpu) -> None:
    for rows, cols, inner in ((64, 4096, 4096), (64, 10240, 4096), (64, 4096, 10240)):
        act = gpu.empty(rows * inner * 2)
        weight = gpu.empty(cols * inner * 2)
        dest = gpu.empty(rows * cols * 4)
        ms = median_ms(
            gpu,
            lambda act=act, weight=weight, dest=dest, rows=rows, cols=cols, inner=inner: gpu.matmul_f16(
                act, weight, dest, rows, cols, inner
            ),
        )
        print(f"matmul_f16 {rows}x{cols} K{inner} {ms:.3f}", flush=True)
