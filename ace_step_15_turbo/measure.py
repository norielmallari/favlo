"""Shader checks and the isolated kernel ceiling for a 10 second clip.

The design point is 64 caption tokens, 32 lyric tokens, 750 timbre frames,
and 125 DiT tokens. Timing uses one command buffer per kernel. The first
sample is discarded. The recorded sample is the median of the next five.
"""

from __future__ import annotations

import math

import numpy as np

from ace_step_15_turbo.metal import Gpu
from ace_step_15_turbo.pack import FP4_MAGNITUDE, dequant_fp4, dequant_fp8, pack_fp4, pack_fp8


def run_kernels() -> None:
    gpu = Gpu()
    try:
        _check_matmul(gpu)
        _check_matmul_fp8(gpu)
        _check_rope(gpu)
        _check_euler(gpu)
        _check_snake(gpu)
        _check_conv(gpu)
        _check_attend(gpu)
        _check_adaln(gpu)
        print("checks passed", flush=True)
        _ceilings(gpu)
    finally:
        gpu.close()


def _median(gpu: Gpu, launch) -> float:
    launch()
    gpu.wait()
    samples = []
    for _ in range(5):
        before = gpu.seconds
        launch()
        gpu.wait()
        samples.append(gpu.seconds - before)
    samples.sort()
    return samples[2]


def _check_matmul(gpu: Gpu) -> None:
    rng = np.random.default_rng(1)
    rows, cols, inner = 64, 32, 256
    activation = rng.standard_normal((rows, inner)).astype(np.float32)
    weight = rng.standard_normal((cols, inner)).astype(np.float32)
    nibbles, scales, _, _ = pack_fp4(weight)
    reference_weight = dequant_fp4(nibbles, scales, cols, inner)
    reference = activation.astype(np.float16).astype(np.float32) @ reference_weight.T
    dest = gpu.matmul(gpu.f32(activation), gpu.upload(nibbles), gpu.upload(scales), rows, cols, inner)
    gpu.wait()
    got = gpu.read(dest, rows * cols, np.dtype(np.float32)).reshape(rows, cols)
    error = float(np.max(np.abs(got - reference)))
    if error > 5e-2:
        message = f"matmul error {error}"
        raise RuntimeError(message)


def _check_matmul_fp8(gpu: Gpu) -> None:
    rng = np.random.default_rng(7)
    rows, cols, inner = 64, 32, 256
    activation = rng.standard_normal((rows, inner)).astype(np.float32)
    weight = rng.standard_normal((cols, inner)).astype(np.float32)
    codes, scales, _, _ = pack_fp8(weight)
    reference_weight = dequant_fp8(codes, scales, cols, inner)
    reference = activation.astype(np.float16).astype(np.float32) @ reference_weight.T
    dest = gpu.matmul(
        gpu.f32(activation), gpu.upload(codes), gpu.upload(scales), rows, cols, inner, "fp8"
    )
    gpu.wait()
    got = gpu.read(dest, rows * cols, np.dtype(np.float32)).reshape(rows, cols)
    error = float(np.max(np.abs(got - reference)))
    if error > 5e-2:
        message = f"fp8 matmul error {error}"
        raise RuntimeError(message)


def _check_rope(gpu: Gpu) -> None:
    seq, heads, dim = 4, 2, 128
    values = np.linspace(-1.0, 1.0, seq * heads * dim, dtype=np.float32).reshape(seq, heads, dim)
    cos, sin = gpu.rope_table(seq, dim, 1_000_000.0)
    dest = gpu.rope(gpu.f32(values), cos, sin, seq, heads, dim)
    gpu.wait()
    got = gpu.read(dest, values.size, np.dtype(np.float32)).reshape(seq, heads, dim)
    half = dim // 2
    positions = np.arange(seq, dtype=np.float32)[:, None]
    index = np.arange(half, dtype=np.float32)[None, :]
    freq = np.exp(-math.log(1_000_000.0) * index / half)
    angle = positions * freq
    table_cos = np.concatenate([np.cos(angle), np.cos(angle)], axis=1)
    table_sin = np.concatenate([np.sin(angle), np.sin(angle)], axis=1)
    first, second = values[..., :half], values[..., half:]
    expect = np.empty_like(values)
    expect[..., :half] = first * table_cos[:, None, :half] - second * table_sin[:, None, :half]
    expect[..., half:] = second * table_cos[:, None, half:] + first * table_sin[:, None, half:]
    error = float(np.max(np.abs(got - expect)))
    if error > 1e-4:
        message = f"rope error {error}"
        raise RuntimeError(message)


def _check_adaln(gpu: Gpu) -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = np.array([0.5, -0.25], dtype=np.float32)
    shift = np.array([0.1, -0.2], dtype=np.float32)
    dest = gpu.adaln(gpu.f32(values), gpu.f32(scale), gpu.f32(shift), 2, 2)
    gpu.wait()
    got = gpu.read(dest, 4, np.dtype(np.float32)).reshape(2, 2)
    expect = values * (1.0 + scale) + shift
    if float(np.max(np.abs(got - expect))) > 1e-5:
        message = "adaln missed"
        raise RuntimeError(message)


def _check_euler(gpu: Gpu) -> None:
    latent = gpu.f32(np.ones(8, dtype=np.float32))
    velocity = gpu.f32(np.full(8, 2, dtype=np.float32))
    gpu.euler(latent, velocity, -0.5, 8)
    gpu.wait()
    got = gpu.read(latent, 8, np.dtype(np.float32))
    if float(np.max(np.abs(got - 0.0))) > 1e-5:
        message = "euler step missed"
        raise RuntimeError(message)


def _check_snake(gpu: Gpu) -> None:
    values = np.array([0.0, 0.5, -0.25, 1.0], dtype=np.float32)
    alpha = np.array([1.0, 2.0], dtype=np.float32)
    beta = np.array([1.0, 4.0], dtype=np.float32)
    dest = gpu.snake(gpu.f32(values.reshape(2, 2)), gpu.f32(np.log(alpha)), gpu.f32(np.log(beta)), 2, 2)
    gpu.wait()
    got = gpu.read(dest, 4, np.dtype(np.float32)).reshape(2, 2)
    raw = values.reshape(2, 2)
    sine = np.sin(alpha[:, None] * raw)
    expect = raw + (sine * sine) / beta[:, None]
    if float(np.max(np.abs(got - expect))) > 1e-5:
        message = "snake missed"
        raise RuntimeError(message)


def _check_conv(gpu: Gpu) -> None:
    cin, cout, kernel, length = 2, 2, 3, 5
    rng = np.random.default_rng(2)
    source = rng.standard_normal((cin, length)).astype(np.float32)
    weight = np.array([0.0, 0.5, 1.0, -0.5, 1.5, -1.0, 2.0, -1.5, 3.0, -2.0, 4.0, -3.0], dtype=np.float32)
    weight = weight.reshape(cout, cin * kernel)
    bias = np.array([0.25, -0.25], dtype=np.float32)
    codes, scales, _, inner = pack_fp8(weight)
    dest, out_len = gpu.conv(
        gpu.f32(source),
        gpu.upload(codes),
        gpu.upload(scales),
        gpu.f32(bias),
        cin,
        cout,
        kernel,
        inner,
        length,
        1,
        1,
        1,
    )
    gpu.wait()
    got = gpu.read(dest, cout * out_len, np.dtype(np.float32)).reshape(cout, out_len)
    restored = dequant_fp8(codes, scales, cout, cin * kernel).reshape(cout, cin, kernel)
    expect = np.zeros((cout, length), dtype=np.float32)
    for oc in range(cout):
        for pos in range(length):
            acc = bias[oc]
            for ic in range(cin):
                for tap in range(kernel):
                    src = pos + tap - 1
                    if 0 <= src < length:
                        acc += source[ic, src] * restored[oc, ic, tap]
            expect[oc, pos] = acc
    error = float(np.max(np.abs(got - expect)))
    if error > 1e-3:
        message = f"conv error {error}"
        raise RuntimeError(message)


def _check_attend(gpu: Gpu) -> None:
    seq, heads, kv_heads, dim = 3, 4, 2, 128
    rng = np.random.default_rng(3)
    query = rng.standard_normal((seq, heads, dim)).astype(np.float32)
    key = rng.standard_normal((seq, kv_heads, dim)).astype(np.float32) * 0.5
    value = rng.standard_normal((seq, kv_heads, dim)).astype(np.float32)
    coded_k, scale_k = gpu.quantize(gpu.f32(key), key.size)
    coded_v, scale_v = gpu.quantize(gpu.f32(value), value.size)
    dest = gpu.attend(
        gpu.f32(query),
        coded_k,
        scale_k,
        coded_v,
        scale_v,
        seq,
        seq,
        heads,
        kv_heads,
        dim,
        -1,
    )
    gpu.wait()
    got = gpu.read(dest, query.size, np.dtype(np.float32)).reshape(seq, heads, dim)
    key_q = _dequant_fp8(gpu.read(coded_k, key.size, np.dtype(np.uint8)), gpu.read(scale_k, key.size // 32, np.dtype(np.uint8)))
    value_q = _dequant_fp8(gpu.read(coded_v, value.size, np.dtype(np.uint8)), gpu.read(scale_v, value.size // 32, np.dtype(np.uint8)))
    key_q = key_q.reshape(seq, kv_heads, dim)
    value_q = value_q.reshape(seq, kv_heads, dim)
    expect = np.zeros_like(query)
    scale = dim ** -0.5
    for head in range(heads):
        group = head // (heads // kv_heads)
        scores = np.einsum("sd,td->st", query[:, head, :], key_q[:, group, :]) * scale
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        expect[:, head, :] = weights @ value_q[:, group, :]
    error = float(np.max(np.abs(got - expect)))
    if error > 2e-2:
        message = f"attend error {error}"
        raise RuntimeError(message)


def _dequant_fp8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    out = np.empty(codes.shape, dtype=np.float32)
    flat = codes.reshape(-1)
    scale = scales.reshape(-1)
    for index, code in enumerate(flat):
        value = int(code)
        sign = -1.0 if value & 0x80 else 1.0
        exponent = (value >> 3) & 0x0F
        mantissa = value & 0x07
        if exponent == 0:
            decoded = sign * (mantissa / 8.0) * 2.0 ** -6
        elif exponent == 15 and mantissa == 7:
            decoded = math.nan
        else:
            decoded = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        exponent_byte = int(scale[index // 32])
        factor = 0.0 if exponent_byte == 0 else math.ldexp(1.0, exponent_byte - 127)
        out.reshape(-1)[index] = decoded * factor
    return out


def _ceilings(gpu: Gpu) -> None:
    """Print the isolated sum. The numbers are this run, not a stored guess."""
    rng = np.random.default_rng(4)
    rows = []

    def time_matmul(name: str, count: int, m: int, n: int, k: int) -> None:
        padded = ((m + 63) // 64) * 64
        activation = rng.standard_normal((padded, k)).astype(np.float32)
        weight = rng.standard_normal((n, k)).astype(np.float32) * 0.01
        nibbles, scales, _, inner = pack_fp4(weight)
        act = gpu.f32(activation)
        packed = gpu.upload(nibbles)
        scale = gpu.upload(scales)
        seconds = _median(gpu, lambda: gpu.matmul(act, packed, scale, padded, n, inner))
        reads = padded * inner * 2 + (n * inner) // 2 + n * (inner // 32)
        writes = padded * n * 4
        rows.append((name, count, seconds, count * (reads + writes)))

    def time_attend(name: str, count: int, seq: int, kv: int, heads: int, kv_heads: int, window: int) -> None:
        query = gpu.f32(rng.standard_normal((seq, heads, 128)).astype(np.float32))
        key = gpu.upload(np.zeros(kv * kv_heads * 128, dtype=np.uint8))
        value = gpu.upload(np.zeros(kv * kv_heads * 128, dtype=np.uint8))
        key_scale = gpu.upload(np.zeros(kv * kv_heads * 4, dtype=np.uint8))
        value_scale = gpu.upload(np.zeros(kv * kv_heads * 4, dtype=np.uint8))
        seconds = _median(
            gpu,
            lambda: gpu.attend(query, key, key_scale, value, value_scale, seq, kv, heads, kv_heads, 128, window),
        )
        reads = seq * heads * 128 * 4 + 2 * kv * kv_heads * 128 + 2 * kv * kv_heads * 4
        writes = seq * heads * 128 * 4
        rows.append((name, count, seconds, count * (reads + writes)))

    time_matmul("text_q", 28, 64, 2048, 1024)
    time_matmul("text_kv", 56, 64, 1024, 1024)
    time_matmul("text_o", 28, 64, 1024, 2048)
    time_matmul("text_up", 56, 64, 3072, 1024)
    time_matmul("text_down", 28, 64, 1024, 3072)
    time_matmul("lyric_q", 8, 32, 2048, 2048)
    time_matmul("lyric_kv", 16, 32, 1024, 2048)
    time_matmul("lyric_o", 8, 32, 2048, 2048)
    time_matmul("lyric_up", 16, 32, 6144, 2048)
    time_matmul("lyric_down", 8, 32, 2048, 6144)
    time_matmul("lyric_embed", 1, 32, 2048, 1024)
    time_matmul("text_proj", 1, 64, 2048, 1024)
    time_matmul("timbre_embed", 1, 768, 2048, 256)
    time_matmul("timbre_q", 4, 768, 2048, 2048)
    time_matmul("timbre_kv", 8, 768, 1024, 2048)
    time_matmul("timbre_o", 4, 768, 2048, 2048)
    time_matmul("timbre_up", 8, 768, 6144, 2048)
    time_matmul("timbre_down", 4, 768, 2048, 6144)
    time_matmul("cond", 1, 128, 2560, 2048)
    time_matmul("dit_q", 256, 128, 4096, 2560)
    time_matmul("dit_kv", 512, 128, 1024, 2560)
    time_matmul("dit_cross_q", 256, 128, 4096, 2560)
    time_matmul("dit_cross_kv", 64, 128, 1024, 2560)
    time_matmul("dit_o", 512, 128, 2560, 4096)
    time_matmul("dit_up", 512, 128, 9728, 2560)
    time_matmul("dit_down", 256, 128, 2560, 9728)
    time_attend("text_attn", 28, 64, 64, 16, 8, -1)
    time_attend("lyric_attn", 8, 32, 32, 16, 8, 128)
    time_attend("timbre_attn", 4, 750, 750, 16, 8, 128)
    time_attend("dit_attn", 256, 125, 125, 32, 8, 128)
    time_attend("dit_cross", 256, 125, 97, 32, 8, -1)
    _time_vae(gpu, rows)
    copy_n = 512 * 1024 * 1024
    source = gpu.buffer(copy_n)
    dest = gpu.buffer(copy_n)
    copy_s = _median(gpu, lambda: gpu._encode("copy_bytes", (copy_n, 1, 1), (256, 1, 1), [source, dest], [copy_n], ["I"]))
    bandwidth = (2 * copy_n) / copy_s / 1e9
    kernel_s = sum(count * seconds for _name, count, seconds, _bytes in rows)
    moved = sum(payload for _name, _count, _seconds, payload in rows)
    print("design 10s caption 64 lyric 32 timbre 750 dit 125", flush=True)
    print(f"bandwidth_gbps {bandwidth:.3f} copy_bytes {copy_n}", flush=True)
    for name, count, seconds, payload in rows:
        print(f"{name} count {count} median_s {seconds:.6f} bytes {payload}", flush=True)
    print(f"kernel_ceiling_s {kernel_s:.6f}", flush=True)
    print(f"moved_bytes {moved}", flush=True)
    print(f"bandwidth_ceiling_s {moved / (bandwidth * 1e9):.6f}", flush=True)


def _time_vae(gpu: Gpu, rows: list[tuple[str, int, float, int]]) -> None:
    specs = [
        ("vae_conv1", 64, 2048, 7, 250, 3, 1, 1, False, 1),
        ("vae_up0", 2048, 1024, 20, 250, 5, 10, 1, True, 1),
        ("vae_res0", 1024, 1024, 7, 2500, 3, 1, 1, False, 3),
        ("vae_point0", 1024, 1024, 1, 2500, 0, 1, 1, False, 3),
        ("vae_up1", 1024, 512, 12, 2500, 3, 6, 1, True, 1),
        ("vae_res1", 512, 512, 7, 15000, 3, 1, 1, False, 3),
        ("vae_point1", 512, 512, 1, 15000, 0, 1, 1, False, 3),
        ("vae_up2", 512, 256, 8, 15000, 2, 4, 1, True, 1),
        ("vae_res2", 256, 256, 7, 60000, 3, 1, 1, False, 3),
        ("vae_point2", 256, 256, 1, 60000, 0, 1, 1, False, 3),
        ("vae_up3", 256, 128, 8, 60000, 2, 4, 1, True, 1),
        ("vae_res3", 128, 128, 7, 240000, 3, 1, 1, False, 3),
        ("vae_point3", 128, 128, 1, 240000, 0, 1, 1, False, 3),
        ("vae_up4", 128, 128, 4, 240000, 1, 2, 1, True, 1),
        ("vae_res4", 128, 128, 7, 480000, 3, 1, 1, False, 3),
        ("vae_point4", 128, 128, 1, 480000, 0, 1, 1, False, 3),
        ("vae_conv2", 128, 2, 7, 480000, 3, 1, 1, False, 1),
    ]
    for name, cin, cout, kernel, length, pad, stride, dilation, transpose, count in specs:
        if transpose:
            weight = np.zeros((cin, cout * kernel), dtype=np.float32)
        else:
            weight = np.zeros((cout, cin * kernel), dtype=np.float32)
        nibbles, scales, _, inner = pack_fp4(weight)
        source_len = length
        values = gpu.f32(np.zeros((cin, source_len), dtype=np.float32))
        packed = gpu.upload(nibbles)
        scale = gpu.upload(scales)

        def launch(
            values=values,
            packed=packed,
            scale=scale,
            cin=cin,
            cout=cout,
            kernel=kernel,
            inner=inner,
            length=length,
            pad=pad,
            stride=stride,
            dilation=dilation,
            transpose=transpose,
        ) -> None:
            if transpose:
                gpu.conv_transpose(values, packed, scale, None, cin, cout, kernel, inner, length, pad, stride)
            else:
                gpu.conv(values, packed, scale, None, cin, cout, kernel, inner, length, pad, stride, dilation)

        seconds = _median(gpu, launch)
        rows.append((name, count, seconds, count * cin * length * 4))
