"""Shader library acceptance for ``favlo --qwen35-4b kernels``."""

from __future__ import annotations

import math
import struct

import numpy as np

from favlo.memory import memory
from qwen35_4b.formats import quantize_fp8
from qwen35_4b.session import GEMV_ROWS, Session, f32, u32


def run_kernels() -> None:
    owner = "qwen35-4b:kernels"
    session = Session(owner)
    try:
        _check_gemv(session)
        _check_int8(session)
        _check_rmsnorm(session)
        _check_gdn(session)
        _check_kv_half(session)
    finally:
        session.close()
        memory.sweep(owner)


def _check_gemv(session: Session) -> None:
    rows, inner = 64, 512
    rng = np.random.default_rng(0)
    activation = rng.standard_normal(inner).astype(np.float32)
    weight = rng.standard_normal((rows, inner)).astype(np.float32) * 0.02
    data, scale = quantize_fp8(weight)
    data_u = np.frombuffer(data, dtype=np.uint8).reshape(rows, inner)
    scale_u = np.frombuffer(scale, dtype=np.uint8).reshape(rows, inner // 32)
    recon = np.empty_like(weight)
    for r in range(rows):
        for b in range(inner // 32):
            s = _e8m0(int(scale_u[r, b]))
            for i in range(32):
                recon[r, b * 32 + i] = _e4m3(int(data_u[r, b * 32 + i])) * s
    expect = recon @ activation

    act = session.upload(activation.tobytes())
    w = session.upload(data)
    sc = session.upload(scale)
    out = session.allocate(rows * 4, zero=True)
    session.launch(
        "gemv_fp8",
        [act, w, sc, out],
        [u32(rows), u32(inner)],
        ((rows + GEMV_ROWS - 1) // GEMV_ROWS, 1, 1),
        (256, 1, 1),
    )
    got = np.frombuffer(memory.load(out, 0, rows * 4), dtype=np.float32)
    if not np.allclose(got, expect, rtol=2e-2, atol=2e-2):
        message = f"gemv_fp8 mismatch max={np.max(np.abs(got - expect))}"
        raise RuntimeError(message)

    out_v = session.allocate(rows * 4, zero=True)
    session.launch(
        "gemv_fp8_vec",
        [act, w, sc, out_v],
        [u32(rows), u32(inner)],
        (rows, 1, 1),
        (32, 1, 1),
    )
    got_v = np.frombuffer(memory.load(out_v, 0, rows * 4), dtype=np.float32)
    if not np.allclose(got_v, expect, rtol=2e-2, atol=2e-2):
        message = f"gemv_fp8_vec mismatch max={np.max(np.abs(got_v - expect))}"
        raise RuntimeError(message)


def _check_int8(session: Session) -> None:
    """fp8 -> int8 restore, int8 activation, int8 matmul2d (both tiles), then scale."""
    rows, cols, inner = 64, 128, 512
    rng = np.random.default_rng(4)
    activation = rng.standard_normal((rows, inner)).astype(np.float32)
    activation[rows - 8 :] = 0.0  # padded rows quantize to zero
    activation = activation.astype(np.float16)
    weight = rng.standard_normal((cols, inner)).astype(np.float32) * 0.02
    data, scale = quantize_fp8(weight)
    data_u = np.frombuffer(data, dtype=np.uint8).reshape(cols, inner)
    scale_u = np.frombuffer(scale, dtype=np.uint8).reshape(cols, inner // 32)
    recon = np.empty_like(weight)
    for c in range(cols):
        for b in range(inner // 32):
            s = _e8m0(int(scale_u[c, b]))
            for i in range(32):
                recon[c, b * 32 + i] = _e4m3(int(data_u[c, b * 32 + i])) * s
    expect = activation.astype(np.float32) @ recon.T

    act = session.upload(activation.tobytes())
    w = session.upload(data)
    sc = session.upload(scale)
    w_i8 = session.allocate(cols * inner, zero=True)
    w_scale = session.allocate(cols * 4, zero=True)
    a_i8 = session.allocate(rows * inner, zero=True)
    a_scale = session.allocate(rows * 4, zero=True)
    session.launch(
        "restore_fp8_i8",
        [w, sc, w_i8, w_scale],
        [u32(inner)],
        (cols, 1, 1),
        (256, 1, 1),
    )
    session.launch(
        "quantize_half_i8",
        [act, a_i8, a_scale],
        [u32(inner)],
        (rows, 1, 1),
        (256, 1, 1),
    )
    codes = np.frombuffer(memory.load(w_i8, 0, cols * inner), dtype=np.int8).reshape(cols, inner)
    channel = np.frombuffer(memory.load(w_scale, 0, cols * 4), dtype=np.float32)
    if int(np.abs(codes).max()) != 127 or np.any(channel <= 0.0):
        message = "restore_fp8_i8 did not fill the int8 range"
        raise RuntimeError(message)
    roundtrip = codes.astype(np.float32) * channel[:, None]
    if not np.allclose(roundtrip, recon, rtol=0.0, atol=float(channel.max())):
        message = f"restore_fp8_i8 roundtrip max={np.max(np.abs(roundtrip - recon))}"
        raise RuntimeError(message)
    act_scale = np.frombuffer(memory.load(a_scale, 0, rows * 4), dtype=np.float32)
    if np.any(act_scale[rows - 8 :] != 0.0) or np.any(act_scale[: rows - 8] <= 0.0):
        message = "quantize_half_i8 row scales are wrong"
        raise RuntimeError(message)

    # Both tiles. 1% of the peak covers the two int8 roundings.
    limit = 0.02 * float(np.max(np.abs(expect)))
    for kernel, groups in (
        ("prefill_i8_n64", (cols // 64, rows // 64, 1)),
        ("prefill_i8_m32", (cols // 128, rows // 32, 1)),
    ):
        out = session.allocate(rows * cols * 4, zero=True)
        session.launch(
            kernel,
            [a_i8, w_i8, out],
            [u32(rows), u32(cols), u32(inner)],
            groups,
            (128, 1, 1),
        )
        session.launch(
            "scale_i8_f32",
            [out, a_scale, w_scale],
            [u32(rows), u32(cols)],
            ((rows * (cols // 4) + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        got = np.frombuffer(memory.load(out, 0, rows * cols * 4), dtype=np.float32)
        got = got.reshape(rows, cols)
        error = float(np.max(np.abs(got - expect)))
        if not np.isfinite(error) or error > limit:
            message = f"{kernel} int8 mismatch max={error} limit={limit}"
            raise RuntimeError(message)


def _check_rmsnorm(session: Session) -> None:
    width = 256
    rng = np.random.default_rng(1)
    x = rng.standard_normal(width).astype(np.float32)
    w = rng.standard_normal(width).astype(np.float16)
    eps = 1e-6
    inv = 1.0 / math.sqrt(float(np.mean(x * x)) + eps)
    expect = x * inv * (w.astype(np.float32) + 1.0)

    xin = session.upload(x.tobytes())
    win = session.upload(w.tobytes())
    out = session.allocate(width * 4, zero=True)
    session.launch(
        "rmsnorm",
        [xin, win, out],
        [u32(width), f32(eps), f32(1.0)],
        (1, 1, 1),
        (256, 1, 1),
    )
    got = np.frombuffer(memory.load(out, 0, width * 4), dtype=np.float32)
    if not np.allclose(got, expect, rtol=1e-4, atol=1e-4):
        message = f"rmsnorm mismatch max={np.max(np.abs(got - expect))}"
        raise RuntimeError(message)


def _check_gdn(session: Session) -> None:
    key_heads, value_heads, dim = 1, 2, 128
    rng = np.random.default_rng(2)
    # mixed: q (key_heads*D) | k (key_heads*D) | v (value_heads*D)
    mixed = rng.standard_normal(key_heads * dim * 2 + value_heads * dim).astype(np.float32)
    proj_a = rng.standard_normal(value_heads).astype(np.float32)
    proj_b = rng.standard_normal(value_heads).astype(np.float32)
    a_log = rng.uniform(0.01, 2.0, size=value_heads).astype(np.float16)
    dt = rng.standard_normal(value_heads).astype(np.float16)
    state0 = np.zeros((value_heads, dim, dim), dtype=np.float32)

    expect = np.zeros((value_heads, dim), dtype=np.float32)
    state = state0.copy()
    for h in range(value_heads):
        kh = h // (value_heads // key_heads)
        q = mixed[kh * dim : (kh + 1) * dim].copy()
        k = mixed[key_heads * dim + kh * dim : key_heads * dim + (kh + 1) * dim].copy()
        v = mixed[2 * key_heads * dim + h * dim : 2 * key_heads * dim + (h + 1) * dim]
        q = q / math.sqrt(float(np.sum(q * q)) + 1e-6) / math.sqrt(dim)
        k = k / math.sqrt(float(np.sum(k * k)) + 1e-6)
        beta = 1.0 / (1.0 + math.exp(-float(proj_b[h])))
        x = float(proj_a[h]) + float(dt[h])
        soft = x if x > 20 else math.log1p(math.exp(x))
        alpha = math.exp(-math.exp(float(a_log[h])) * soft)
        S = state[h]  # key-major S[c, r]
        sk = S.T @ k  # sk[r] = sum_c S[c,r]*k[c]
        delta = beta * (v - alpha * sk)
        S = alpha * S + np.outer(k, delta)
        state[h] = S
        expect[h] = S.T @ q

    st = session.upload(state0.tobytes())
    mx = session.upload(mixed.tobytes())
    pa = session.upload(proj_a.tobytes())
    pb = session.upload(proj_b.tobytes())
    al = session.upload(a_log.tobytes())
    db = session.upload(dt.tobytes())
    out = session.allocate(value_heads * dim * 4, zero=True)
    session.launch(
        "gdn_step",
        [st, mx, pa, pb, al, db, out],
        [u32(key_heads), u32(value_heads)],
        (value_heads, 1, 1),
        (128, 1, 1),
    )
    got = np.frombuffer(memory.load(out, 0, value_heads * dim * 4), dtype=np.float32).reshape(
        value_heads, dim
    )
    if not np.allclose(got, expect, rtol=1e-3, atol=1e-3):
        message = f"gdn_step mismatch max={np.max(np.abs(got - expect))}"
        raise RuntimeError(message)


def _check_kv_half(session: Session) -> None:
    """One causal prefill tile and a one-token decode, both on a float16 cache."""
    heads, kv, dim, rows = 16, 4, 256, 2
    group = heads // kv
    scale = dim**-0.5
    rng = np.random.default_rng(3)
    query = rng.standard_normal((rows, heads, dim * 2)).astype(np.float32) * 0.05
    key = rng.standard_normal((rows, kv, dim)).astype(np.float32) * 0.05
    value = rng.standard_normal((rows, kv, dim)).astype(np.float32) * 0.05
    q_buf = session.upload(np.ascontiguousarray(query).tobytes())
    k_buf = session.upload(np.ascontiguousarray(key).tobytes())
    v_buf = session.upload(np.ascontiguousarray(value).tobytes())
    k_data = session.allocate(rows * kv * dim * 2, zero=True)
    v_data = session.allocate(rows * kv * dim * 2, zero=True)
    scores = session.allocate(rows * heads * rows * 4, zero=True)
    out = session.allocate(rows * heads * dim * 4, zero=True)
    for plane in (0, 1):
        session.launch(
            "store_kv_rows_half",
            [k_buf, v_buf, k_data, v_data],
            [u32(0), u32(kv), u32(rows), u32(plane)],
            (kv, rows, 1),
            (dim, 1, 1),
        )
    session.launch(
        "attend_prefill_half",
        [q_buf, k_data, v_data, scores, out],
        [u32(0), u32(rows), u32(rows), u32(heads), u32(kv), u32(group), f32(scale)],
        (heads, rows, 1),
        (dim, 1, 1),
    )
    got = np.frombuffer(memory.load(out, 0, rows * heads * dim * 4), dtype=np.float32)
    got = got.reshape(rows, heads, dim)
    expect = _kv_half_reference(query, key, value, scale, group)
    if not np.allclose(got, expect, rtol=2e-3, atol=2e-3):
        message = f"attend_prefill_half mismatch max={np.max(np.abs(got - expect))}"
        raise RuntimeError(message)

    q0 = session.upload(np.ascontiguousarray(query[0]).tobytes())
    scores0 = session.allocate(heads * 4, zero=True)
    out0 = session.allocate(heads * dim * 4, zero=True)
    session.launch(
        "attend_decode_half",
        [q0, k_data, v_data, scores0, out0],
        [u32(1), u32(1), u32(kv), u32(group), f32(scale)],
        (heads, 1, 1),
        (dim, 1, 1),
    )
    got0 = np.frombuffer(memory.load(out0, 0, heads * dim * 4), dtype=np.float32)
    if not np.allclose(got0.reshape(heads, dim), expect[0], rtol=2e-3, atol=2e-3):
        message = f"attend_decode_half mismatch max={np.max(np.abs(got0 - expect[0].ravel()))}"
        raise RuntimeError(message)


def _kv_half_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scale: float,
    group: int,
) -> np.ndarray:
    rows, heads, _ = query.shape
    dim = key.shape[-1]
    key_h = key.astype(np.float16).astype(np.float32)
    value_h = value.astype(np.float16).astype(np.float32)
    expect = np.empty((rows, heads, dim), dtype=np.float32)
    for row in range(rows):
        length = row + 1
        for head in range(heads):
            q = query[row, head, :dim]
            gate = query[row, head, dim:].astype(np.float64)
            kvh = head // group
            dots = np.empty(length, dtype=np.float64)
            for position in range(length):
                dots[position] = float(np.dot(q, key_h[position, kvh])) * scale
            dots -= dots.max()
            weight = np.exp(dots)
            weight /= weight.sum()
            acc = np.zeros(dim, dtype=np.float64)
            for position in range(length):
                acc += weight[position] * value_h[position, kvh]
            sig = 1.0 / (1.0 + np.exp(-gate))
            expect[row, head] = (acc * sig).astype(np.float32)
    return expect


def _e8m0(code: int) -> float:
    if code == 0:
        return float(np.frombuffer(struct.pack("<I", 1 << 22), dtype=np.float32)[0])
    return float(np.frombuffer(struct.pack("<I", code << 23), dtype=np.float32)[0])


def _e4m3(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0xF
    mant = code & 7
    if exp == 0:
        if mant == 0:
            return 0.0
        return sign * (mant / 8.0) * (2.0 ** (1 - 7))
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
