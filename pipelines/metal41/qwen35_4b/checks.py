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
        _check_rmsnorm(session)
        _check_gdn(session)
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
