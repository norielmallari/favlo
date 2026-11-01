"""Shader checks on synthetic tiles. NumPy stays in this module."""

from __future__ import annotations

import math
import struct

import numpy as np

from favlo.memory import memory
from spark_x26_1_7b.formats import dequant_fp8, quantize_fp8
from spark_x26_1_7b.sample import Sampler
from spark_x26_1_7b.session import Session, f32, u32

_OWNER = "spark-x26-1-7b-kernels"


def run() -> None:
    session = Session(_OWNER)
    try:
        _gemv(session)
        _gemv_vec_rms(session)
        _gemv_vec_add_cached(session)
        _matmul(session, 64, 64, 256, "matmul_fp8_64", 64, 64)
        _matmul(session, 32, 128, 256, "matmul_fp8_32", 32, 128)
        _rmsnorm(session)
        _rope(session)
        _geglu(session)
        _gate(session)
        _argmax(session)
        _nucleus(session)
        _e4m3(session)
        _f16(session)
    finally:
        session.close()


def _gemv(session: Session) -> None:
    weight = _exact((4, 256))
    packed, scales = quantize_fp8(weight)
    activation = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    data = session.upload(packed)
    scale = session.upload(scales)
    act = session.upload(activation.tobytes())
    dest = session.allocate(16)
    session.launch(
        "gemv_fp8",
        [act, data, scale, dest],
        [u32(256)],
        (1, 1, 1),
        (32, 1, 1),
    )
    got = _f32(dest, 4)
    expect = weight @ activation
    _close("gemv", got, expect)


def _gemv_vec_rms(session: Session) -> None:
    rows, inner = 32, 256
    weight = _exact((rows, inner))
    packed, scales = quantize_fp8(weight)
    activation = np.linspace(-1.0, 1.0, inner, dtype=np.float32)
    gain = np.ones(inner, dtype=np.float16)
    data = session.upload(packed)
    scale = session.upload(scales)
    act = session.upload(activation.tobytes())
    norm_w = session.upload(gain.tobytes())
    dest = session.allocate(rows * 4)
    normed = session.allocate(inner * 4)
    session.launch(
        "gemv_fp8_vec_rms",
        [act, norm_w, data, scale, dest, normed],
        [u32(inner), f32(1e-6)],
        (1, 1, 1),
        (256, 1, 1),
    )
    inv = 1.0 / np.sqrt(np.mean(activation * activation) + 1e-6)
    restored = dequant_fp8(packed, scales, rows, inner)
    expect = restored @ (activation * inv)
    _close("gemv_fp8_vec_rms", _f32(dest, rows), expect, atol=2e-2)


def _gemv_vec_add_cached(session: Session) -> None:
    rows, inner = 32, 256
    weight = _exact((rows, inner))
    packed, scales = quantize_fp8(weight)
    activation = np.linspace(-0.5, 0.5, inner, dtype=np.float32)
    hidden = np.ones(rows, dtype=np.float32)
    data = session.upload(packed)
    scale = session.upload(scales)
    act = session.upload(activation.tobytes())
    dest = session.upload(hidden.tobytes())
    session.launch(
        "gemv_fp8_vec_add_cached",
        [act, data, scale, dest],
        [u32(inner)],
        (1, 1, 1),
        (256, 1, 1),
    )
    restored = dequant_fp8(packed, scales, rows, inner)
    expect = hidden + restored @ activation
    _close("gemv_fp8_vec_add_cached", _f32(dest, rows), expect, atol=2e-2)


def _matmul(
    session: Session,
    rows: int,
    cols: int,
    inner: int,
    name: str,
    tile_m: int,
    tile_n: int,
) -> None:
    weight = _exact((cols, inner))
    packed, scales = quantize_fp8(weight)
    left = np.ones((rows, inner), dtype=np.float32) * 0.5
    half = np.ascontiguousarray(left, dtype=np.float16)
    act = session.upload(half.tobytes())
    data = session.upload(packed)
    scale = session.upload(scales)
    dest = session.allocate(rows * cols * 4)
    session.launch(
        name,
        [act, data, scale, dest],
        [u32(rows), u32(cols), u32(inner)],
        (cols // tile_n, rows // tile_m, 1),
        (128, 1, 1),
    )
    got = _f32(dest, rows * cols).reshape(rows, cols)
    restored = dequant_fp8(packed, scales, cols, inner)
    expect = left @ restored.T
    _close(name, got, expect, atol=2e-2)


def _rmsnorm(session: Session) -> None:
    source = np.arange(8, dtype=np.float32)
    # The kernel walks n with a 32-lane group. Use 32 channels and a short prefix.
    values = np.zeros(32, dtype=np.float32)
    values[:8] = source
    weight = np.ones(32, dtype=np.float32)
    src = session.upload(values.tobytes())
    gain = session.upload(weight.astype(np.float16).tobytes())
    dest = session.allocate(32 * 4)
    session.launch(
        "rmsnorm",
        [src, gain, dest],
        [u32(32), f32(1e-6)],
        (1, 1, 1),
        (32, 1, 1),
    )
    got = _f32(dest, 32)
    inv = 1.0 / np.sqrt(np.mean(values * values) + 1e-6)
    expect = values * inv
    _close("rmsnorm", got, expect, atol=1e-4)


def _rope(session: Session) -> None:
    row = np.arange(_QKV, dtype=np.float32) * 0.001
    handle = session.upload(row.tobytes())
    session.launch(
        "rope_qk",
        [handle],
        [struct.pack("<IIf", 0, 64, 5_000_000.0)],
        (10, 1, 1),
        (128, 1, 1),
    )
    got = _f32(handle, _QKV)
    _close("rope", got, row, atol=1e-5)


def _geglu(session: Session) -> None:
    gate = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    up = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gate_h = session.upload(gate.tobytes())
    up_h = session.upload(up.tobytes())
    dest = session.allocate(16)
    session.launch("geglu", [gate_h, up_h, dest], [u32(4)], (1, 1, 1), (32, 1, 1))
    got = _f32(dest, 4)
    erf = np.array([math.erf(float(value) / math.sqrt(2.0)) for value in gate], dtype=np.float32)
    gelu = 0.5 * gate * (1.0 + erf)
    _close("geglu", got, gelu * up, atol=1e-5)


def _gate(session: Session) -> None:
    mixed = np.ones(8 * 256, dtype=np.float32)
    gates = np.zeros(8, dtype=np.float32)
    gates[0] = 0.0
    mixed_h = session.upload(mixed.tobytes())
    gate_h = session.upload(gates.tobytes())
    session.launch(
        "gate_rows",
        [mixed_h, gate_h],
        [],
        (1, 8, 1),
        (32, 1, 1),
    )
    got = _f32(mixed_h, 8 * 256)
    expect = mixed.copy()
    for head in range(8):
        scale = 1.0 / (1.0 + np.exp(-gates[head]))
        expect[head * 256 : (head + 1) * 256] *= scale
    _close("gate", got, expect, atol=1e-5)


def _argmax(session: Session) -> None:
    sampler = Sampler(session)
    logits = np.zeros(131072, dtype=np.float32)
    logits[7] = 4.0
    logits[3] = 4.0
    handle = session.upload(logits.tobytes())
    chosen = sampler.draw(handle, temperature=0.0, top_p=1.0, top_k=0, presence=0.0, frequency=0.0)
    if chosen != 3:
        message = f"argmax kept {chosen}, expected the lower index 3"
        raise RuntimeError(message)


def _nucleus(session: Session) -> None:
    sampler = Sampler(session)
    sampler.reseed(1)
    logits = np.full(131072, -80.0, dtype=np.float32)
    logits[11] = 40.0
    handle = session.upload(logits.tobytes())
    chosen = sampler.draw(handle, temperature=1.0, top_p=0.95, top_k=0, presence=0.0, frequency=0.0)
    if chosen != 11:
        message = f"nucleus kept {chosen}, expected 11"
        raise RuntimeError(message)


def _e4m3(session: Session) -> None:
    row = np.zeros(_QKV, dtype=np.float32)
    row[2048:2048 + 32] = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    row[2560:2560 + 32] = np.linspace(0.25, 1.0, 32, dtype=np.float32)
    handle = session.upload(row.tobytes())
    key = session.allocate(2 * 32 * 256, zero=True)
    key_scale = session.allocate(2 * 32 * 8, zero=True)
    value = session.allocate(2 * 256 * 32, zero=True)
    value_scale = session.allocate(2 * 256 * 1, zero=True)
    scratch = session.allocate(2 * 256 * 32 * 4, zero=True)
    session.launch(
        "store_full",
        [handle, key, key_scale, value, value_scale, scratch],
        [struct.pack("<II", 0, 32)],
        (1, 1, 1),
        (32, 1, 1),
    )
    scores = session.allocate(8 * 32 * 4, zero=True)
    queries = np.zeros(_QKV, dtype=np.float32)
    queries[:32] = 1.0
    query = session.upload(queries.tobytes())
    session.launch(
        "score_keys",
        [query, key, key_scale, scores],
        [struct.pack("<IIII", 0, 32, 0, 32)],
        (8, 1, 1),
        (32, 1, 1),
    )
    got = _f32(scores, 8)
    if not np.isfinite(got).all():
        message = "e4m3 score was not finite"
        raise RuntimeError(message)
    # 3.0 sits at unit 384 under the block scale. That code is exponent 15
    # mantissa 4, which decodes to 3.0. Saturating the bin decodes to 3.5.
    peak = np.zeros(_QKV, dtype=np.float32)
    peak[2048] = 3.0
    stored = session.upload(peak.tobytes())
    session.launch(
        "store_full",
        [stored, key, key_scale, value, value_scale, scratch],
        [struct.pack("<II", 0, 32)],
        (1, 1, 1),
        (32, 1, 1),
    )
    code = memory.load(key, 0, 1)[0]
    if code != 0x7C:
        message = f"e4m3 stored {code}, expected 124 for the 384 bin"
        raise RuntimeError(message)


def _f16(session: Session) -> None:
    row = np.zeros(_QKV, dtype=np.float32)
    row[2048] = 1.5
    row[2560] = 2.0
    handle = session.upload(row.tobytes())
    key = session.allocate(2 * 32 * 256 * 2, zero=True)
    value = session.allocate(2 * 256 * 32 * 2, zero=True)
    session.launch(
        "store_full_span_f16",
        [handle, key, value],
        [struct.pack("<III", 0, 32, 1)],
        (16, 1, 1),
        (32, 1, 1),
    )
    stored = struct.unpack("<e", memory.load(key, 0, 2))[0]
    if abs(stored - 1.5) > 1e-3:
        message = f"f16 key stored {stored}, expected 1.5"
        raise RuntimeError(message)
    queries = np.zeros(_QKV, dtype=np.float32)
    queries[0] = 1.0
    query = session.upload(queries.tobytes())
    gates = session.upload(np.zeros(8, dtype=np.float32).tobytes())
    mixed = session.allocate(8 * 256 * 4, zero=True)
    session.launch(
        "attend_span_f16",
        [query, key, value, gates, mixed],
        [struct.pack("<IIIII", 0, 32, 0, 32, 1)],
        (8, 1, 1),
        (32, 1, 1),
    )
    got = struct.unpack_from("<f", memory.load(mixed, 0, 4))[0]
    if not math.isfinite(got):
        message = "f16 attend was not finite"
        raise RuntimeError(message)


def _exact(shape: tuple[int, int]) -> np.ndarray:
    table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    count = int(np.prod(shape))
    values = table[np.arange(count) % table.size]
    signs = np.where(np.arange(count) % 2, -1.0, 1.0).astype(np.float32)
    return (values * signs).reshape(shape)


def _f32(handle: object, count: int) -> np.ndarray:
    raw = memory.load(handle, 0, count * 4)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _close(name: str, got: np.ndarray, expect: np.ndarray, atol: float = 1e-4) -> None:
    if got.shape != expect.shape or not np.allclose(got, expect, rtol=1e-3, atol=atol):
        delta = float(np.max(np.abs(got - expect))) if got.shape == expect.shape else -1.0
        message = f"{name} missed the reference by {delta}"
        raise RuntimeError(message)


_QKV = 3072
