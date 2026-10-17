"""Host formulas and the dispatches that have to match them."""

from __future__ import annotations

import numpy as np

from qwen38_27b.formats import dequant_e4m3_block, dequant_fp4, quantize_fp4
from qwen38_27b.session import Session

EPS = np.float32(1e-6)


def _close(
    name: str,
    got: np.ndarray,
    expect: np.ndarray,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    if got.shape != expect.shape:
        message = f"{name} shape {got.shape} != {expect.shape}"
        raise AssertionError(message)
    if not np.allclose(got, expect, rtol=rtol, atol=atol, equal_nan=True):
        delta = float(np.max(np.abs(got.astype(np.float64) - expect.astype(np.float64))))
        message = f"{name} max abs {delta}"
        raise AssertionError(message)


def _probe_fp4(rows: int, cols: int, inner: int) -> tuple[np.ndarray, bytes, bytes, np.ndarray]:
    activation = np.zeros((rows, inner), dtype=np.float16)
    activation[:, 0] = np.float16(1.0)
    codes = np.zeros((cols, inner), dtype=np.uint8)
    codes[:, 0] = np.arange(cols, dtype=np.uint8) % 8
    packed = bytearray()
    flat = codes.reshape(-1)
    for index in range(0, flat.size, 2):
        packed.append(int(flat[index] & 0xF) | (int(flat[index + 1] & 0xF) << 4))
    blocks = inner // 32
    scales = bytearray()
    for _row in range(cols):
        scales.append(130)
        scales.extend(b"\x00" * (blocks - 1))
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    column = magnitudes[np.arange(cols) % 8] * np.float32(8.0)
    expected = np.broadcast_to(column, (rows, cols)).copy()
    return activation, bytes(packed), bytes(scales), expected


def check_fp4_tile(session: Session) -> None:
    activation, data, scales, expected = _probe_fp4(64, 64, 256)
    got = session.prefill("prefill_fp4_n64", activation, data, scales, 64, 64, 256)
    _close("fp4 tile", got, expected, rtol=0.0, atol=0.0)
    activation, data, scales, expected = _probe_fp4(64, 32, 32)
    got = session.prefill("prefill_fp4_e8m0", activation, data, scales, 64, 32, 32)
    _close("fp4 32-col", got, expected, rtol=0.0, atol=0.0)


def check_fp4_gemv(session: Session) -> None:
    activation, data, scales, expected = _probe_fp4(1, 32, 256)
    got = session.gemv_fp4(activation.reshape(-1), data, scales, 32, 256)
    _close("fp4 gemv", got, expected.reshape(-1), rtol=0.0, atol=0.0)


def check_fp4_x4(session: Session) -> None:
    rng = np.random.default_rng(0)
    cols, inner = 8, 256
    activation = rng.normal(scale=0.05, size=inner).astype(np.float32)
    weight = rng.normal(scale=0.05, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_fp4(weight)
    destination = session.zeros(cols * 4)
    session.dispatch(
        "gemv_fp4_x4",
        [
            session.upload(activation),
            session.upload(data),
            session.upload(scales),
            destination,
        ],
        [np.uint32(cols).tobytes(), np.uint32(inner).tobytes()],
        (cols // 4, 1, 1),
        (32, 1, 1),
    )
    expect = dequant_fp4(data, scales, cols, inner) @ activation
    _close("fp4 x4", session.download_f32(destination, cols), expect.astype(np.float32), atol=1e-4)


def check_fp4_span2(session: Session) -> None:
    rng = np.random.default_rng(1)
    cols, inner = 8, 256
    act0 = rng.normal(scale=0.05, size=inner).astype(np.float32)
    act1 = rng.normal(scale=0.05, size=inner).astype(np.float32)
    weight = rng.normal(scale=0.05, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_fp4(weight)
    dest0 = session.zeros(cols * 4)
    dest1 = session.zeros(cols * 4)
    session.dispatch(
        "gemv_fp4_span2",
        [
            session.upload(act0),
            session.upload(act1),
            session.upload(data),
            session.upload(scales),
            dest0,
            dest1,
        ],
        [np.uint32(cols).tobytes(), np.uint32(inner).tobytes()],
        (cols // 4, 1, 1),
        (32, 1, 1),
    )
    matrix = dequant_fp4(data, scales, cols, inner)
    _close("fp4 span2 a", session.download_f32(dest0, cols), (matrix @ act0).astype(np.float32), atol=1e-4)
    _close("fp4 span2 b", session.download_f32(dest1, cols), (matrix @ act1).astype(np.float32), atol=1e-4)


def check_rms(session: Session) -> None:
    width = 5120
    values = np.ones(width, dtype=np.float32)
    weight = np.full(width, 2.0, dtype=np.float16)
    destination = session.zeros(width * 4)
    session.dispatch(
        "rmsnorm_f32",
        [session.upload(values), session.upload(weight), destination],
        [np.uint32(width).tobytes(), np.float32(EPS).tobytes()],
        (1, 1, 1),
        (256, 1, 1),
    )
    inv = np.float32(1.0) / np.sqrt(np.mean(values * values) + EPS)
    _close("rms", session.download_f32(destination, width), values * inv * np.float32(3.0), atol=1e-4)


def check_swiglu(session: Session) -> None:
    gate = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    up = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    destination = session.zeros(16)
    session.dispatch(
        "swiglu_f32",
        [session.upload(gate), session.upload(up), destination],
        [],
        (4, 1, 1),
        (1, 1, 1),
    )
    silu = gate / (1.0 + np.exp(-gate))
    _close("swiglu", session.download_f32(destination, 4), (silu * up).astype(np.float32), atol=1e-5)


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.where(value > 20.0, value, np.log1p(np.exp(value))).astype(np.float32)


def check_deltanet(session: Session) -> None:
    mixed = np.ones(10240, dtype=np.float32)
    raw_a = np.zeros(64, dtype=np.float32)
    raw_b = np.zeros(64, dtype=np.float32)
    gate = np.ones(6144, dtype=np.float32)
    a_log = np.zeros(48, dtype=np.float16)
    dt_bias = np.zeros(48, dtype=np.float16)
    norm = np.ones(128, dtype=np.float16)
    state0 = np.ones((48, 128, 128), dtype=np.float32)
    destination = session.zeros(6144 * 4)
    state = session.upload(state0)
    session.dispatch(
        "gdn_state_rows",
        [
            state,
            session.upload(mixed),
            session.upload(raw_a),
            session.upload(raw_b),
            session.upload(gate),
            session.upload(a_log),
            session.upload(dt_bias),
            session.upload(norm),
            destination,
        ],
        [np.float32(EPS).tobytes(), np.uint32(1).tobytes()],
        (48, 1, 1),
        (128, 1, 1),
    )
    q = np.ones((16, 128), dtype=np.float32)
    k = np.ones((16, 128), dtype=np.float32)
    v = np.ones((48, 128), dtype=np.float32)
    qn = q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + EPS) / np.sqrt(np.float32(128.0))
    kn = k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + EPS)
    decay = -np.exp(np.float32(0.0)) * _softplus(np.float32(0.0))
    alpha = np.exp(decay).astype(np.float32)
    beta = np.float32(1.0 / (1.0 + np.exp(np.float32(0.0))))
    out = np.zeros((48, 128), dtype=np.float32)
    silu = np.float32(1.0) / (1.0 + np.exp(np.float32(-1.0)))
    for head in range(48):
        state_h = state0[head] * alpha
        retrieved = state_h @ kn[head // 3]
        delta = beta * (v[head] - retrieved)
        state_h = state_h + np.outer(delta, kn[head // 3])
        y = state_h @ qn[head // 3]
        inv = 1.0 / np.sqrt(np.mean(y * y) + EPS)
        out[head] = y * inv * silu
    got = session.download_f32(destination, 48 * 128).reshape(48, 128)
    _close("deltanet", got, out, atol=2e-3)


def check_conv(session: Session) -> None:
    channels = 10240
    history = np.zeros((3, channels), dtype=np.float16)
    current = np.ones(channels, dtype=np.float32)
    weight = np.zeros((channels, 4), dtype=np.float16)
    weight[:, 3] = np.float16(1.0)
    destination = session.zeros(channels * 4)
    session.dispatch(
        "gdn_conv_shift",
        [
            session.upload(history),
            session.upload(current),
            session.upload(weight),
            destination,
        ],
        [np.uint32(channels).tobytes()],
        (channels // 256, 1, 1),
        (256, 1, 1),
    )
    silu = current / (1.0 + np.exp(-current))
    _close("conv", session.download_f32(destination, channels), silu.astype(np.float32), atol=1e-4)


def check_e4m3(session: Session) -> None:
    codes = np.array([0x00, 0x38, 0x7E, 0x80, 0x40], dtype=np.uint8)
    padded = np.zeros(32, dtype=np.uint8)
    padded[: codes.size] = codes
    destination = session.zeros(32 * 4)
    session.dispatch(
        "dequant_e4m3",
        [session.upload(padded), destination],
        [np.uint32(127).tobytes()],
        (32, 1, 1),
        (1, 1, 1),
    )
    expect = dequant_e4m3_block(padded, 127)
    _close("e4m3", session.download_f32(destination, 32), expect, atol=1e-4)


def check_rope(session: Session) -> None:
    query = np.ones(24 * 256, dtype=np.float32)
    key = np.ones(4 * 256, dtype=np.float32)
    weight = np.ones(256, dtype=np.float16)
    q_buf = session.upload(query)
    k_buf = session.upload(key)
    session.dispatch(
        "qk_norm_rope",
        [q_buf, k_buf, session.upload(weight), session.upload(weight)],
        [np.float32(1e7).tobytes(), np.float32(EPS).tobytes(), np.uint32(0).tobytes()],
        (28, 1, 1),
        (256, 1, 1),
    )
    got = session.download_f32(q_buf, 24 * 256)
    inv = np.float32(2.0) / np.sqrt(np.float32(1.0) + EPS)
    _close("rope", got, np.full_like(got, inv), atol=1e-4)


def check_attention(session: Session) -> None:
    """Four equal tokens. fp8 of 1 is exact, and a zero gate is sigmoid 1/2."""
    from qwen38_27b.attn import attend_plane, bind_plane
    from qwen38_27b.cache import plane_bytes

    class Runtime:
        pass

    runtime = Runtime()
    runtime.session = session
    runtime.max_pos = 256
    runtime.k_cache = [session.zeros(plane_bytes(256))]
    runtime.v_cache = [session.zeros(plane_bytes(256))]
    bind_plane(runtime)
    count = 4
    query = np.ones((count, 24, 256), dtype=np.float32)
    key = np.ones((count, 4, 256), dtype=np.float32)
    value = np.ones((count, 4, 256), dtype=np.float32)
    for token in range(count):
        value[token] = np.float32(token + 1)
    gate = np.zeros((count, 24, 256), dtype=np.float32)
    dest = session.zeros(count * 24 * 256 * 4)
    session.begin()
    attend_plane(
        runtime,
        query=session.upload(query),
        key=session.upload(key),
        value=session.upload(value),
        gate=session.upload(gate),
        dest=dest,
        pos0=0,
        count=count,
        slot=0,
    )
    session.end()
    got = session.download_f32(dest, count * 24 * 256).reshape(count, 24, 256)
    expect = np.zeros_like(got)
    for token in range(count):
        expect[token] = np.float32(0.5 * (token + 2) / 2.0)
    _close("attention", got, expect, atol=2e-3)


CHECKS = (
    ("fp4 tile", check_fp4_tile),
    ("fp4 gemv", check_fp4_gemv),
    ("fp4 x4", check_fp4_x4),
    ("fp4 span2", check_fp4_span2),
    ("rms", check_rms),
    ("swiglu", check_swiglu),
    ("deltanet", check_deltanet),
    ("conv", check_conv),
    ("e4m3", check_e4m3),
    ("rope", check_rope),
    ("attention", check_attention),
)


def run_checks(session: Session | None = None) -> dict[str, object]:
    device = session if session is not None else Session()
    rows: list[tuple[str, bool, str]] = []
    for name, fn in CHECKS:
        try:
            fn(device)
        except (AssertionError, RuntimeError, ValueError) as exc:
            rows.append((name, False, str(exc)))
        else:
            rows.append((name, True, ""))
    passed = all(ok for _name, ok, _detail in rows)
    return {"ok": passed, "rows": rows, "device": device.device_name}


def format_report(report: dict[str, object]) -> str:
    lines = [f"device: {report['device']}"]
    rows = report["rows"]
    assert isinstance(rows, list)
    for name, ok, detail in rows:
        state = "ok" if ok else "mismatch"
        suffix = f" {detail}" if detail else ""
        lines.append(f"{name}: {state}{suffix}")
    return "\n".join(lines)
