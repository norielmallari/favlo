"""Host formulas and the dispatches that have to match them."""

from __future__ import annotations

import numpy as np

from ternary_bonsai_2.formats import dequant_e4m3_block, dequant_fp8, dequant_int2, quantize_fp8, quantize_int2
from ternary_bonsai_2.session import Session

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


def _int2_case(
    rows: int, cols: int, inner: int, seed: int
) -> tuple[np.ndarray, bytes, bytes, np.ndarray]:
    rng = np.random.default_rng(seed)
    activation = rng.normal(scale=0.05, size=(rows, inner)).astype(np.float16)
    weight = rng.normal(scale=0.25, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_int2(weight)
    matrix = dequant_int2(data, scales, cols, inner)
    expected = activation.astype(np.float32) @ matrix.T
    return activation, data, scales, expected


def check_fp4_tile(session: Session) -> None:
    activation, data, scales, expected = _int2_case(64, 64, 512, 1)
    got = session.prefill("prefill_int2_n64", activation, data, scales, 64, 64, 512)
    _close("int2 tile", got, expected, atol=2e-3)
    activation, data, scales, expected = _int2_case(64, 32, 512, 2)
    got = session.prefill("prefill_int2_n32", activation, data, scales, 64, 32, 512)
    _close("int2 32-col", got, expected, atol=2e-3)
    activation, data, scales, expected = _int2_case(64, 128, 512, 3)
    got = session.prefill("prefill_int2_m32", activation, data, scales, 64, 128, 512)
    _close("int2 32x128", got, expected, atol=2e-3)


def check_fp4_gemv(session: Session) -> None:
    activation, data, scales, expected = _int2_case(1, 32, 512, 4)
    got = session.gemv_int2(activation.reshape(-1), data, scales, 32, 512)
    _close("int2 gemv", got, expected.reshape(-1), atol=2e-3)


def check_fp4_x4(session: Session) -> None:
    rng = np.random.default_rng(0)
    cols, inner = 8, 512
    activation = rng.normal(scale=0.05, size=inner).astype(np.float32)
    weight = rng.normal(scale=0.25, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_int2(weight)
    destination = session.zeros(cols * 4)
    session.dispatch(
        "gemv_int2_x4",
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
    expect = dequant_int2(data, scales, cols, inner) @ activation
    _close("int2 x4", session.download_f32(destination, cols), expect.astype(np.float32), atol=2e-3)


def check_fp4_span2(session: Session) -> None:
    rng = np.random.default_rng(1)
    cols, inner = 8, 512
    act0 = rng.normal(scale=0.05, size=inner).astype(np.float32)
    act1 = rng.normal(scale=0.05, size=inner).astype(np.float32)
    weight = rng.normal(scale=0.25, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_int2(weight)
    dest0 = session.zeros(cols * 4)
    dest1 = session.zeros(cols * 4)
    session.dispatch(
        "gemv_int2_span2",
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
    matrix = dequant_int2(data, scales, cols, inner)
    _close("int2 span2 a", session.download_f32(dest0, cols), (matrix @ act0).astype(np.float32), atol=2e-3)
    _close("int2 span2 b", session.download_f32(dest1, cols), (matrix @ act1).astype(np.float32), atol=2e-3)


def _fwht(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    span = 1
    while span < out.size:
        for start in range(0, out.size, span * 2):
            for offset in range(span):
                left = out[start + offset]
                right = out[start + offset + span]
                out[start + offset] = left + right
                out[start + offset + span] = left - right
        span *= 2
    return out


def check_hadamard(session: Session) -> None:
    rng = np.random.default_rng(5)
    width = 1024
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
    src = rng.normal(scale=0.5, size=width).astype(np.float32)
    dest = session.zeros(width * 4)
    back = session.zeros(width * 4)
    sign_buf = session.upload(signs.astype(np.float16))
    session.dispatch(
        "hadamard_rows",
        [session.upload(src), dest, sign_buf],
        [np.uint32(1).tobytes(), np.uint32(width).tobytes(), np.uint32(0).tobytes()],
        (1, 1, 1),
        (256, 1, 1),
    )
    expect = (_fwht(src * signs) / 32.0).astype(np.float32)
    _close("hadamard", session.download_f32(dest, width), expect, atol=2e-4)
    session.dispatch(
        "hadamard_rows",
        [dest, back, sign_buf],
        [np.uint32(1).tobytes(), np.uint32(width).tobytes(), np.uint32(1).tobytes()],
        (1, 1, 1),
        (256, 1, 1),
    )
    _close("hadamard inverse", session.download_f32(back, width), src, atol=2e-3)


def _fp8_case(cols: int, inner: int, seed: int) -> tuple[np.ndarray, bytes, bytes, np.ndarray]:
    rng = np.random.default_rng(seed)
    activation = rng.normal(scale=0.05, size=inner).astype(np.float32)
    weight = rng.normal(scale=0.05, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_fp8(weight)
    expect = dequant_fp8(data, scales, cols, inner) @ activation
    return activation, data, scales, expect.astype(np.float32)


def check_fp8_prefill(session: Session) -> None:
    rng = np.random.default_rng(2)
    shapes = (("fp8 gate", 64, 128, 256), ("fp8 down", 64, 64, 512))
    for label, rows, cols, inner in shapes:
        activation = rng.normal(scale=0.05, size=(rows, inner)).astype(np.float16)
        weight = rng.normal(scale=0.05, size=(cols, inner)).astype(np.float32)
        data, scales = quantize_fp8(weight)
        got = session.prefill("prefill_fp8_n64", activation, data, scales, rows, cols, inner)
        matrix = dequant_fp8(data, scales, cols, inner)
        expect = activation.astype(np.float32) @ matrix.T
        _close(label, got, expect.astype(np.float32), atol=2e-3)
    activation = rng.normal(scale=0.05, size=(64, 256)).astype(np.float16)
    weight = rng.normal(scale=0.05, size=(128, 256)).astype(np.float32)
    data, scales = quantize_fp8(weight)
    got = session.prefill("prefill_fp8_m32", activation, data, scales, 64, 128, 256)
    matrix = dequant_fp8(data, scales, 128, 256)
    expect = activation.astype(np.float32) @ matrix.T
    _close("fp8 32x128", got, expect.astype(np.float32), atol=2e-3)


def check_fp8_x4(session: Session) -> None:
    activation, data, scales, expect = _fp8_case(8, 256, 3)
    destination = session.zeros(8 * 4)
    session.dispatch(
        "gemv_fp8_x4",
        [
            session.upload(activation),
            session.upload(data),
            session.upload(scales),
            destination,
        ],
        [np.uint32(8).tobytes(), np.uint32(256).tobytes()],
        (2, 1, 1),
        (256, 1, 1),
    )
    _close("fp8 x4", session.download_f32(destination, 8), expect, atol=1e-3)
    activation, data, scales, expect = _fp8_case(4, 2560, 5)
    destination = session.zeros(4 * 4)
    session.dispatch(
        "gemv_fp8_x4",
        [
            session.upload(activation),
            session.upload(data),
            session.upload(scales),
            destination,
        ],
        [np.uint32(4).tobytes(), np.uint32(2560).tobytes()],
        (1, 1, 1),
        (160, 1, 1),
    )
    _close("fp8 x4 160", session.download_f32(destination, 4), expect, atol=1e-3)


def check_fp8_gate_up(session: Session) -> None:
    rng = np.random.default_rng(4)
    rows, inner = 8, 256
    activation = rng.normal(scale=0.05, size=inner).astype(np.float32)
    gate = rng.normal(scale=0.05, size=(rows, inner)).astype(np.float32)
    up = rng.normal(scale=0.05, size=(rows, inner)).astype(np.float32)
    gate_data, gate_scales = quantize_fp8(gate)
    up_data, up_scales = quantize_fp8(up)
    gate_dest = session.zeros(rows * 4)
    up_dest = session.zeros(rows * 4)
    session.dispatch(
        "ffn_gate_up_fp8",
        [
            session.upload(activation),
            session.upload(gate_data),
            session.upload(gate_scales),
            session.upload(up_data),
            session.upload(up_scales),
            gate_dest,
            up_dest,
        ],
        [np.uint32(rows).tobytes(), np.uint32(inner).tobytes()],
        (rows // 4, 1, 1),
        (256, 1, 1),
    )
    gate_expect = dequant_fp8(gate_data, gate_scales, rows, inner) @ activation
    up_expect = dequant_fp8(up_data, up_scales, rows, inner) @ activation
    _close("fp8 gate", session.download_f32(gate_dest, rows), gate_expect.astype(np.float32), atol=1e-3)
    _close("fp8 up", session.download_f32(up_dest, rows), up_expect.astype(np.float32), atol=1e-3)


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
        [
            np.float32(1e7).tobytes(),
            np.float32(EPS).tobytes(),
            np.uint32(0).tobytes(),
            np.uint32(0).tobytes(),
            np.uint32(0).tobytes(),
        ],
        (28, 1, 1),
        (256, 1, 1),
    )
    got = session.download_f32(q_buf, 24 * 256)
    inv = np.float32(2.0) / np.sqrt(np.float32(1.0) + EPS)
    _close("rope", got, np.full_like(got, inv), atol=1e-4)


def check_vision_gelu(session: Session) -> None:
    values = np.array([0.0, 1.0, -0.5, 20.0, -20.0], dtype=np.float64)
    destination = session.zeros(values.size * 4)
    session.dispatch(
        "vision_gelu_tanh",
        [session.upload(values.astype(np.float32)), destination],
        [np.uint32(values.size).tobytes()],
        (1, 1, 1),
        (32, 1, 1),
    )
    cube = values * values * values
    inner = np.sqrt(2.0 / np.pi) * (values + 0.044715 * cube)
    expect = (0.5 * values * (1.0 + np.tanh(inner))).astype(np.float32)
    _close("vision gelu", session.download_f32(destination, values.size), expect, atol=1e-4)


def check_vision_gelu_erf(session: Session) -> None:
    values = np.array([0.0, 1.0, -0.5], dtype=np.float32)
    destination = session.zeros(values.size * 4)
    session.dispatch(
        "vision_gelu_erf",
        [session.upload(values), destination],
        [np.uint32(values.size).tobytes()],
        (1, 1, 1),
        (values.size, 1, 1),
    )
    import math

    scaled = values * np.float32(2.0**-0.5)
    expect = np.float32(0.5) * values * (1.0 + np.vectorize(math.erf)(scaled))
    _close("vision erf gelu", session.download_f32(destination, values.size), expect, atol=1e-5)


def check_attention(session: Session) -> None:
    """Four equal tokens. fp8 of 1 is exact, and a zero gate is sigmoid 1/2."""
    from ternary_bonsai_2.attn import attend_plane, bind_plane
    from ternary_bonsai_2.cache import plane_bytes

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


def check_attention_blocks(session: Session) -> None:
    """Decode across two 512-key blocks. Values 1 and 2 are exact in fp8."""
    from ternary_bonsai_2.attn import attend_plane, bind_plane
    from ternary_bonsai_2.cache import plane_bytes

    class Runtime:
        pass

    runtime = Runtime()
    runtime.session = session
    runtime.max_pos = 2048
    runtime.k_cache = [session.zeros(plane_bytes(2048))]
    runtime.v_cache = [session.zeros(plane_bytes(2048))]
    bind_plane(runtime)
    count = 1024
    query = np.ones((count, 24, 256), dtype=np.float32)
    key = np.ones((count, 4, 256), dtype=np.float32)
    value = np.ones((count, 4, 256), dtype=np.float32)
    value[512:] = np.float32(2)
    gate = np.zeros((count, 24, 256), dtype=np.float32)
    session.begin()
    attend_plane(
        runtime,
        query=session.upload(query),
        key=session.upload(key),
        value=session.upload(value),
        gate=session.upload(gate),
        dest=session.zeros(count * 24 * 256 * 4),
        pos0=0,
        count=count,
        slot=0,
    )
    session.end()
    one = 1
    dest = session.zeros(one * 24 * 256 * 4)
    session.begin()
    attend_plane(
        runtime,
        query=session.upload(np.ones((one, 24, 256), dtype=np.float32)),
        key=session.upload(np.ones((one, 4, 256), dtype=np.float32)),
        value=session.upload(np.full((one, 4, 256), np.float32(2))),
        gate=session.upload(np.zeros((one, 24, 256), dtype=np.float32)),
        dest=dest,
        pos0=count,
        count=one,
        slot=0,
    )
    session.end()
    got = session.download_f32(dest, one * 24 * 256).reshape(one, 24, 256)
    expect = np.full_like(got, np.float32(0.5 * (512 + 513 * 2) / 1025))
    _close("attention blocks", got, expect, atol=2e-3)


CHECKS = (
    ("int2 tile", check_fp4_tile),
    ("hadamard", check_hadamard),
    ("int2 gemv", check_fp4_gemv),
    ("int2 x4", check_fp4_x4),
    ("int2 span2", check_fp4_span2),
    ("fp8 prefill", check_fp8_prefill),
    ("fp8 x4", check_fp8_x4),
    ("fp8 gate up", check_fp8_gate_up),
    ("rms", check_rms),
    ("swiglu", check_swiglu),
    ("deltanet", check_deltanet),
    ("conv", check_conv),
    ("e4m3", check_e4m3),
    ("rope", check_rope),
    ("vision gelu", check_vision_gelu),
    ("vision erf gelu", check_vision_gelu_erf),
    ("attention", check_attention),
    ("attention blocks", check_attention_blocks),
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
