"""Host formulas and the dispatches that have to match them."""

from __future__ import annotations

import numpy as np

from qwen36_35b_a3b.formats import (
    EXPERT_FP8,
    EXPERT_IMAGE,
    FP8_DOWN_DATA,
    FP8_DOWN_SCALE,
    FP8_GATE_SCALE,
    dequant_e4m3_block,
    dequant_fp4,
    dequant_fp8,
    fp4_image,
    fp8_image,
    pack_fp4_matrix,
    pack_fp8_matrix,
    quantize_fp4,
    quantize_fp8,
)
from qwen36_35b_a3b.session import Session

EPS = np.float32(1e-6)


def _close_codes(name: str, got: np.ndarray, expect: np.ndarray) -> None:
    """A one-code gap is the peak scale landing on a rounding boundary."""
    gap = np.abs(got.astype(np.int16) - expect.astype(np.int16))
    if int(np.max(gap)) > 1:
        message = f"{name} codes differ by {int(np.max(gap))}"
        raise AssertionError(message)


def _close_ulp(name: str, got: np.ndarray, expect: np.ndarray) -> None:
    """One float32 ulp. The peak divided by 127 can land on either side."""
    gap = np.abs(got.astype(np.float64) - expect.astype(np.float64))
    room = np.spacing(expect.astype(np.float32)).astype(np.float64)
    if np.any(gap > room):
        delta = float(np.max(gap))
        message = f"{name} max abs {delta}"
        raise AssertionError(message)


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


def _column_one(rows: int, inner: int) -> tuple[np.ndarray, bytes, bytes, np.ndarray]:
    weight = np.zeros((rows, inner), dtype=np.float32)
    weight[:, 0] = 1.0
    data, scales = quantize_fp4(weight)
    restored = dequant_fp4(data, scales, rows, inner)[:, 0]
    return weight, data, scales, restored


def check_fp4_tile(session: Session) -> None:
    activation, data, scales, expected = _probe_fp4(64, 32, 32)
    got = session.prefill("prefill_fp4_e8m0", activation, data, scales, 64, 32, 32)
    _close("fp4 tile", got, expected, rtol=0.0, atol=0.0)


def check_fp8_tile(session: Session) -> None:
    rows, cols, inner = 64, 32, 32
    activation = np.zeros((rows, inner), dtype=np.float16)
    activation[:, 0] = np.float16(1.0)
    weights = bytearray(cols * inner)
    for row in range(cols):
        weights[row * inner] = 0x38
    scales = bytes([127]) * (cols * (inner // 32))
    got = session.prefill("prefill_fp8_e8m0", activation, bytes(weights), scales, rows, cols, inner)
    _close("fp8 tile", got, np.ones((rows, cols), dtype=np.float32), atol=1e-3)


def check_fp4_gemv(session: Session) -> None:
    activation, data, scales, expected = _probe_fp4(1, 32, 32)
    got = session.gemv_fp4(activation.reshape(-1), data, scales, 32, 32)
    _close("fp4 gemv", got, expected.reshape(-1), rtol=0.0, atol=0.0)


def check_wide_fp4(session: Session) -> None:
    rows, cols, inner = 128, 64, 64
    rng = np.random.default_rng(0)
    activation = rng.normal(scale=0.05, size=(rows, inner)).astype(np.float16)
    weight = rng.normal(scale=0.05, size=(cols, inner)).astype(np.float32)
    data, scales = quantize_fp4(weight)
    got = session.prefill("prefill_fp4_e8m0", activation, data, scales, rows, cols, inner)
    expect = activation.astype(np.float32) @ dequant_fp4(data, scales, cols, inner).T
    _close("fp4 128x64x64", got, expect, rtol=2e-2, atol=2e-2)


def check_rows_32(session: Session) -> None:
    activation, data, scales, expected = _probe_fp4(64, 32, 256)
    got = session.prefill("prefill_fp4_e8m0", activation, data, scales, 64, 32, 256)
    _close("32-row projection", got, expected, rtol=0.0, atol=0.0)


def check_one_row(session: Session) -> None:
    weights = np.ones((1, 2048), dtype=np.float16)
    activation = np.ones(2048, dtype=np.float16)
    got = session.gemv_f16(activation, weights)
    _close("1-row gate", got, np.array([2048.0], dtype=np.float32), rtol=0.0, atol=0.0)


def check_router(session: Session) -> None:
    weights = np.zeros((256, 2048), dtype=np.float16)
    weights[:, 0] = np.arange(256, dtype=np.float16)
    activation = np.zeros(2048, dtype=np.float16)
    activation[0] = np.float16(1.0)
    got = session.gemv_f16(activation, weights)
    _close("router", got, np.arange(256, dtype=np.float32), rtol=0.0, atol=0.0)


def check_expert_image(session: Session) -> None:
    _gate, gate_data, gate_scales, gate_col = _column_one(1024, 2048)
    _down, down_data, down_scales, down_col = _column_one(2048, 512)
    gate_weight = np.zeros((1024, 2048), dtype=np.float32)
    gate_weight[:, 0] = 1.0
    down_weight = np.zeros((2048, 512), dtype=np.float32)
    down_weight[:, 0] = 1.0
    blob = fp4_image(gate_weight) + fp4_image(down_weight)
    if len(blob) != EXPERT_IMAGE:
        message = f"expert image {len(blob)} != {EXPERT_IMAGE}"
        raise AssertionError(message)
    activation = np.zeros(2048, dtype=np.float16)
    activation[0] = np.float16(1.0)
    got = session.gemv_fp4(activation, gate_data, gate_scales, 1024, 2048)
    _close("expert gate gemv", got, gate_col)
    chunk = np.zeros((64, 2048), dtype=np.float16)
    chunk[:, 0] = np.float16(1.0)
    tiled = session.prefill("prefill_fp4_e8m0", chunk, gate_data, gate_scales, 64, 1024, 2048)
    _close("expert gate tile", tiled, np.broadcast_to(gate_col, (64, 1024)).copy())
    short = np.zeros(512, dtype=np.float16)
    short[0] = np.float16(1.0)
    got_down = session.gemv_fp4(short, down_data, down_scales, 2048, 512)
    _close("expert down gemv", got_down, down_col)


def check_vision_pad(session: Session) -> None:
    logical = np.zeros((4304, 1152), dtype=np.float32)
    logical[:, 0] = 1.0
    data, scales, rows, inner = pack_fp4_matrix(logical, rows=4320, inner=1280)
    if (rows, inner) != (4320, 1280):
        message = f"vision pad stored {(rows, inner)}"
        raise AssertionError(message)
    activation = np.zeros((64, inner), dtype=np.float16)
    activation[:, 0] = np.float16(1.0)
    got = session.prefill("prefill_fp4_e8m0", activation, data, scales, 64, rows, inner)
    restored = dequant_fp4(data, scales, rows, inner)
    expect = activation.astype(np.float32) @ restored.T
    _close("vision fc1", got[:, :4304], expect[:, :4304])
    _close("vision fc1 pad rows", got[:, 4304:], np.zeros((64, 16), dtype=np.float32), atol=1e-5)
    if np.max(np.abs(restored[:, 1152:])) > 1e-6:
        message = "vision K pad is not zero"
        raise AssertionError(message)


def check_head72(session: Session) -> None:
    rng = np.random.default_rng(1)
    tokens = 4
    query = rng.normal(scale=0.1, size=72).astype(np.float32)
    key = rng.normal(scale=0.1, size=(tokens, 72)).astype(np.float32)
    value = rng.normal(scale=0.1, size=(tokens, 72)).astype(np.float32)
    destination = session.zeros(72 * 4)
    session.dispatch(
        "attend_head72",
        [session.upload(query), session.upload(key), session.upload(value), destination],
        [np.uint32(tokens).tobytes()],
        (1, 1, 1),
        (96, 1, 1),
    )
    scores = (key @ query) / np.float32(np.sqrt(72.0))
    weights = np.exp(scores - np.max(scores))
    weights /= np.sum(weights)
    expect = weights @ value
    _close("head 72", session.download_f32(destination, 72), expect.astype(np.float32))


def check_gelu(session: Session) -> None:
    values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    destination = session.zeros(values.size * 4)
    session.dispatch(
        "gelu_tanh_f32",
        [session.upload(values), destination],
        [],
        (values.size, 1, 1),
        (1, 1, 1),
    )
    cube = values + np.float32(0.044715) * values**3
    expect = np.float32(0.5) * values * (1.0 + np.tanh(np.float32(0.7978845608028654) * cube))
    _close("gelu", session.download_f32(destination, values.size), expect, atol=1e-5)


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.where(value > 20.0, value, np.log1p(np.exp(value))).astype(np.float32)


def check_deltanet(session: Session) -> None:
    mixed = np.ones(8192, dtype=np.float32)
    raw_a = np.zeros(32, dtype=np.float32)
    raw_b = np.zeros(32, dtype=np.float32)
    gate = np.ones((32, 128), dtype=np.float32)
    a_log = np.zeros(32, dtype=np.float16)
    dt_bias = np.zeros(32, dtype=np.float16)
    norm = np.ones(128, dtype=np.float16)
    state0 = np.ones((32, 128, 128), dtype=np.float32)
    destination = session.zeros(32 * 128 * 4)
    session.dispatch(
        "gdn_state",
        [
            session.upload(state0),
            session.upload(mixed),
            session.upload(raw_a),
            session.upload(raw_b),
            session.upload(gate),
            session.upload(a_log),
            session.upload(dt_bias),
            session.upload(norm),
            destination,
        ],
        [struct_f32(EPS)],
        (32, 1, 1),
        (128, 1, 1),
    )
    q = np.ones((16, 128), dtype=np.float32)
    k = np.ones((16, 128), dtype=np.float32)
    v = np.ones((32, 128), dtype=np.float32)
    qn = q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + EPS) / np.sqrt(np.float32(128.0))
    kn = k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + EPS)
    decay = -np.exp(np.float32(0.0)) * _softplus(np.float32(0.0))
    alpha = np.exp(decay).astype(np.float32)
    beta = np.float32(1.0 / (1.0 + np.exp(np.float32(0.0))))
    out = np.zeros((32, 128), dtype=np.float32)
    for head in range(32):
        state = state0[head] * alpha
        retrieved = state @ kn[head // 2]
        delta = beta * (v[head] - retrieved)
        state = state + np.outer(delta, kn[head // 2])
        y = state @ qn[head // 2]
        inv = 1.0 / np.sqrt(np.mean(y * y) + EPS)
        silu = gate[head] * (1.0 / (1.0 + np.exp(-gate[head])))
        out[head] = y * inv * norm.astype(np.float32) * silu
    got = session.download_f32(destination, 32 * 128).reshape(32, 128)
    _close("deltanet", got, out, atol=1e-4)


def check_gdn_chunk(session: Session) -> None:
    """Chunkwise prefill matches the one-token recurrence on a short sequence."""
    count = 5
    rng = np.random.default_rng(3)
    mixed = rng.normal(size=(count, 8192)).astype(np.float32) * np.float32(0.2)
    raw_a = rng.normal(size=(count, 32)).astype(np.float32) * np.float32(0.3)
    raw_b = rng.normal(size=(count, 32)).astype(np.float32) * np.float32(0.3)
    gate = rng.normal(size=(count, 4096)).astype(np.float32) * np.float32(0.2)
    a_log = rng.normal(size=32).astype(np.float16) * np.float16(0.1)
    dt_bias = rng.normal(size=32).astype(np.float16) * np.float16(0.1)
    norm = rng.normal(size=128).astype(np.float16) * np.float16(0.05)
    state0 = rng.normal(size=(32, 128, 128)).astype(np.float32) * np.float32(0.02)
    buffers = {
        "mixed": session.upload(mixed),
        "a": session.upload(raw_a),
        "b": session.upload(raw_b),
        "gate": session.upload(gate),
        "alog": session.upload(a_log),
        "bias": session.upload(dt_bias),
        "norm": session.upload(norm),
        "state": session.upload(state0),
        "y": session.zeros(count * 4096 * 4),
        "kkt": session.zeros(32 * 64 * 64 * 4),
        "w": session.zeros(32 * 64 * 128 * 4),
        "u": session.zeros(32 * 64 * 128 * 4),
        "vbar": session.zeros(32 * 64 * 128 * 4),
        "logg": session.zeros(32 * 64 * 4),
    }
    session.dispatch(
        "gdn_l2_rows",
        [buffers["mixed"]],
        [struct_f32(float(EPS)), _u32(count)],
        (count, 16, 1),
        (128, 1, 1),
    )
    session.dispatch(
        "gdn_chunk_solve",
        [
            buffers["state"],
            buffers["mixed"],
            buffers["a"],
            buffers["b"],
            buffers["alog"],
            buffers["bias"],
            buffers["kkt"],
            buffers["w"],
            buffers["u"],
            buffers["vbar"],
            buffers["logg"],
        ],
        [_u32(0), _u32(count)],
        (32, 1, 1),
        (128, 1, 1),
    )
    session.dispatch(
        "gdn_chunk_out",
        [
            buffers["state"],
            buffers["mixed"],
            buffers["vbar"],
            buffers["logg"],
            buffers["y"],
        ],
        [_u32(0), _u32(count)],
        (32, count, 1),
        (128, 1, 1),
    )
    session.dispatch(
        "gdn_chunk_state",
        [buffers["state"], buffers["mixed"], buffers["vbar"], buffers["logg"]],
        [_u32(0), _u32(count)],
        (32, 128, 1),
        (128, 1, 1),
    )
    session.dispatch(
        "gdn_y_norm",
        [buffers["y"], buffers["gate"], buffers["norm"]],
        [struct_f32(float(EPS)), _u32(count)],
        (count, 32, 1),
        (128, 1, 1),
    )
    got = session.download_f32(buffers["y"], count * 4096).reshape(count, 32, 128)
    host = mixed.copy()
    for token in range(count):
        for qhead in range(16):
            q = host[token, qhead * 128 : (qhead + 1) * 128]
            k = host[token, 2048 + qhead * 128 : 2048 + (qhead + 1) * 128]
            q /= np.sqrt(np.sum(q * q) + EPS)
            q *= np.float32(1.0 / np.sqrt(128.0))
            k /= np.sqrt(np.sum(k * k) + EPS)
            host[token, qhead * 128 : (qhead + 1) * 128] = q
            host[token, 2048 + qhead * 128 : 2048 + (qhead + 1) * 128] = k
    state = state0.copy()
    expect = np.zeros((count, 32, 128), dtype=np.float32)
    for token in range(count):
        for head in range(32):
            q = host[token, (head // 2) * 128 : (head // 2 + 1) * 128]
            k = host[token, 2048 + (head // 2) * 128 : 2048 + (head // 2 + 1) * 128]
            v = host[token, 4096 + head * 128 : 4096 + (head + 1) * 128]
            aval = raw_a[token, head]
            bval = raw_b[token, head]
            g = -np.exp(np.float32(a_log[head])) * _softplus(aval + np.float32(dt_bias[head]))
            alpha = np.exp(g).astype(np.float32)
            beta = np.float32(1.0 / (1.0 + np.exp(-bval)))
            state[head] *= alpha
            delta = beta * (v - state[head] @ k)
            state[head] = state[head] + np.outer(delta, k)
            expect[token, head] = state[head] @ q
    for token in range(count):
        for head in range(32):
            y = expect[token, head]
            inv = np.float32(1.0) / np.sqrt(np.mean(y * y) + EPS)
            z = gate[token, head * 128 : (head + 1) * 128]
            silu = z / (np.float32(1.0) + np.exp(-z))
            expect[token, head] = y * inv * norm.astype(np.float32) * silu
    _close("gdn chunk", got, expect, atol=2e-3, rtol=2e-3)


def _u32(value: int) -> bytes:
    return np.uint32(value).tobytes()


def check_conv(session: Session) -> None:
    channels = 8192
    history = np.zeros((3, channels), dtype=np.float16)
    current = np.ones(channels, dtype=np.float32)
    weight = np.zeros((channels, 4), dtype=np.float16)
    weight[:, 3] = np.float16(1.0)
    destination = session.zeros(channels * 4)
    session.dispatch(
        "gdn_conv",
        [
            session.upload(history),
            session.upload(current),
            session.upload(weight),
            destination,
        ],
        [np.uint32(channels).tobytes()],
        (channels, 1, 1),
        (1, 1, 1),
    )
    silu = current / (1.0 + np.exp(-current))
    _close("conv", session.download_f32(destination, channels), silu.astype(np.float32), atol=1e-5)


def check_conv_rows(session: Session) -> None:
    channels = 8
    count = 4
    rng = np.random.default_rng(3)
    history = rng.normal(size=(3, channels)).astype(np.float16)
    qkv = rng.normal(size=(count, channels)).astype(np.float32)
    weight = rng.normal(size=(channels, 4)).astype(np.float16)
    mixed = session.zeros(count * channels * 4)
    kept = session.upload(history.copy())
    session.begin()
    session.encode(
        "gdn_conv_rows",
        [
            (kept, 0),
            (session.upload(qkv), 0),
            (session.upload(weight), 0),
            (mixed, 0),
        ],
        [_u32(count), _u32(channels)],
        (count, 1, 1),
        (256, 1, 1),
    )
    session.encode(
        "gdn_conv_keep",
        [(kept, 0), (session.upload(qkv), 0)],
        [_u32(count), _u32(channels)],
        (1, 1, 1),
        (256, 1, 1),
    )
    session.end()
    taps = np.zeros((count, 4, channels), dtype=np.float32)
    past = history.astype(np.float32)
    for token in range(count):
        taps[token, 3] = qkv[token]
        for age in range(1, 4):
            index = token - age
            if index >= 0:
                taps[token, 3 - age] = qkv[index]
            else:
                taps[token, 3 - age] = past[index + 3]
    acc = np.zeros((count, channels), dtype=np.float32)
    for tap in range(4):
        acc += taps[:, tap, :] * weight[:, tap].astype(np.float32)
    expect = acc / (1.0 + np.exp(-acc))
    got = session.download_f32(mixed, count * channels).reshape(count, channels)
    _close("conv rows", got, expect, atol=1e-4)
    kept_got = np.frombuffer(session.download(kept, 3 * channels * 2), dtype=np.float16).reshape(
        3, channels
    )
    _close("conv keep", kept_got.astype(np.float32), qkv[-3:].astype(np.float32), atol=1e-3)


def check_e4m3(session: Session) -> None:
    codes = np.array([0x00, 0x38, 0x7E, 0x80, 0x40], dtype=np.uint8)
    padded = np.zeros(32, dtype=np.uint8)
    padded[: codes.size] = codes
    destination = session.zeros(32 * 4)
    session.dispatch(
        "dequant_e4m3",
        [session.upload(padded), destination],
        [np.uint32(130).tobytes()],
        (32, 1, 1),
        (1, 1, 1),
    )
    expect = dequant_e4m3_block(padded, 130)
    _close("e4m3", session.download_f32(destination, 32), expect, atol=1e-3)


def check_rope(session: Session) -> None:
    hidden = np.ones(256, dtype=np.float32)
    positions = np.array([1, 1, 1], dtype=np.uint32)
    buffer = session.upload(hidden)
    session.dispatch(
        "rope_partial",
        [buffer],
        [positions.tobytes(), struct_f32(np.float32(1e7))],
        (32, 1, 1),
        (1, 1, 1),
    )
    got = session.download_f32(buffer, 256)
    expect = hidden.copy()
    theta = np.float64(1e7)
    for pair in range(32):
        angle = theta ** (-pair / 32.0)
        c = np.cos(angle)
        s = np.sin(angle)
        i0 = pair
        i1 = pair + 32
        x0 = float(expect[i0])
        x1 = float(expect[i1])
        expect[i0] = x0 * c - x1 * s
        expect[i1] = x0 * s + x1 * c
    _close("rope", got[:64], expect[:64].astype(np.float32), atol=1e-5)
    _close("rope tail", got[64:], hidden[64:], atol=0.0)


def _router8_expect(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(range(256), key=lambda index: (-float(logits[index]), index))[:8]
    chosen = logits[order]
    raised = np.exp(chosen - chosen[0])
    weights = (raised / np.sum(raised)).astype(np.float32)
    return np.asarray(order, dtype=np.uint32), weights


def check_router_coeff(session: Session) -> None:
    rng = np.random.default_rng(2)
    router = rng.normal(scale=0.02, size=(256, 2048)).astype(np.float16)
    coeff = rng.normal(scale=0.02, size=2048).astype(np.float16)
    activation = rng.normal(scale=0.02, size=2048).astype(np.float32)
    router_out = session.zeros(256 * 4)
    coeff_out = session.zeros(4)
    session.dispatch(
        "router_coeff",
        [
            session.upload(activation),
            session.upload(np.ascontiguousarray(router)),
            session.upload(np.ascontiguousarray(coeff)),
            router_out,
            coeff_out,
        ],
        [],
        (257, 1, 1),
        (32, 1, 1),
    )
    expect = router.astype(np.float32) @ activation
    _close("router coeff rows", session.download_f32(router_out, 256), expect, rtol=1e-3, atol=1e-3)
    got = float(session.download_f32(coeff_out, 1)[0])
    logit = float(coeff.astype(np.float32) @ activation)
    target = 1.0 / (1.0 + np.exp(-logit))
    if abs(got - target) > 1e-3:
        message = f"router coeff sigmoid {got} != {target}"
        raise AssertionError(message)


def check_router8(session: Session) -> None:
    rng = np.random.default_rng(1)
    cases = [
        np.arange(256, dtype=np.float32),
        np.zeros(256, dtype=np.float32),
        rng.normal(size=256).astype(np.float32),
    ]
    tied = cases[2].copy()
    tied[4] = tied[200]
    cases.append(tied)
    for logits in cases:
        index = session.zeros(8 * 4)
        weight = session.zeros(8 * 4)
        session.dispatch(
            "router8",
            [session.upload(logits), index, weight],
            [],
            (1, 1, 1),
            (256, 1, 1),
        )
        expect_index, expect_weight = _router8_expect(logits)
        got_index = np.frombuffer(session.download(index, 8 * 4), dtype=np.uint32).copy()
        if not np.array_equal(got_index, expect_index):
            message = f"router8 index {got_index.tolist()} != {expect_index.tolist()}"
            raise AssertionError(message)
        _close("router8 weight", session.download_f32(weight, 8), expect_weight, atol=1e-5)


def check_topk(session: Session) -> None:
    logits = np.arange(256, dtype=np.float32)
    index = session.zeros(8 * 4)
    weight = session.zeros(8 * 4)
    session.dispatch(
        "topk8",
        [session.upload(logits), index, weight],
        [],
        (1, 1, 1),
        (256, 1, 1),
    )
    got_index = np.frombuffer(session.download(index, 8 * 4), dtype=np.uint32).copy()
    chosen = logits[255:247:-1]
    weights = (chosen / np.sum(chosen)).astype(np.float32)
    _close("topk weight", session.download_f32(weight, 8), weights, atol=1e-5)
    if not np.array_equal(got_index, np.arange(255, 247, -1, dtype=np.uint32)):
        message = f"topk index {got_index.tolist()}"
        raise AssertionError(message)


def check_elementwise(session: Session) -> None:
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    for name, expect in (
        ("silu_f32", values / (1.0 + np.exp(-values))),
        ("sigmoid_f32", 1.0 / (1.0 + np.exp(-values))),
        ("softplus_f32", _softplus(values)),
    ):
        destination = session.zeros(values.size * 4)
        session.dispatch(
            name,
            [session.upload(values), destination],
            [],
            (values.size, 1, 1),
            (1, 1, 1),
        )
        got = session.download_f32(destination, values.size)
        _close(name, got, expect.astype(np.float32), atol=1e-5)
    left = np.ones(4, dtype=np.float32)
    right = np.full(4, 2.0, dtype=np.float32)
    added = session.zeros(16)
    session.dispatch(
        "residual_add_f32",
        [session.upload(left), session.upload(right), added],
        [],
        (4, 1, 1),
        (1, 1, 1),
    )
    _close("residual", session.download_f32(added, 4), np.full(4, 3.0, dtype=np.float32), atol=0.0)
    gate = values
    up = np.full_like(values, 2.0)
    swiglu = session.zeros(values.size * 4)
    session.dispatch(
        "swiglu_f32",
        [session.upload(gate), session.upload(up), swiglu],
        [],
        (values.size, 1, 1),
        (1, 1, 1),
    )
    expect = (gate / (1.0 + np.exp(-gate))) * up
    got = session.download_f32(swiglu, values.size)
    _close("swiglu", got, expect.astype(np.float32), atol=1e-5)


def check_norms(session: Session) -> None:
    width = 32
    values = np.ones(width, dtype=np.float32)
    weight = np.zeros(width, dtype=np.float16)
    destination = session.zeros(width * 4)
    session.dispatch(
        "rmsnorm_f32",
        [session.upload(values), session.upload(weight), destination],
        [np.uint32(width).tobytes(), struct_f32(EPS)],
        (1, 1, 1),
        (256, 1, 1),
    )
    expect = values / np.sqrt(np.mean(values * values) + EPS)
    got = session.download_f32(destination, width)
    _close("rmsnorm", got, expect.astype(np.float32), atol=1e-5)
    bias = np.ones(width, dtype=np.float16)
    gain = np.ones(width, dtype=np.float16)
    session.dispatch(
        "layernorm_f32",
        [session.upload(values), session.upload(gain), session.upload(bias), destination],
        [np.uint32(width).tobytes(), struct_f32(EPS)],
        (1, 1, 1),
        (256, 1, 1),
    )
    centered = values - np.mean(values)
    layer = centered / np.sqrt(np.mean(centered * centered) + EPS) + 1.0
    got = session.download_f32(destination, width)
    _close("layernorm", got, layer.astype(np.float32), atol=1e-4)
    heads = np.ones((2, 128), dtype=np.float32)
    buffer = session.upload(heads)
    session.dispatch(
        "l2_heads_f32",
        [buffer],
        [struct_f32(EPS), struct_f32(np.float32(1.0))],
        (2, 1, 1),
        (128, 1, 1),
    )
    got = session.download_f32(buffer, 256).reshape(2, 128)
    scale = 1.0 / np.sqrt(np.float32(128.0) + EPS)
    _close("l2", got, np.full_like(got, scale), atol=1e-5)


def check_moe_group(session: Session) -> None:
    count = 3
    ids = np.array(
        [
            [1, 2, 1, 0, 3, 3, 2, 0],
            [2, 2, 3, 1, 0, 0, 5, 5],
            [1, 4, 4, 4, 1, 2, 0, 3],
        ],
        dtype=np.uint32,
    )
    mix = (np.arange(count * 8, dtype=np.float32) * np.float32(0.01)).reshape(count, 8)
    csr_i = session.zeros((count * 8 + 1024) * 4)
    csr_w = session.zeros((count * 8 + 1024) * 4)
    expert_of = session.zeros(256 * 4)
    slot_n = session.zeros(256 * 4)
    slot_csr = session.zeros(256 * 4)
    row_base = session.zeros(256 * 4)
    jobs = session.zeros(512 * 2 * 4)
    info = session.zeros(8 * 16)
    csr_lane = session.zeros((count * 8 + 1024) * 4)
    pack_row = session.zeros(count * 8 * 4)
    grouped = [
        session.upload(ids),
        session.upload(mix),
        csr_i,
        csr_w,
        expert_of,
        slot_n,
        slot_csr,
        row_base,
        jobs,
        info,
        csr_lane,
        pack_row,
    ]
    session.dispatch(
        "moe_group",
        grouped,
        [np.uint32(count).tobytes()],
        (1, 1, 1),
        (256, 1, 1),
    )
    header = np.frombuffer(session.download(info, 16), dtype=np.uint32)
    n_slots, n_jobs, total_rows = (int(header[0]), int(header[1]), int(header[2]))
    flat_e = ids.reshape(-1)
    flat_w = mix.reshape(-1)
    flat_t = np.repeat(np.arange(count, dtype=np.uint32), 8)
    flat_lane = np.tile(np.arange(8, dtype=np.uint32), count)
    order = np.argsort(flat_e, kind="stable")
    flat_e = flat_e[order]
    flat_t = flat_t[order]
    flat_w = flat_w[order]
    flat_lane = flat_lane[order]
    expect_pack = np.zeros(count * 8, dtype=np.uint32)
    counts = np.bincount(flat_e, minlength=256)
    offsets: list[int] = []
    cursor = 0
    begin = 0
    tokens: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    expect_slots: list[tuple[int, int, int, int]] = []
    row = 0
    expect_jobs: list[tuple[int, int]] = []
    for expert in range(256):
        align = (4 - cursor % 4) % 4
        cursor += align
        offsets.append(cursor)
        n = int(counts[expert])
        if n:
            tokens.append(flat_t[begin : begin + n])
            weights.append(flat_w[begin : begin + n])
            lanes = flat_lane[begin : begin + n]
            for local in range(n):
                at = int(tokens[-1][local]) * 8 + int(lanes[local])
                expect_pack[at] = row + local
            pad = (n + 63) & ~63
            slot = len(expect_slots)
            expect_slots.append((expert, n, cursor, row))
            for tile in range(pad // 64):
                expect_jobs.append((slot, tile))
            row += pad
        cursor += n
        begin += n
    assert (n_slots, n_jobs, total_rows) == (len(expect_slots), len(expect_jobs), row)
    got_of = np.frombuffer(session.download(expert_of, n_slots * 4), dtype=np.uint32)
    got_n = np.frombuffer(session.download(slot_n, n_slots * 4), dtype=np.uint32)
    got_csr = np.frombuffer(session.download(slot_csr, n_slots * 4), dtype=np.uint32)
    got_row = np.frombuffer(session.download(row_base, n_slots * 4), dtype=np.uint32)
    for slot, (expert, n, start, base) in enumerate(expect_slots):
        assert int(got_of[slot]) == expert
        assert int(got_n[slot]) == n
        assert int(got_csr[slot]) == start
        assert int(got_row[slot]) == base
        got_tokens = np.frombuffer(session.download(csr_i, (start + n) * 4), dtype=np.uint32)
        got_weights = session.download_f32(csr_w, start + n)
        assert np.array_equal(got_tokens[start : start + n], tokens[slot])
        _close(f"mix {expert}", got_weights[start : start + n], weights[slot], atol=0.0)
    got_jobs = np.frombuffer(session.download(jobs, n_jobs * 2 * 4), dtype=np.uint32).reshape(
        n_jobs, 2
    )
    for index, pair in enumerate(expect_jobs):
        assert (int(got_jobs[index, 0]), int(got_jobs[index, 1])) == pair
    got_pack = np.frombuffer(session.download(pack_row, count * 8 * 4), dtype=np.uint32)
    assert np.array_equal(got_pack, expect_pack)
    # The zero grid is record 1, 16 bytes past the header. A GPU write of that
    # record has to be what the next encoder dispatches.
    nfloat = total_rows * 1024
    pack = session.zeros(nfloat * 4)
    memoryview(pack.contents().as_buffer(nfloat * 4))[:] = np.ones(
        nfloat, dtype=np.float32
    ).tobytes()
    bindings = [
        (session.upload(ids), 0),
        (session.upload(mix), 0),
        (csr_i, 0),
        (csr_w, 0),
        (expert_of, 0),
        (slot_n, 0),
        (slot_csr, 0),
        (row_base, 0),
        (jobs, 0),
        (info, 0),
        (csr_lane, 0),
        (pack_row, 0),
    ]
    session.begin()
    session.encode("moe_group", bindings, [np.uint32(count).tobytes()], (1, 1, 1), (256, 1, 1))
    session.split()
    session.encode_indirect("zero_f32", [(pack, 0)], [], info, 32, (256, 1, 1))
    session.end()
    cleared = session.download_f32(pack, nfloat)
    if not np.all(cleared == 0):
        message = "indirect zero grid did not cover the grouped rows"
        raise AssertionError(message)


def check_grouped_expert(session: Session) -> None:
    """One device expert group matches the fp4 gate, SiLU, and down product."""
    count = 4
    hidden = 2048
    gate = np.zeros((1024, hidden), dtype=np.float32)
    gate[:, 0] = 1.0
    down = np.zeros((hidden, 512), dtype=np.float32)
    down[:, 0] = 1.0
    image = fp4_image(gate) + fp4_image(down)
    if len(image) != EXPERT_IMAGE:
        message = f"expert image {len(image)} != {EXPERT_IMAGE}"
        raise AssertionError(message)
    gate_data, gate_scales, _, _ = pack_fp4_matrix(gate)
    down_data, down_scales, _, _ = pack_fp4_matrix(down)
    act = np.zeros(hidden, dtype=np.float32)
    act[0] = 1.0
    gate_out = dequant_fp4(gate_data, gate_scales, 1024, hidden) @ act
    gated = gate_out[:512] * (1.0 / (1.0 + np.exp(-gate_out[:512]))) * gate_out[512:]
    expect = dequant_fp4(down_data, down_scales, hidden, 512) @ gated.astype(np.float16).astype(
        np.float32
    )
    ids = np.zeros((count, 8), dtype=np.uint32)
    mix = np.full((count, 8), np.float32(0.125))
    normed = np.zeros((count, hidden), dtype=np.float32)
    normed[:, 0] = 1.0
    rows = 64
    csr_i = session.zeros((count * 8 + 1024) * 4)
    csr_w = session.zeros((count * 8 + 1024) * 4)
    csr_lane = session.zeros((count * 8 + 1024) * 4)
    expert_of = session.zeros(256 * 4)
    slot_n = session.zeros(256 * 4)
    slot_csr = session.zeros(256 * 4)
    row_base = session.zeros(256 * 4)
    jobs = session.zeros(512 * 2 * 4)
    info = session.zeros(8 * 16)
    pack_row = session.zeros(count * 8 * 4)
    pack = session.zeros(rows * hidden * 2 + 1024)
    gate_buf = session.zeros(rows * 1024 * 4 + 1024)
    gated_buf = session.zeros(rows * 512 * 4 + 1024)
    down_buf = session.zeros(rows * hidden * 4 + 1024)
    hidden_buf = session.zeros(count * hidden * 4)
    file = session.upload(np.frombuffer(image, dtype=np.uint8))
    width = int(session.pipes["prefill_fp4_expert_tile"].threadExecutionWidth())
    tile_threads = (width * 4, 1, 1)
    threads = (256, 1, 1)

    def indirect(name, bindings, constants, record, group_threads):
        session.encode_indirect(
            name, bindings, constants, info, 16 + record * 16, group_threads
        )

    session.begin()
    session.encode(
        "moe_group",
        [
            (session.upload(ids), 0),
            (session.upload(mix), 0),
            (csr_i, 0),
            (csr_w, 0),
            (expert_of, 0),
            (slot_n, 0),
            (slot_csr, 0),
            (row_base, 0),
            (jobs, 0),
            (info, 0),
            (csr_lane, 0),
            (pack_row, 0),
        ],
        [np.uint32(count).tobytes()],
        (1, 1, 1),
        threads,
    )
    session.split()
    indirect("zero_f32", [(pack, 0)], [], 1, threads)
    indirect(
        "gather_half_grouped",
        [
            (session.upload(normed), 0),
            (csr_i, 0),
            (pack, 0),
            (slot_n, 0),
            (slot_csr, 0),
            (row_base, 0),
            (jobs, 0),
            (info, 0),
        ],
        [np.uint32(hidden).tobytes()],
        0,
        threads,
    )
    indirect(
        "prefill_fp4_expert_tile",
        [
            (pack, 0),
            (file, 0),
            (gate_buf, 0),
            (jobs, 0),
            (expert_of, 0),
            (row_base, 0),
            (info, 0),
        ],
        [
            np.uint32(hidden).tobytes(),
            np.uint32(1024).tobytes(),
            np.uint32(0).tobytes(),
            np.uint32(1_048_576).tobytes(),
            np.uint32(EXPERT_IMAGE).tobytes(),
        ],
        2,
        tile_threads,
    )
    indirect(
        "silu_mul_grouped",
        [(gate_buf, 0), (gated_buf, 0), (info, 0)],
        [],
        3,
        threads,
    )
    indirect(
        "cast_f32_f16_n",
        [(gated_buf, 0), (pack, 0), (info, 0)],
        [],
        4,
        threads,
    )
    indirect(
        "prefill_fp4_expert_tile",
        [
            (pack, 0),
            (file, 0),
            (down_buf, 0),
            (jobs, 0),
            (expert_of, 0),
            (row_base, 0),
            (info, 0),
        ],
        [
            np.uint32(512).tobytes(),
            np.uint32(hidden).tobytes(),
            np.uint32(1_114_112).tobytes(),
            np.uint32(1_114_112 + 524_288).tobytes(),
            np.uint32(EXPERT_IMAGE).tobytes(),
        ],
        5,
        tile_threads,
    )
    session.encode(
        "scatter_add_grouped",
        [
            (hidden_buf, 0),
            (down_buf, 0),
            (pack_row, 0),
            (session.upload(mix), 0),
        ],
        [np.uint32(hidden).tobytes(), np.uint32(count).tobytes()],
        (count, 1, 1),
        threads,
    )
    session.end()
    got_down = session.download_f32(down_buf, 32 * hidden).reshape(32, hidden)
    for row in range(32):
        _close(f"grouped down {row}", got_down[row], expect, atol=2e-3)
    got_hidden = session.download_f32(hidden_buf, count * hidden).reshape(count, hidden)
    for token in range(count):
        _close(f"grouped mix {token}", got_hidden[token], expect, atol=2e-3)


def check_remap_expert_ids(session: Session) -> None:
    ids = np.array([2, 0, 5, 1], dtype=np.uint32)
    order = np.arange(256, dtype=np.uint8)
    order[2] = np.uint8(9)
    order[0] = np.uint8(4)
    order[5] = np.uint8(7)
    order[1] = np.uint8(3)
    source = session.upload(ids)
    session.dispatch(
        "remap_expert_ids",
        [source, session.upload(order.tobytes())],
        [np.uint32(4).tobytes(), np.uint32(0).tobytes()],
        (1, 1, 1),
        (32, 1, 1),
    )
    got = np.frombuffer(session.download(source, 16), dtype=np.uint32)
    assert got.tolist() == [9, 4, 7, 3]


def check_gather_and_attend(session: Session) -> None:
    table = np.arange(32, dtype=np.float32).reshape(4, 8)
    destination = session.zeros(8 * 4)
    session.dispatch(
        "gather_row",
        [session.upload(table), destination],
        [np.uint32(8).tobytes(), np.uint32(2).tobytes()],
        (8, 1, 1),
        (1, 1, 1),
    )
    _close("gather", session.download_f32(destination, 8), table[2], atol=0.0)
    rng = np.random.default_rng(2)
    tokens = 3
    query = rng.normal(scale=0.1, size=256).astype(np.float32)
    key = rng.normal(scale=0.1, size=(tokens, 256)).astype(np.float32)
    value = rng.normal(scale=0.1, size=(tokens, 256)).astype(np.float32)
    out = session.zeros(256 * 4)
    session.dispatch(
        "attend_gqa",
        [session.upload(query), session.upload(key), session.upload(value), out],
        [np.uint32(tokens).tobytes()],
        (1, 1, 1),
        (256, 1, 1),
    )
    scores = (key @ query) / np.float32(16.0)
    weights = np.exp(scores - np.max(scores))
    weights /= np.sum(weights)
    _close("attend", session.download_f32(out, 256), (weights @ value).astype(np.float32))


def struct_f32(value: float) -> bytes:
    return np.float32(value).tobytes()


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (np.float32(1.0) + np.exp(-values))


def _routed_row(
    gate: np.ndarray, down: np.ndarray, act: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    proj = gate @ act
    mid = _silu(proj[:512]) * proj[512:]
    return down @ mid, mid


def check_fp8_expert(session: Session) -> None:
    """Gate row 17 matches the dequantized dot. The fp4 and fp8 partials add."""
    if FP8_GATE_SCALE != 1024 * 2048:
        message = "fp8 gate scale does not follow the data plane"
        raise AssertionError(message)
    if FP8_DOWN_DATA != FP8_GATE_SCALE + 1024 * (2048 // 32):
        message = "fp8 down plane does not follow the gate scale"
        raise AssertionError(message)
    if FP8_DOWN_SCALE != FP8_DOWN_DATA + 2048 * 512:
        message = "fp8 down scale does not follow the down data"
        raise AssertionError(message)
    if EXPERT_FP8 != FP8_DOWN_SCALE + 2048 * (512 // 32):
        message = f"fp8 expert image is not {EXPERT_FP8}"
        raise AssertionError(message)
    rng = np.random.default_rng(0)
    gate = rng.normal(scale=0.02, size=(1024, 2048)).astype(np.float32)
    data, scales, _rows, _inner = pack_fp8_matrix(gate)
    blob = fp8_image(gate)
    scale_at = blob[FP8_GATE_SCALE : FP8_GATE_SCALE + len(scales)]
    if len(data) != FP8_GATE_SCALE or scale_at != scales:
        message = "fp8 gate image does not place the scale plane on the data boundary"
        raise AssertionError(message)
    restored = dequant_fp8(data, scales, 1024, 2048)
    act = rng.normal(scale=0.02, size=(64, 2048)).astype(np.float16)
    expect = act.astype(np.float32) @ restored.T
    info = np.zeros(4, dtype=np.uint32)
    info[1] = 1
    jobs = np.zeros(2, dtype=np.uint32)
    expert_of = np.zeros(1, dtype=np.uint32)
    row_base = np.zeros(1, dtype=np.uint32)
    offsets = np.zeros(1, dtype=np.uint64)
    widths = np.array([8], dtype=np.uint8)
    dest = session.zeros(64 * 1024 * 4)
    width = int(session.pipes["prefill_fp8_expert_tile_at"].threadExecutionWidth())
    session.begin()
    session.encode(
        "prefill_fp8_expert_tile_at",
        [
            (session.upload(act), 0),
            (session.upload(np.frombuffer(blob, dtype=np.uint8)), 0),
            (dest, 0),
            (session.upload(jobs), 0),
            (session.upload(expert_of), 0),
            (session.upload(row_base), 0),
            (session.upload(info), 0),
            (session.upload(offsets), 0),
            (session.upload(widths), 0),
        ],
        [
            np.uint32(2048).tobytes(),
            np.uint32(1024).tobytes(),
            np.uint32(0).tobytes(),
            np.uint32(FP8_GATE_SCALE).tobytes(),
        ],
        (16, 1, 1),
        (width * 4, 1, 1),
    )
    session.end()
    got = session.download_f32(dest, 64 * 1024).reshape(64, 1024)
    _close("fp8 expert row 17", got[0, 17:18], expect[0, 17:18])
    _close("fp8 expert tile", got, expect)

    gate8 = rng.normal(scale=0.02, size=(1024, 2048)).astype(np.float32)
    down8 = rng.normal(scale=0.02, size=(2048, 512)).astype(np.float32)
    gate4 = rng.normal(scale=0.02, size=(1024, 2048)).astype(np.float32)
    down4 = rng.normal(scale=0.02, size=(2048, 512)).astype(np.float32)
    gate_blob = fp8_image(gate8)
    fp8_blob = gate_blob + fp8_image(down8)
    fp4_blob = fp4_image(gate4) + fp4_image(down4)
    fp8_len = len(gate_blob) == FP8_DOWN_DATA and len(fp8_blob) == EXPERT_FP8
    if not fp8_len or len(fp4_blob) != EXPERT_IMAGE:
        message = "mixed expert file is not one fp8 image plus one fp4 image"
        raise AssertionError(message)
    g8, s8, _, _ = pack_fp8_matrix(gate8)
    d8, t8, _, _ = pack_fp8_matrix(down8)
    g4, s4, _, _ = pack_fp4_matrix(gate4)
    d4, t4, _, _ = pack_fp4_matrix(down4)
    vector = rng.normal(scale=0.02, size=2048).astype(np.float32)
    fp8_gate = dequant_fp8(g8, s8, 1024, 2048)
    fp8_down = dequant_fp8(d8, t8, 2048, 512)
    fp4_gate = dequant_fp4(g4, s4, 1024, 2048)
    fp4_down = dequant_fp4(d4, t4, 2048, 512)
    fp8_row, fp8_mid = _routed_row(fp8_gate, fp8_down, vector)
    fp4_row, fp4_mid = _routed_row(fp4_gate, fp4_down, vector)
    ids = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint32)
    mix = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    table = np.zeros(256, dtype=np.uint64)
    table[1] = np.uint64(EXPERT_FP8)
    width_row = np.full(256, 4, dtype=np.uint8)
    width_row[0] = 8
    file = session.upload(np.frombuffer(fp8_blob + fp4_blob, dtype=np.uint8))
    id_buf = session.upload(ids)
    mix_buf = session.upload(mix)
    off_buf = session.upload(table)
    width_buf = session.upload(width_row)
    act_buf = session.upload(vector)
    gated = session.zeros(8 * 512 * 4)
    shared_gated = session.zeros(512 * 4)
    hidden = session.zeros(2048 * 4)
    coeff = session.upload(np.zeros(1, dtype=np.float32))
    shared_gate_w = session.zeros(2_097_152)
    shared_gate_s = session.zeros(65_536)
    shared_down_w = session.zeros(1_048_576)
    shared_down_s = session.zeros(32_768)
    dot = (32, 1, 1)
    session.begin()
    session.encode(
        "moe_gate_silu_fp8",
        [(act_buf, 0), (file, 0), (id_buf, 0), (gated, 0), (off_buf, 0), (width_buf, 0)],
        [],
        (4096, 1, 1),
        dot,
    )
    session.end()
    mid = session.download_f32(gated, 8 * 512).reshape(8, 512)
    _close("fp8 expert gate", mid[0], fp8_mid)
    _close("fp4 slot skipped by fp8 gate", mid[1], np.zeros(512, dtype=np.float32), atol=0.0)
    session.begin()
    session.encode(
        "moe_gate_silu_w",
        [
            (act_buf, 0),
            (file, 0),
            (id_buf, 0),
            (shared_gate_w, 0),
            (shared_gate_s, 0),
            (gated, 0),
            (shared_gated, 0),
            (off_buf, 0),
            (width_buf, 0),
        ],
        [],
        (4608, 1, 1),
        dot,
    )
    session.end()
    mid = session.download_f32(gated, 8 * 512).reshape(8, 512)
    _close("fp8 gate kept", mid[0], fp8_mid)
    _close("fp4 expert gate", mid[1], fp4_mid)
    session.begin()
    session.encode(
        "moe_down_mix_w",
        [
            (gated, 0),
            (shared_gated, 0),
            (file, 0),
            (id_buf, 0),
            (mix_buf, 0),
            (shared_down_w, 0),
            (shared_down_s, 0),
            (coeff, 0),
            (hidden, 0),
            (off_buf, 0),
            (width_buf, 0),
        ],
        [],
        (2048, 1, 1),
        dot,
    )
    session.encode(
        "moe_down_mix_fp8",
        [
            (gated, 0),
            (file, 0),
            (id_buf, 0),
            (mix_buf, 0),
            (hidden, 0),
            (off_buf, 0),
            (width_buf, 0),
        ],
        [],
        (2048, 1, 1),
        dot,
    )
    session.end()
    summed = (fp8_row + fp4_row).astype(np.float32)
    _close("fp8 and fp4 partials", session.download_f32(hidden, 2048), summed)


def _channel_i8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak = np.max(np.abs(matrix), axis=1)
    scale = np.where(peak > 0.0, peak / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    inv = np.zeros_like(peak)
    np.divide(np.float32(127.0), peak, out=inv, where=peak > 0.0)
    codes = np.rint(matrix * inv[:, None])
    return np.clip(codes, -127, 127).astype(np.int8), scale


def check_i8_prefill(session: Session) -> None:
    """Packed fp4 and fp8 restore into int8, then a 64-row int8 tile."""
    rng = np.random.default_rng(7)
    cases = (
        ("fp4", 64, 64, 256),
        ("fp8", 64, 64, 256),
    )
    for kind, rows, cols, inner in cases:
        activation = rng.normal(scale=0.25, size=(rows, inner)).astype(np.float16)
        activation[0, :] = np.float16(0.0)
        weight = rng.normal(scale=0.25, size=(cols, inner)).astype(np.float32)
        weight[1, :] = np.float32(0.0)
        if kind == "fp4":
            data, scales = quantize_fp4(weight)
            restored = dequant_fp4(data, scales, cols, inner)
            restore = "restore_fp4_i8"
        else:
            data, scales = quantize_fp8(weight)
            restored = dequant_fp8(data, scales, cols, inner)
            restore = "restore_fp8_i8"
        expect_codes, expect_scale = _channel_i8(restored)
        act_codes, act_scale = _channel_i8(activation.astype(np.float32))
        stage = session.zeros(cols * inner + cols * 4)
        act = session.zeros(rows * inner)
        act_factors = session.zeros(rows * 4)
        destination = session.zeros(rows * cols * 4)
        width = int(session.pipes["prefill_i8_n64"].threadExecutionWidth())
        session.begin()
        session.encode(
            restore,
            [
                (session.upload(np.frombuffer(data, dtype=np.uint8)), 0),
                (session.upload(np.frombuffer(scales, dtype=np.uint8)), 0),
                (stage, 0),
                (stage, cols * inner),
            ],
            [np.uint32(cols).tobytes(), np.uint32(inner).tobytes()],
            (cols, 1, 1),
            (256, 1, 1),
        )
        session.encode(
            "quantize_half_i8",
            [
                (session.upload(np.ascontiguousarray(activation)), 0),
                (act, 0),
                (act_factors, 0),
            ],
            [np.uint32(rows).tobytes(), np.uint32(inner).tobytes()],
            (rows, 1, 1),
            (256, 1, 1),
        )
        session.encode(
            "prefill_i8_n64",
            [(act, 0), (stage, 0), (destination, 0)],
            [
                np.uint32(rows).tobytes(),
                np.uint32(cols).tobytes(),
                np.uint32(inner).tobytes(),
            ],
            (cols // 64, rows // 64, 1),
            (width * 4, 1, 1),
        )
        session.encode(
            "scale_i8_f32",
            [(destination, 0), (act_factors, 0), (stage, cols * inner)],
            [np.uint32(rows).tobytes(), np.uint32(cols).tobytes()],
            ((rows * cols + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        session.end()
        raw = session.download(stage, cols * inner + cols * 4)
        got_codes = np.frombuffer(raw[: cols * inner], dtype=np.int8).reshape(cols, inner)
        got_scale = np.frombuffer(raw[cols * inner :], dtype=np.float32)
        _close_codes(f"{kind} int8 restore", got_codes, expect_codes)
        _close_ulp(f"{kind} int8 scale", got_scale, expect_scale)
        got_act = np.frombuffer(session.download(act, rows * inner), dtype=np.int8).reshape(
            rows, inner
        )
        got_act_scale = session.download_f32(act_factors, rows)
        _close_codes(f"{kind} int8 activation", got_act, act_codes)
        _close_ulp(f"{kind} int8 act scale", got_act_scale, act_scale)
        acc = got_act.astype(np.int64) @ got_codes.astype(np.int64).T
        expect = (
            acc
            * got_act_scale.astype(np.float64)[:, None]
            * got_scale.astype(np.float64)[None, :]
        ).astype(np.float32)
        got = session.download_f32(destination, rows * cols).reshape(rows, cols)
        _close(f"{kind} int8 tile", got, expect, rtol=1e-5, atol=1e-4)


CHECKS = (
    ("fp4 tile", check_fp4_tile),
    ("fp8 tile", check_fp8_tile),
    ("int8 prefill", check_i8_prefill),
    ("fp4 gemv", check_fp4_gemv),
    ("fp4 wide", check_wide_fp4),
    ("32-row", check_rows_32),
    ("1-row", check_one_row),
    ("router", check_router),
    ("expert", check_expert_image),
    ("vision pad", check_vision_pad),
    ("head 72", check_head72),
    ("gelu", check_gelu),
    ("deltanet", check_deltanet),
    ("gdn chunk", check_gdn_chunk),
    ("conv", check_conv),
    ("conv rows", check_conv_rows),
    ("e4m3", check_e4m3),
    ("rope", check_rope),
    ("router coeff", check_router_coeff),
    ("router8", check_router8),
    ("topk", check_topk),
    ("elementwise", check_elementwise),
    ("norms", check_norms),
    ("gather", check_gather_and_attend),
    ("expert slots", check_remap_expert_ids),
    ("moe group", check_moe_group),
    ("grouped expert", check_grouped_expert),
    ("fp8 expert", check_fp8_expert),
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
