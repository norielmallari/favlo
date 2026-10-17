"""Host formulas and the dispatches that have to match them."""

from __future__ import annotations

import numpy as np

from qwen36_35b_a3b.formats import (
    EXPERT_IMAGE,
    dequant_e4m3_block,
    dequant_fp4,
    fp4_image,
    pack_fp4_matrix,
    quantize_fp4,
)
from qwen36_35b_a3b.session import Session

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
    norm = np.zeros(128, dtype=np.float16)
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
        out[head] = y * inv * silu
    got = session.download_f32(destination, 32 * 128).reshape(32, 128)
    _close("deltanet", got, out, atol=1e-4)


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
        i0 = pair * 2
        x0 = float(expect[i0])
        x1 = float(expect[i0 + 1])
        expect[i0] = x0 * c - x1 * s
        expect[i0 + 1] = x0 * s + x1 * c
    _close("rope", got[:64], expect[:64].astype(np.float32), atol=1e-5)
    _close("rope tail", got[64:], hidden[64:], atol=0.0)


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


CHECKS = (
    ("fp4 tile", check_fp4_tile),
    ("fp8 tile", check_fp8_tile),
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
    ("conv", check_conv),
    ("e4m3", check_e4m3),
    ("rope", check_rope),
    ("topk", check_topk),
    ("elementwise", check_elementwise),
    ("norms", check_norms),
    ("gather", check_gather_and_attend),
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
