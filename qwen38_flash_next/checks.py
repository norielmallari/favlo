"""Host formulas the kernel checks compare against."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from qwen38_flash_next.formats import quantize_matrix
from qwen38_flash_next.session import MetalSession, _buffer_bytes


def check_state_kernels() -> None:
    """GDN delta and grouped-query attention match the tech-report recurrences on one step."""
    session = MetalSession()
    rng = np.random.default_rng(0)
    state = rng.normal(size=(48, 128, 128)).astype(np.float32)
    query = rng.normal(size=(48, 128)).astype(np.float32)
    key = rng.normal(size=(48, 128)).astype(np.float32)
    value = rng.normal(size=(48, 128)).astype(np.float32)
    alpha = rng.uniform(0.2, 0.9, size=48).astype(np.float32)
    beta = rng.uniform(0.1, 0.8, size=48).astype(np.float32)
    buffer = session.upload(np.ascontiguousarray(state).tobytes())
    got = session.gdn(buffer, query, key, value, alpha, beta)
    raw = _buffer_bytes(buffer, state.nbytes)
    updated = np.frombuffer(raw, dtype=np.float32).reshape(state.shape)
    tilde = alpha[:, None, None] * state
    error = value - np.einsum("hdv,hd->hv", tilde, key)
    expect_state = tilde + beta[:, None, None] * key[:, :, None] * error[:, None, :]
    expect_y = np.einsum("hdv,hd->hv", expect_state, query).reshape(-1)
    if not np.allclose(updated, expect_state, rtol=1e-4, atol=1e-4):
        message = "GDN state shader diverged"
        raise RuntimeError(message)
    if not np.allclose(got, expect_y, rtol=1e-4, atol=1e-4):
        message = "GDN output shader diverged"
        raise RuntimeError(message)
    tokens = 5
    q = rng.normal(size=(24, 256)).astype(np.float32)
    k = rng.normal(size=(tokens, 2, 256)).astype(np.float32)
    v = rng.normal(size=(tokens, 2, 256)).astype(np.float32)
    attended = session.attend(q, k, v).reshape(24, 256)
    scale = 1.0 / math.sqrt(256.0)
    expect = np.empty_like(q)
    for head in range(24):
        kv = head // 12
        scores = k[:, kv] @ q[head] * scale
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        expect[head] = probs @ v[:, kv]
    if not np.allclose(attended, expect, rtol=1e-4, atol=1e-4):
        message = "attention shader diverged"
        raise RuntimeError(message)


def _rms(
    values: np.ndarray, weight: np.ndarray, eps: float, *, zero_centered: bool = True
) -> np.ndarray:
    variance = np.mean(np.square(values), axis=-1, keepdims=True)
    # Hyper-connection norms are Gemma-style offsets. The GDN head norm is a
    # multiplicative gain, initialized at one, so the stored value is the gain.
    gain = (1.0 + weight) if zero_centered else weight
    return values * np.reciprocal(np.sqrt(variance + eps)) * gain


def _silu(values: np.ndarray) -> np.ndarray:
    # A large negative input overflows exp and saturates to zero, which is SiLU.
    with np.errstate(over="ignore"):
        return values / (1.0 + np.exp(-values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # A large negative input overflows exp and saturates to zero, which is sigmoid.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-values))


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _l2norm(values: np.ndarray, eps: float) -> np.ndarray:
    return values / np.sqrt(np.sum(np.square(values), axis=-1, keepdims=True) + eps)


def _rope(values: np.ndarray, position: int, rotary_dim: int, theta: float) -> np.ndarray:
    half = rotary_dim // 2
    freqs = np.float64(theta) ** (-np.arange(half, dtype=np.float64) / half)
    angles = position * freqs
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    out = values.copy()
    first = values[:, :half]
    second = values[:, half:rotary_dim]
    out[:, :half] = first * cos - second * sin
    out[:, half:rotary_dim] = first * sin + second * cos
    return out


def _causal_conv(history: list[np.ndarray], current: np.ndarray, weight: np.ndarray) -> np.ndarray:
    kernel = weight.shape[1]
    frames = [*history, current]
    if len(frames) < kernel:
        zero = np.zeros_like(current)
        frames = [zero] * (kernel - len(frames)) + frames
    mixed = np.zeros_like(current)
    for tap, frame in enumerate(frames[-kernel:]):
        mixed += weight[:, tap] * frame
    history.append(current)
    del history[:-kernel + 1]
    return mixed


def _sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng: Any) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scores = logits.astype(np.float64) / temperature
    if 0 < top_k < scores.size:
        cutoff = np.partition(scores, -top_k)[-top_k]
        scores = np.where(scores < cutoff, -np.inf, scores)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    total = float(probs.sum())
    if not math.isfinite(total) or total <= 0.0:
        return int(np.argmax(logits))
    probs /= total
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        remove = cumulative > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        probs[order[remove]] = 0.0
        probs /= probs.sum()
    return int(rng.choice(probs.size, p=probs))


_I64_MASK = (1 << 64) - 1
_I64_SIGN = 1 << 63


def _earlier_token(tokens: Sequence[int], shift: int, eos: int) -> int:
    """Token `shift` steps before the last one.

    Shift 0 is the current token, kept even when it is EOS. A lookback that
    starts before this segment, or that crosses an earlier EOS, is the EOS id.
    """
    if shift == 0:
        return int(tokens[-1])
    last = len(tokens) - 1
    segment_start = 0
    for index, item in enumerate(tokens[:-1]):
        if int(item) == eos:
            segment_start = index + 1
    if last - segment_start < shift:
        return eos
    return int(tokens[last - shift])


def _ngram_row(
    tokens: Sequence[int],
    head: int,
    multipliers: np.ndarray,
    offsets: np.ndarray,
    vocabs: np.ndarray,
    *,
    heads_per_ngram: int,
    ngram_size: int,
    eos: int,
) -> int:
    """Bigram heads, then trigram heads. Wrapping int64 XOR, positive remainder.

    Shift 0 is the current token and uses the first multiplier. Each older
    token uses the next multiplier. A cleared lookback contributes the EOS id.
    """
    order = 2 + head // heads_per_ngram
    mults = [int(value) for value in np.reshape(multipliers, -1).tolist()]
    mixed = 0
    for shift in range(order):
        product = (_earlier_token(tokens, shift, eos) * mults[shift]) & _I64_MASK
        mixed ^= product
    signed = mixed - (1 << 64) if mixed >= _I64_SIGN else mixed
    return int(offsets[head]) + signed % int(vocabs[head])


def _dilated_conv(
    history: list[np.ndarray], current: np.ndarray, weight: np.ndarray, dilation: int
) -> np.ndarray:
    """Causal depthwise conv. Weight tap 0 is the oldest sample."""
    kernel = int(weight.shape[1])
    span = (kernel - 1) * dilation
    frames = [*history, current]
    if len(frames) < span + 1:
        zero = np.zeros_like(current)
        frames = [zero] * (span + 1 - len(frames)) + frames
    window = frames[-(span + 1) :]
    mixed = np.zeros_like(current)
    for tap in range(kernel):
        mixed += weight[:, tap] * window[tap * dilation]
    history.append(np.array(current, copy=True))
    del history[:-span]
    return mixed


def _close(name: str, got: np.ndarray, expect: np.ndarray) -> None:
    if np.allclose(got, expect, rtol=1e-4, atol=1e-4):
        return
    delta = float(np.max(np.abs(got.astype(np.float64) - expect.astype(np.float64))))
    message = f"{name} shader diverged by {delta}"
    raise RuntimeError(message)


def check_act_kernels() -> None:
    """Elementwise and reduction shaders match the host formulas on one draw."""
    gpu = MetalSession()
    rng = np.random.default_rng(1)
    values = rng.normal(size=128).astype(np.float32)
    src = gpu.dev_from(values)
    _close("silu", gpu.read_f32(gpu.silu(src)), _silu(values))
    _close("sigmoid", gpu.read_f32(gpu.sigmoid(src)), _sigmoid(values))
    _close("softplus", gpu.read_f32(gpu.softplus(src)), _softplus(values))
    _close("scale", gpu.read_f32(gpu.scale(src, 0.25)), values * np.float32(0.25))
    other = rng.normal(size=128).astype(np.float32)
    _close("mul", gpu.read_f32(gpu.mul(src, gpu.dev_from(other))), values * other)
    rows, cols = 4, 32
    matrix = rng.normal(size=(rows, cols)).astype(np.float32)
    gain = rng.normal(size=(rows, cols)).astype(np.float32)
    weight = gpu.upload(np.ascontiguousarray(gain).tobytes())
    got = gpu.rms(gpu.dev_from(matrix), weight, 0, cols, rows, 1e-6, True, rows)
    _close("rms", gpu.read_f32(got).reshape(rows, cols), _rms(matrix, gain, 1e-6))
    shared = rng.normal(size=cols).astype(np.float32)
    shared_buf = gpu.upload(shared.tobytes())
    got = gpu.rms(gpu.dev_from(matrix), shared_buf, 0, cols, rows, 1e-6, False, 0)
    flat = _rms(matrix, shared, 1e-6, zero_centered=False)
    _close("rms-gain", gpu.read_f32(got).reshape(rows, cols), flat)
    heads = rng.normal(size=(16, 8)).astype(np.float32)
    got = gpu.l2(gpu.dev_from(heads), 16, 8, 1e-6)
    _close("l2", gpu.read_f32(got).reshape(16, 8), _l2norm(heads, 1e-6))
    spun = rng.normal(size=(2, 16)).astype(np.float32)
    got = gpu.rope(gpu.dev_from(spun), 2, 16, 8, 3, 1.0e4)
    _close("rope", gpu.read_f32(got).reshape(2, 16), _rope(spun, 3, 8, 1.0e4))
    channels, taps = 8, 4
    conv_w = rng.normal(size=(channels, taps)).astype(np.float32)
    conv_buf = gpu.upload(np.ascontiguousarray(conv_w).tobytes())
    history = gpu.zeros((taps - 1) * channels * 4)
    host_hist: list[np.ndarray] = []
    for _ in range(3):
        current = rng.normal(size=channels).astype(np.float32)
        expect = _causal_conv(host_hist, current, conv_w)
        got = gpu.causal_conv(history, gpu.dev_from(current), conv_buf, 0, channels, taps)
        _close("conv", gpu.read_f32(got), expect)
    dilate = 3
    span = (taps - 1) * dilate
    dilate_hist = gpu.zeros(span * channels * 4)
    host_dilate: list[np.ndarray] = []
    for _ in range(4):
        current = rng.normal(size=channels).astype(np.float32)
        expect = _dilated_conv(host_dilate, current, conv_w, dilate)
        got = gpu.dilated_conv(
            dilate_hist, gpu.dev_from(current), conv_buf, 0, channels, taps, dilate
        )
        _close("dilated", gpu.read_f32(got), expect)
    logits = np.arange(64, dtype=np.float32)
    index, chosen = gpu.topk(gpu.dev_from(logits), 64, 4)
    ids = gpu.read_u32(index)
    weights = gpu.read_f32(chosen)
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs /= probs.sum()
    order = np.argsort(probs)[::-1][:4]
    if not np.array_equal(ids, order.astype(np.uint32)):
        message = f"top-k ids {ids} expected {order}"
        raise RuntimeError(message)
    _close("top-k", weights, (probs[order] / probs[order].sum()).astype(np.float32))
    wide = rng.normal(size=1000).astype(np.float32)
    if gpu.argmax(gpu.dev_from(wide)) != int(np.argmax(wide)):
        message = "argmax shader diverged"
        raise RuntimeError(message)
    gate = rng.normal(size=(2, 4, 8)).astype(np.float32)
    hats = rng.normal(size=(2, 4, 8)).astype(np.float32)
    got = gpu.branch_mean(gpu.dev_from(gate), gpu.dev_from(hats), 2, 4, 8)
    _close("mean", gpu.read_f32(got), np.mean(gate * hats, axis=1).reshape(-1))
    weight_m = rng.normal(size=(32, 64)).astype(np.float32)
    packed, scales = quantize_matrix(weight_m)
    loaded = gpu.load_int4(packed + scales, 32, 64, 64)
    vector = rng.normal(size=64).astype(np.float32)
    expect = gpu.int4(loaded, vector)
    got = gpu.linear(loaded, gpu.dev_from(vector), 1, 64)
    _close("gemv", gpu.read_f32(got), expect)
    batch = rng.normal(size=(2, 64)).astype(np.float32)
    expect = gpu.int4_rows(loaded, batch)
    got = gpu.linear(loaded, gpu.dev_from(batch), 2, 64)
    _close("gemm", gpu.read_f32(got), expect.reshape(-1))


def _topk(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    probs = np.exp(shifted)
    probs /= probs.sum()
    choice = np.argpartition(probs, -k)[-k:]
    choice = choice[np.argsort(probs[choice])[::-1]]
    weight = probs[choice]
    weight = weight / weight.sum()
    return choice, weight.astype(np.float32)
