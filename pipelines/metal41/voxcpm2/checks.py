"""Host formulas checked against one Metal dispatch."""

from __future__ import annotations

import numpy as np

from voxcpm2.block import rmsnorm, rope_angles
from voxcpm2.metal import Session


def rmsnorm_matches() -> None:
    """A float16 row matches the host RMSNorm within one ulp of the cast."""
    hidden = np.array([0.5, -1.25, 0.75, 2.0, -0.25, 1.5, 0.125, -0.875], dtype=np.float32)
    weight = np.array([1.0, 0.5, 1.5, 0.25, 1.0, 2.0, 0.75, 1.25], dtype=np.float32)
    eps = 1e-5
    hidden16 = hidden.astype(np.float16)
    weight16 = weight.astype(np.float16)
    host = rmsnorm(hidden16.astype(np.float32), weight16.astype(np.float32), eps)
    session = Session("voxcpm2:kernels")
    try:
        device = session.rmsnorm(hidden16, weight16, eps)
    finally:
        session.close()
    delta = np.max(np.abs(device.astype(np.float32) - host.astype(np.float16).astype(np.float32)))
    if delta > np.float32(1e-3):
        message = f"rmsnorm_row disagrees with the host formula by {delta}"
        raise AssertionError(message)


def linear_matches() -> None:
    """A padded half matmul matches ``x @ weight.T + bias``."""
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(2, 64)).astype(np.float32)
    weight = rng.normal(size=(4, 64)).astype(np.float32) * 0.05
    bias = rng.normal(size=(4,)).astype(np.float32) * 0.05
    cast_hidden = hidden.astype(np.float16).astype(np.float32)
    cast_weight = weight.astype(np.float16).astype(np.float32)
    host = cast_hidden @ cast_weight.T
    host = host + bias
    session = Session("voxcpm2:linear")
    try:
        device = session.linear(hidden, weight, bias)
    finally:
        session.close()
    delta = float(np.max(np.abs(device - host)))
    if delta > 2e-2:
        message = f"matmul_f16 disagrees with the host formula by {delta}"
        raise AssertionError(message)


def rope_matches() -> None:
    """Rotate-half matches the host angles on one position."""
    rng = np.random.default_rng(1)
    dim = 8
    hidden = rng.normal(size=(1, 2, dim)).astype(np.float32)
    factors = (1.0, 1.1, 0.9, 1.2)
    cos, sin = rope_angles(3, dim=dim, base=10000.0, factors=factors, scale=1.0)
    cos = np.asarray(cos, dtype=np.float32).reshape(1, dim)
    sin = np.asarray(sin, dtype=np.float32).reshape(1, dim)
    half = dim // 2
    host = np.empty_like(hidden)
    left, right = hidden[..., :half], hidden[..., half:]
    host[..., :half] = left * cos[:, None, :half] - right * sin[:, None, :half]
    host[..., half:] = right * cos[:, None, :half] + left * sin[:, None, :half]
    session = Session("voxcpm2:rope")
    try:
        device = session.rope(hidden, cos, sin)
    finally:
        session.close()
    delta = float(np.max(np.abs(device - host)))
    if delta > 1e-5:
        message = f"rope_half disagrees with the host formula by {delta}"
        raise AssertionError(message)


def attention_matches() -> None:
    """Causal grouped-query attention matches the host softmax."""
    rng = np.random.default_rng(2)
    query = rng.normal(size=(3, 4, 8)).astype(np.float32)
    key = rng.normal(size=(3, 2, 8)).astype(np.float32)
    value = rng.normal(size=(3, 2, 8)).astype(np.float32)
    host = _gqa(query, key, value, causal=True)
    session = Session("voxcpm2:attention")
    try:
        device = session.attention(query, key, value, causal=True)
    finally:
        session.close()
    delta = float(np.max(np.abs(device - host)))
    if delta > 1e-4:
        message = f"gqa_attention disagrees with the host formula by {delta}"
        raise AssertionError(message)


def conv_matches() -> None:
    """A causal conv and a causal transpose match the host loops."""
    rng = np.random.default_rng(3)
    source = rng.normal(size=(2, 5)).astype(np.float32)
    weight = rng.normal(size=(3, 2, 3)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)
    host = _conv(source, weight, bias, left=2, dilation=1, groups=1)
    session = Session("voxcpm2:conv")
    try:
        device = session.conv1d(source, weight, bias, left=2, dilation=1, groups=1)
    finally:
        session.close()
    delta = float(np.max(np.abs(device - host)))
    if delta > 1e-4:
        message = f"causal_conv1d disagrees with the host formula by {delta}"
        raise AssertionError(message)
    again = np.random.default_rng(4)
    src = again.normal(size=(2, 3)).astype(np.float32)
    kern = again.normal(size=(2, 3, 4)).astype(np.float32)
    shift = again.normal(size=(3,)).astype(np.float32)
    expect = _transpose(src, kern, shift, stride=2)
    session = Session("voxcpm2:transpose")
    try:
        got = session.conv_transpose1d(src, kern, shift, stride=2)
    finally:
        session.close()
    delta = float(np.max(np.abs(got - expect)))
    if delta > 1e-4:
        message = f"causal_transpose1d disagrees with the host formula by {delta}"
        raise AssertionError(message)


def _gqa(query: np.ndarray, key: np.ndarray, value: np.ndarray, *, causal: bool) -> np.ndarray:
    q_len, heads, dim = query.shape
    kv_len, kv_heads, _ = key.shape
    group = heads // kv_heads
    scale = 1.0 / np.sqrt(dim)
    out = np.zeros_like(query)
    for qi in range(q_len):
        limit = kv_len if not causal else (kv_len - q_len + qi + 1)
        for head in range(heads):
            kv = head // group
            scores = np.array([np.dot(query[qi, head], key[t, kv]) * scale for t in range(limit)])
            scores = scores - np.max(scores)
            weights = np.exp(scores)
            weights /= np.sum(weights)
            acc = np.zeros(dim, dtype=np.float32)
            for t in range(limit):
                acc += weights[t] * value[t, kv]
            out[qi, head] = acc
    return out


def _conv(
    source: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    left: int,
    dilation: int,
    groups: int,
) -> np.ndarray:
    cin, time = source.shape
    cout, cin_g, width = weight.shape
    out = np.zeros((cout, time), dtype=np.float32)
    cout_g = cout // groups
    for oc in range(cout):
        group = oc // cout_g
        for t in range(time):
            acc = float(bias[oc])
            for ic in range(cin_g):
                for tap in range(width):
                    index = t + tap * dilation - left
                    if 0 <= index < time:
                        acc += float(source[group * cin_g + ic, index]) * float(weight[oc, ic, tap])
            out[oc, t] = acc
    del cin
    return out


def _transpose(
    source: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
) -> np.ndarray:
    cin, tin = source.shape
    _, cout, width = weight.shape
    tout = tin * stride
    out = np.zeros((cout, tout), dtype=np.float32)
    for oc in range(cout):
        for t in range(tout):
            acc = float(bias[oc])
            for ic in range(cin):
                for tap in range(width):
                    if t < tap or (t - tap) % stride != 0:
                        continue
                    i = (t - tap) // stride
                    if i < tin:
                        acc += float(source[ic, i]) * float(weight[ic, oc, tap])
            out[oc, t] = acc
    return out
