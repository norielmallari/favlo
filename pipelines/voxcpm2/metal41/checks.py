"""Host formulas checked against one Metal dispatch."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from voxcpm2.metal41.block import residual_add, rmsnorm, rope_angles, silu, swiglu
from voxcpm2.metal41.metal import Session
from voxcpm2.metal41.stack import Cache, Stack, _angles
from voxcpm2.metal41.store import Store


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


def activation_matches() -> None:
    """SiLU, SwiGLU, the residual add, and a row copy match the host formulas.

    One command buffer also runs a half matmul, the bias add, SiLU, and a residual.
    """
    rng = np.random.default_rng(6)
    gate = rng.normal(size=(2, 8)).astype(np.float32)
    up = rng.normal(size=(2, 8)).astype(np.float32)
    session = Session("voxcpm2:act")
    try:
        _silu_matches(session, gate)
        _swiglu_matches(session, gate, up)
        _add_matches(session, gate, up)
        _copy_matches(session)
        _norm_matches(session, rng)
        _chain_matches(session, rng)
    finally:
        session.close()


def stack_matches() -> None:
    """One tower step in a single command buffer matches the sync kernels."""
    rng = np.random.default_rng(7)
    hidden = 32
    heads = 4
    kv_heads = 2
    dim = 8
    ffn = 64
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
    path = Path(handle.name)
    handle.close()
    _save_bf16(path, _tower_tensors(rng, hidden, heads, kv_heads, dim, ffn))
    session = Session("voxcpm2:stack")
    store: Store | None = None
    try:
        store = Store(path)
        stack = Stack(
            session,
            store,
            "tower.layers",
            "tower.norm.weight",
            layers=1,
            hidden=hidden,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=dim,
            eps=1e-5,
            rope=True,
            theta=10000.0,
            factors=(1.0, 1.1, 0.9, 1.2),
            scale=1.0,
        )
        host = _HostCache(kv_heads, dim)
        device = Cache(1, kv_heads, dim, capacity=4)
        prefill = rng.normal(size=(3, hidden)).astype(np.float32)
        _near(
            "stack prefill",
            stack.forward(prefill, np.arange(3), True, device),
            _sync_forward(stack, prefill, np.arange(3), True, host),
        )
        step = rng.normal(size=(hidden,)).astype(np.float32)
        _near(
            "stack decode",
            stack.forward(step, np.array([3]), True, device),
            _sync_forward(stack, step, np.array([3]), True, host),
        )
        if device.length != 4:
            message = f"the cache length is {device.length}"
            raise AssertionError(message)
        context = rng.normal(size=(4, hidden)).astype(np.float32)
        _near(
            "stack context",
            stack.forward(context, np.arange(4), False, Cache(1, kv_heads, dim)),
            _sync_forward(stack, context, np.arange(4), False, _HostCache(kv_heads, dim)),
        )
    finally:
        if store is not None:
            store.close()
        session.close()
        path.unlink(missing_ok=True)


def _silu_matches(session: Session, gate: np.ndarray) -> None:
    src = session.load(gate)
    dst = session.plane(int(gate.shape[0]), int(gate.shape[1]))
    try:
        session.silu(src, dst)
        got = session.read(dst)
    finally:
        session.release_all([src, dst])
    _near("silu_f32", got, silu(gate), tol=1e-5)


def _swiglu_matches(session: Session, gate: np.ndarray, up: np.ndarray) -> None:
    left = session.load(gate)
    right = session.load(up)
    try:
        session.swiglu(left, right, left)
        got = session.read(left)
    finally:
        session.release_all([left, right])
    _near("swiglu_f32", got, swiglu(gate, up), tol=1e-5)


def _add_matches(session: Session, left: np.ndarray, right: np.ndarray) -> None:
    src = session.load(left)
    other = session.load(right)
    try:
        session.add(src, other, src)
        got = session.read(src)
    finally:
        session.release_all([src, other])
    _near("add_f32", got, left + right, tol=1e-6)


def _copy_matches(session: Session) -> None:
    raw = np.arange(8, dtype=np.float32).reshape(2, 4)
    src = session.load(raw)
    dst = session.plane(4, 4, zero=True)
    try:
        session.copy(src, dst, 2, 0, 1)
        got = session.read(dst)
    finally:
        session.release_all([src, dst])
    expect = np.zeros((4, 4), dtype=np.float32)
    expect[1:3] = raw
    _near("copy_rows", got, expect, tol=0.0)


def _norm_matches(session: Session, rng: np.random.Generator) -> None:
    hidden = rng.normal(size=(2, 8)).astype(np.float32)
    weight = rng.normal(size=(8,)).astype(np.float32) * 0.25 + 1.0
    eps = 1e-5
    session.prepare_linear(2, 8, 8, 0)
    gain = session.upload(np.ascontiguousarray(weight, dtype=np.float16).tobytes())
    src = session.load(hidden)
    dst = session.plane(2, 8)
    try:
        session.norm(src, gain, dst, eps)
        got = session.read(dst)
    finally:
        session.release_all([src, dst], [gain])
    cast = hidden.astype(np.float16).astype(np.float32)
    scale = weight.astype(np.float16).astype(np.float32)
    host = rmsnorm(cast, scale, eps).astype(np.float16).astype(np.float32)
    _near("cast_f32_f16", got, host, tol=1e-3)


def _chain_matches(session: Session, rng: np.random.Generator) -> None:
    hidden = rng.normal(size=(2, 64)).astype(np.float32)
    weight = rng.normal(size=(4, 64)).astype(np.float32) * 0.05
    bias = rng.normal(size=(4,)).astype(np.float32) * 0.05
    residual = rng.normal(size=(2, 4)).astype(np.float32) * 0.05
    session.prepare_linear(2, 64, 4, 0)
    resident = session.resident(weight)
    shift = session.upload(np.ascontiguousarray(bias, dtype=np.float32).tobytes())
    src = session.load(hidden)
    res = session.load(residual)
    proj = session.plane(2, 4)
    act = session.plane(2, 4)
    out = session.plane(2, 4)
    try:
        session.project(src, resident, shift, proj)
        session.silu(proj, act)
        session.add(act, res, out)
        got = session.read(out)
    finally:
        session.release_all([src, res, proj, act, out], [shift, resident.handle])
    cast_h = hidden.astype(np.float16).astype(np.float32)
    cast_w = weight.astype(np.float16).astype(np.float32)
    host = silu(cast_h @ cast_w.T + bias) + residual
    _near("bias_pack", got, host, tol=2e-2)


def _near(name: str, got: np.ndarray, expect: np.ndarray, tol: float = 1e-3) -> None:
    delta = float(np.max(np.abs(got - expect)))
    if delta > tol:
        message = f"{name} disagrees with the host formula by {delta}"
        raise AssertionError(message)


class _HostCache:
    def __init__(self, kv_heads: int, dim: int) -> None:
        self.k = [np.zeros((0, kv_heads, dim), dtype=np.float32)]
        self.v = [np.zeros((0, kv_heads, dim), dtype=np.float32)]


def _sync_forward(
    stack: Stack,
    hidden: np.ndarray,
    positions: np.ndarray,
    causal: bool,
    cache: _HostCache,
) -> np.ndarray:
    data = np.asarray(hidden, dtype=np.float32)
    squeeze = data.ndim == 1
    if squeeze:
        data = data.reshape(1, -1)
    cos = sin = None
    if stack.rope:
        cos, sin = _angles(positions, stack.head_dim, stack.theta, stack.factors, stack.scale)
    for layer in range(stack.layers):
        data = _sync_layer(stack, layer, data, causal, cache, cos, sin)
    data = _sync_norm(stack, data, stack.norm)
    if squeeze:
        return data.reshape(-1)
    return data


def _sync_layer(
    stack: Stack,
    layer: int,
    data: np.ndarray,
    causal: bool,
    cache: _HostCache,
    cos: np.ndarray | None,
    sin: np.ndarray | None,
) -> np.ndarray:
    seq = int(data.shape[0])
    stem = f"{stack.prefix}.{layer}"
    session = stack.session
    store = stack.store
    normed = _sync_norm(stack, data, f"{stem}.input_layernorm.weight")
    query = store.linear(session, f"{stem}.self_attn.q_proj.weight", normed)
    key = store.linear(session, f"{stem}.self_attn.k_proj.weight", normed)
    value = store.linear(session, f"{stem}.self_attn.v_proj.weight", normed)
    query = query.reshape(seq, stack.heads, stack.head_dim)
    key = key.reshape(seq, stack.kv_heads, stack.head_dim)
    value = value.reshape(seq, stack.kv_heads, stack.head_dim)
    if cos is not None and sin is not None:
        query = session.rope(query, cos, sin)
        key = session.rope(key, cos, sin)
    cache.k[0] = np.concatenate((cache.k[0], key), axis=0)
    cache.v[0] = np.concatenate((cache.v[0], value), axis=0)
    mixed = session.attention(query, cache.k[0], cache.v[0], causal=causal)
    mixed = mixed.reshape(seq, stack.heads * stack.head_dim)
    update = store.linear(session, f"{stem}.self_attn.o_proj.weight", mixed)
    data = residual_add(data, update, use_mup=False, scale_depth=1.0, layers=stack.layers)
    normed = _sync_norm(stack, data, f"{stem}.post_attention_layernorm.weight")
    gate = store.linear(session, f"{stem}.mlp.gate_proj.weight", normed)
    up = store.linear(session, f"{stem}.mlp.up_proj.weight", normed)
    down = store.linear(session, f"{stem}.mlp.down_proj.weight", swiglu(gate, up))
    return residual_add(data, down, use_mup=False, scale_depth=1.0, layers=stack.layers)


def _sync_norm(stack: Stack, hidden: np.ndarray, name: str) -> np.ndarray:
    return stack.session.rmsnorm(hidden, stack.store.vector(name), stack.eps).astype(np.float32)


def _tower_tensors(
    rng: np.random.Generator,
    hidden: int,
    heads: int,
    kv_heads: int,
    dim: int,
    ffn: int,
) -> dict[str, np.ndarray]:
    def matrix(rows: int, cols: int) -> np.ndarray:
        return (rng.normal(size=(rows, cols)) * 0.05).astype(np.float32)

    def gain(width: int) -> np.ndarray:
        return (rng.normal(size=(width,)) * 0.1 + 1.0).astype(np.float32)

    prefix = "tower.layers.0"
    q = heads * dim
    kv = kv_heads * dim
    return {
        f"{prefix}.self_attn.q_proj.weight": matrix(q, hidden),
        f"{prefix}.self_attn.k_proj.weight": matrix(kv, hidden),
        f"{prefix}.self_attn.v_proj.weight": matrix(kv, hidden),
        f"{prefix}.self_attn.o_proj.weight": matrix(hidden, q),
        f"{prefix}.mlp.gate_proj.weight": matrix(ffn, hidden),
        f"{prefix}.mlp.up_proj.weight": matrix(ffn, hidden),
        f"{prefix}.mlp.down_proj.weight": matrix(hidden, ffn),
        f"{prefix}.input_layernorm.weight": gain(hidden),
        f"{prefix}.post_attention_layernorm.weight": gain(hidden),
        "tower.norm.weight": gain(hidden),
    }


def _save_bf16(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict[str, object]] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, value in tensors.items():
        raw = _bf16(value)
        header[name] = {
            "dtype": "BF16",
            "shape": [int(dim) for dim in value.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"".join(blobs))


def _bf16(array: np.ndarray) -> bytes:
    values = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return (values >> 16).astype(np.uint16).tobytes()


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
