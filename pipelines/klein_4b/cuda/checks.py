"""Compile the step library and compare it with the host formulas."""

from __future__ import annotations

import numpy as np

from klein_4b.cuda import engine
from klein_4b.cuda.formats import pack_int4, restore_int4
from klein_4b.cuda.ops import Ops
from klein_4b.cuda.rope import apply_dit, apply_text, dit_table, latent_ids, text_table
from klein_4b.cuda.store import Tensor


def run() -> None:
    engine.ensure()
    ops = Ops()
    try:
        _gemm(ops)
        _nibble()
        _layernorm(ops)
        _swiglu(ops)
        _rope(ops)
        _groupnorm(ops)
        _euler(ops)
        _attend(ops)
        _conv(ops)
    finally:
        ops.close()
    print("klein_4b kernels ok")


def _gemm(ops: Ops) -> None:
    rng = np.random.default_rng(0)
    rows, cols, inner = 96, 80, 160
    act = rng.standard_normal((rows, inner)).astype(np.float16)
    weight = rng.standard_normal((cols, inner)).astype(np.float32) * np.float32(0.05)
    codes, scales, logical_rows, logical_inner = pack_int4(weight)
    restored = restore_int4(codes, scales)[:logical_rows, :logical_inner]
    expected = act.astype(np.float32) @ restored.T
    device_act = ops.upload_f16(act, rows, inner)
    tensor = _tensor(codes, scales, logical_rows, logical_inner, codes.shape[1] * 2)
    out = ops.linear(device_act, tensor)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(device_act, out, tensor.codes, tensor.scales)
    _close(got, expected, "gemm")


def _nibble() -> None:
    weight = np.zeros((128, 128), dtype=np.float32)
    weight[0, 0] = 7.0
    weight[0, 1] = -7.0
    codes, scales, _rows, _inner = pack_int4(weight)
    restored = restore_int4(codes, scales)
    if int(codes[0, 0] & 0x0F) == 0 or int(codes[0, 0] >> 4) == 0:
        message = "int4 nibbles were not packed"
        raise RuntimeError(message)
    if abs(float(restored[0, 0]) - 7.0) > 1e-3 or abs(float(restored[0, 1]) + 7.0) > 1e-3:
        message = "int4 nibble dequant does not match the row"
        raise RuntimeError(message)


def _layernorm(ops: Ops) -> None:
    rng = np.random.default_rng(1)
    values = rng.standard_normal((4, 32)).astype(np.float16).astype(np.float32)
    device = ops.upload_f16(values, 4, 32)
    out = ops.layernorm(device, 1e-6)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(device, out)
    mean = values.mean(axis=1, keepdims=True)
    var = values.var(axis=1, keepdims=True)
    expected = (values - mean) / np.sqrt(var + 1e-6)
    _close(got, expected, "layernorm")


def _swiglu(ops: Ops) -> None:
    rng = np.random.default_rng(2)
    values = rng.standard_normal((2, 32)).astype(np.float16).astype(np.float32)
    device = ops.upload_f16(values, 2, 32)
    out = ops.swiglu(device)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(device, out)
    gate, up = np.split(values, 2, axis=1)
    expected = (gate / (1.0 + np.exp(-gate))) * up
    _close(got, expected, "swiglu")


def _rope(ops: Ops) -> None:
    cos, sin = text_table(3)
    values = np.random.default_rng(3).standard_normal((3, 128)).astype(np.float16).astype(np.float32)
    device = ops.upload_f16(values, 3, 128)
    uploaded_cos = ops.upload_f32(cos)
    uploaded_sin = ops.upload_f32(sin)
    out = ops.rope(device, 3, 1, 128, uploaded_cos, uploaded_sin, text=True)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(device, out, uploaded_cos, uploaded_sin)
    _close(got, apply_text(values, cos, sin), "rope-text")

    ids = latent_ids(2, 2, 0.0)
    dit_cos, dit_sin = dit_table(ids)
    dit_values = np.random.default_rng(4).standard_normal((4, 128)).astype(np.float16).astype(np.float32)
    device = ops.upload_f16(dit_values, 4, 128)
    uploaded_cos = ops.upload_f32(dit_cos)
    uploaded_sin = ops.upload_f32(dit_sin)
    out = ops.rope(device, 4, 1, 128, uploaded_cos, uploaded_sin, text=False)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(device, out, uploaded_cos, uploaded_sin)
    _close(got, apply_dit(dit_values, dit_cos, dit_sin), "rope-dit")


def _groupnorm(ops: Ops) -> None:
    rng = np.random.default_rng(5)
    channels, height, width = 32, 2, 2
    values = rng.standard_normal((channels, height, width)).astype(np.float32)
    weight = rng.standard_normal(channels).astype(np.float32)
    bias = rng.standard_normal(channels).astype(np.float32)
    volume = ops.upload_volume(values, fp32=True)
    w_ptr = ops.upload_f32(weight)
    b_ptr = ops.upload_f32(bias)
    tensor = Tensor(kind="f16", ptr=w_ptr, bias=b_ptr, rows=channels)
    # The kernel reads fp16 affine terms. Re-upload them as fp16.
    ops.free(w_ptr, b_ptr)
    w_plane = ops.upload_f16(weight, 1, channels)
    b_plane = ops.upload_f16(bias, 1, channels)
    tensor = Tensor(kind="f16", ptr=w_plane.ptr, bias=b_plane.ptr, rows=channels)
    out = ops.groupnorm(volume, tensor, groups=32, eps=1e-6, silu=False)
    got = ops.download_volume(out)
    ops.free(volume, out, w_plane, b_plane)
    view = values.reshape(32, -1)
    mean = view.mean(axis=1)
    var = view.var(axis=1)
    normed = ((view - mean[:, None]) * (1.0 / np.sqrt(var[:, None] + 1e-6))).reshape(values.shape)
    scale = weight.astype(np.float16).astype(np.float32)
    shift = bias.astype(np.float16).astype(np.float32)
    expected = normed * scale[:, None, None] + shift[:, None, None]
    _close(got, expected, "groupnorm")


def _euler(ops: Ops) -> None:
    latent = np.full((2, 4), 0.25, dtype=np.float32)
    velocity = np.full((2, 4), -0.5, dtype=np.float32)
    left = ops.upload_f16(latent, 2, 4)
    right = ops.upload_f16(velocity, 2, 4)
    out = ops.euler(left, right, -0.25)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(left, right, out)
    _close(got, latent + (-0.25) * velocity, "euler")


def _attend(ops: Ops) -> None:
    _attend_case(ops, seq=40, heads=4, kv_heads=2, dim=128, causal=1, valid=28)
    _attend_case(ops, seq=6, heads=1, kv_heads=1, dim=512, causal=0, valid=6)


def _attend_case(ops: Ops, seq: int, heads: int, kv_heads: int, dim: int, causal: int, valid: int) -> None:
    rng = np.random.default_rng(dim + seq)
    query = rng.standard_normal((seq, heads * dim)).astype(np.float16)
    key = rng.standard_normal((seq, kv_heads * dim)).astype(np.float16)
    value = rng.standard_normal((seq, kv_heads * dim)).astype(np.float16)
    q = ops.upload_f16(query, seq, heads * dim)
    k = ops.upload_f16(key, seq, kv_heads * dim)
    v = ops.upload_f16(value, seq, kv_heads * dim)
    out = ops.attend(q, k, v, seq, heads, kv_heads, dim, causal, valid)
    got = ops.download_f16(out).astype(np.float32)
    ops.free(q, k, v, out)
    expected = _attend_host(query, key, value, heads, kv_heads, dim, causal, valid)
    _close(got, expected, f"attend dim {dim}")


def _attend_host(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    heads: int,
    kv_heads: int,
    dim: int,
    causal: int,
    valid: int,
) -> np.ndarray:
    seq = query.shape[0]
    scale = float(dim) ** -0.5
    q = query.reshape(seq, heads, dim).astype(np.float32)
    k = key.reshape(seq, kv_heads, dim).astype(np.float32)
    v = value.reshape(seq, kv_heads, dim).astype(np.float32)
    out = np.zeros((seq, heads, dim), dtype=np.float32)
    repeat = heads // kv_heads
    for head in range(heads):
        kv = head // repeat
        for row in range(seq):
            weights = []
            chosen = []
            for col in range(seq):
                if col < valid and (not causal or col <= row):
                    weights.append(float(np.dot(q[row, head], k[col, kv]) * scale))
                    chosen.append(col)
            if not chosen:
                continue
            peak = max(weights)
            coeff = np.exp(np.asarray(weights, dtype=np.float32) - np.float32(peak))
            coeff /= coeff.sum()
            mixed = np.zeros(dim, dtype=np.float32)
            for index, col in enumerate(chosen):
                mixed += coeff[index] * v[col, kv]
            out[row, head] = mixed
    return out.reshape(seq, heads * dim)


def _conv(ops: Ops) -> None:
    rng = np.random.default_rng(4)
    volume = (rng.standard_normal((4, 68, 68)).astype(np.float32) * np.float32(0.2)).astype(np.float16).astype(np.float32)
    filters = rng.standard_normal((3, 4, 3, 3)).astype(np.float32) * np.float32(0.1)
    codes, scales, rows, inner = pack_int4(filters.reshape(3, -1))
    tensor = _tensor(codes, scales, rows, inner, codes.shape[1] * 2)
    tensor.shape = (3, 4, 3, 3)
    uploaded = ops.upload_volume(volume, fp32=True)
    out = ops.conv(uploaded, tensor, stride=1, pad=1)
    got = ops.download_volume(out).astype(np.float32)
    ops.free(uploaded, out, tensor.codes, tensor.scales)
    padded = np.pad(volume, ((0, 0), (1, 1), (1, 1)))
    expected = np.zeros((3, 68, 68), np.float32)
    restored = restore_int4(codes, scales)[:rows, :inner].reshape(3, 4, 3, 3)
    for oc in range(3):
        acc = np.zeros((68, 68), np.float32)
        for ic in range(4):
            for ky in range(3):
                for kx in range(3):
                    acc += padded[ic, ky : ky + 68, kx : kx + 68] * np.float32(restored[oc, ic, ky, kx])
        expected[oc] = acc
    _close(got, expected, "conv")


def _tensor(codes: np.ndarray, scales: np.ndarray, rows: int, inner: int, stored_k: int) -> Tensor:
    ops_codes = engine.alloc(codes.nbytes)
    ops_scales = engine.alloc(scales.nbytes)
    engine.call(
        "klein_copy_h2d",
        engine.pointer(ops_codes),
        codes.ctypes.data_as(__import__("ctypes").c_void_p),
        __import__("ctypes").c_uint64(codes.nbytes),
    )
    engine.call(
        "klein_copy_h2d",
        engine.pointer(ops_scales),
        np.ascontiguousarray(scales, dtype=np.float32).ctypes.data_as(__import__("ctypes").c_void_p),
        __import__("ctypes").c_uint64(scales.nbytes),
    )
    return Tensor(
        kind="int4",
        ptr=ops_codes,
        rows=rows,
        inner=inner,
        stored_k=stored_k,
        codes=ops_codes,
        scales=ops_scales,
    )


def _close(got: np.ndarray, expected: np.ndarray, label: str) -> None:
    error = float(np.max(np.abs(got - expected)))
    if not np.allclose(got, expected, rtol=2e-2, atol=2e-2):
        message = f"{label} max abs error {error}"
        raise RuntimeError(message)
