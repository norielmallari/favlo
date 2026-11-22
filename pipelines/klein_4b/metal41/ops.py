"""Host norms, RoPE, and the two GPU ops. Residuals stay float16."""

from __future__ import annotations

import numpy as np

from klein_4b.portable.constants import NORM_EPS, VAE_EPS, VAE_GROUPS
from klein_4b.portable.rope import apply_dit, apply_text

from .metal import Gpu

_F16 = np.float32(65504)


def silu(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float32)
    return data / (np.float32(1) + np.exp(-data))


def rmsnorm(values: np.ndarray, weight: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    data = values.astype(np.float32)
    scale = np.sqrt(np.mean(data * data, axis=-1, keepdims=True) + np.float32(eps))
    return data / scale * weight.astype(np.float32)


def layernorm(values: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    data = values.astype(np.float32)
    mean = np.mean(data, axis=-1, keepdims=True)
    var = np.mean((data - mean) ** 2, axis=-1, keepdims=True)
    return (data - mean) / np.sqrt(var + np.float32(eps))


def as_f16(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(values, -_F16, _F16), dtype=np.float16)


def linear(gpu: Gpu, values: np.ndarray, key: str) -> np.ndarray:
    return gpu.linear(as_f16(values), key)


def modulate(values: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = layernorm(values)
    return as_f16((1.0 + scale.astype(np.float32)) * normalized + shift.astype(np.float32))


def gated(residual: np.ndarray, gate: np.ndarray, branch: np.ndarray) -> np.ndarray:
    mixed = residual.astype(np.float32) + gate.astype(np.float32) * branch.astype(np.float32)
    return as_f16(mixed)


def swiglu(values: np.ndarray) -> np.ndarray:
    half = values.shape[-1] // 2
    data = values.astype(np.float32)
    return as_f16(silu(data[..., :half]) * data[..., half:])


def text_rope(values: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return apply_text(values.astype(np.float32), cos, sin)


def dit_rope(values: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return apply_dit(values.astype(np.float32), cos, sin)


def attend(
    gpu: Gpu,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    causal: bool,
    valid: int,
) -> np.ndarray:
    return gpu.attend(query.astype(np.float32), key.astype(np.float32), value.astype(np.float32), causal=causal, valid=valid)


def groupnorm(volume: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Affine group norm over ``[C, H, W]``."""
    channels, height, width = volume.shape
    if channels % VAE_GROUPS:
        message = f"group norm wants a multiple of {VAE_GROUPS} channels, got {channels}"
        raise ValueError(message)
    data = volume.astype(np.float32).reshape(VAE_GROUPS, channels // VAE_GROUPS, height * width)
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    var = np.mean((data - mean) ** 2, axis=(1, 2), keepdims=True)
    norm = ((data - mean) / np.sqrt(var + np.float32(VAE_EPS))).reshape(channels, height, width)
    return norm * weight.astype(np.float32)[:, None, None] + bias.astype(np.float32)[:, None, None]


def im2col(volume: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """``[C, H, W]`` to ``[oh * ow, C * kh * kw]`` float16."""
    channels, height, width = volume.shape
    if pad:
        padded = np.pad(volume.astype(np.float16), ((0, 0), (pad, pad), (pad, pad)))
    else:
        padded = np.ascontiguousarray(volume, dtype=np.float16)
    out_h = (height + 2 * pad - kh) // stride + 1
    out_w = (width + 2 * pad - kw) // stride + 1
    s0, s1, s2 = padded.strides
    view = np.lib.stride_tricks.as_strided(
        padded,
        shape=(channels, kh, kw, out_h, out_w),
        strides=(s0, s1, s2, s1 * stride, s2 * stride),
        writeable=False,
    )
    cols = np.transpose(view, (3, 4, 0, 1, 2)).reshape(out_h * out_w, channels * kh * kw)
    return np.ascontiguousarray(cols, dtype=np.float16)


def conv(gpu: Gpu, volume: np.ndarray, key: str, *, stride: int, pad: int) -> np.ndarray:
    """One convolution. The weight is a packed ``[N, C*kh*kw]`` matrix."""
    bank = gpu.spec(key)
    if bank.kh is None or bank.kw is None:
        message = f"{key} is not a convolution"
        raise ValueError(message)
    channels, height, width = volume.shape
    cols = im2col(volume, bank.kh, bank.kw, stride, pad)
    out_h = (height + 2 * pad - bank.kh) // stride + 1
    out_w = (width + 2 * pad - bank.kw) // stride + 1
    flat = gpu.linear(cols, key)
    return as_f16(flat.reshape(out_h, out_w, bank.n).transpose(2, 0, 1))


def nearest2(volume: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(volume, 2, axis=1), 2, axis=2)


def reference_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    causal: bool,
    valid: int,
) -> np.ndarray:
    """Host softmax for the kernels check. ``query`` is ``[seq, heads, dim]``."""
    scale = 1.0 / np.sqrt(query.shape[-1])
    scores = np.einsum("qhd,khd->hqk", query, key) * scale
    seq = query.shape[0]
    keys = np.arange(seq)
    mask = keys[None, :] >= valid
    if causal:
        mask = mask | (keys[None, :] > np.arange(seq)[:, None])
    scores = np.where(mask[None, :, :], -1.0e30, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-20)
    return np.einsum("hqk,khd->qhd", weights, value).astype(np.float32)
