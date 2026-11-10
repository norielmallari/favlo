"""Symmetric int8 pack for umT5 linears."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT8_TILE_N = 128
K_ALIGN = 256


@dataclass(frozen=True)
class LinearPack:
    kind: str
    packed: np.ndarray
    scales: np.ndarray
    bias: np.ndarray
    perm: np.ndarray
    logical_n: int
    logical_k: int
    padded_n: int
    stored_k: int


def stored_k(logical: int) -> int:
    if logical <= 0:
        message = f"K must be positive, got {logical}"
        raise ValueError(message)
    return ((logical + K_ALIGN - 1) // K_ALIGN) * K_ALIGN


def pad_to(n: int, tile: int) -> int:
    if n < 0:
        message = f"extent must be non-negative, got {n}"
        raise ValueError(message)
    return ((n + tile - 1) // tile) * tile


def pack_int8(weight: np.ndarray, bias: np.ndarray | None = None) -> LinearPack:
    """Permute-free int8. One float32 scale per output row."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    logical_n, logical_k = matrix.shape
    sk = stored_k(logical_k)
    padded_n = pad_to(logical_n, INT8_TILE_N)
    padded = np.zeros((padded_n, sk), dtype=np.float32)
    padded[:logical_n, :logical_k] = matrix
    peak = np.max(np.abs(padded), axis=1)
    scale = np.zeros(peak.shape, dtype=np.float32)
    live = peak > 0
    scale[live] = peak[live] / np.float32(127.0)
    inverse = np.zeros(peak.shape, dtype=np.float32)
    inverse[live] = np.float32(127.0) / peak[live]
    codes = np.rint(padded * inverse[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    if bias is None:
        stored_bias = np.zeros(logical_n, dtype=np.float16)
    else:
        stored_bias = np.ascontiguousarray(bias, dtype=np.float16).reshape(logical_n)
    return LinearPack(
        kind="int8",
        packed=np.ascontiguousarray(codes),
        scales=np.ascontiguousarray(scale),
        bias=stored_bias,
        perm=np.arange(sk, dtype=np.uint32),
        logical_n=logical_n,
        logical_k=logical_k,
        padded_n=padded_n,
        stored_k=sk,
    )


def restore_int8(pack: LinearPack) -> np.ndarray:
    codes = np.asarray(pack.packed, dtype=np.float32)
    scales = np.asarray(pack.scales, dtype=np.float32)
    restored = codes * scales[:, None]
    return np.ascontiguousarray(restored[: pack.logical_n, : pack.logical_k])


def relative_error(restored: np.ndarray, source: np.ndarray) -> float:
    residual = restored.astype(np.float32) - source.astype(np.float32)
    denom = float(np.sqrt(np.mean(np.square(source.astype(np.float32)))))
    if denom == 0.0:
        return 0.0
    numer = float(np.sqrt(np.mean(np.square(residual))))
    return numer / denom
