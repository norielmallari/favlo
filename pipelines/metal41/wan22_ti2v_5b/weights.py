"""One packed matrix and the attention scores around ``matmul2d``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wan22_ti2v_5b.activations import softmax
from wan22_ti2v_5b.gpu import Gpu


@dataclass
class Matrix:
    kind: str
    packed: np.ndarray
    scales: np.ndarray | None
    bias: np.ndarray | None
    perm: np.ndarray | None
    logical_n: int
    logical_k: int


def load_matrix(path: Path) -> Matrix:
    if path.suffix == ".npy":
        weight = np.load(path)
        bias_path = path.with_name(path.name.replace(".npy", ".bias.npy"))
        bias = np.load(bias_path) if bias_path.is_file() else None
        return Matrix("f16", weight, None, bias, None, int(weight.shape[0]), int(weight.shape[1]))
    with np.load(path) as packed:
        kind = str(packed["kind"])
        data = np.array(packed["packed"])
        scales = np.array(packed["scales"])
        bias = np.array(packed["bias"]) if "bias" in packed.files else None
        perm = np.array(packed["perm"]) if "perm" in packed.files else None
        logical_n = int(packed["logical_n"]) if "logical_n" in packed.files else int(data.shape[0])
        logical_k = int(packed["logical_k"]) if "logical_k" in packed.files else int(data.shape[1])
    if perm is not None and np.array_equal(perm, np.arange(perm.shape[0], dtype=perm.dtype)):
        perm = None
    return Matrix(kind, data, scales, bias, perm, logical_n, logical_k)


def apply(gpu: Gpu, rows: np.ndarray, matrix: Matrix) -> np.ndarray:
    """``rows @ weight.T`` plus bias. ``rows`` is ``[M, K]``."""
    data = np.ascontiguousarray(rows, dtype=np.float32)
    stored_k = int(matrix.packed.shape[1])
    if matrix.kind == "int8" and data.shape[1] != stored_k:
        if data.shape[1] > stored_k:
            message = f"K {data.shape[1]} is wider than the int8 weight"
            raise RuntimeError(message)
        padded = np.zeros((data.shape[0], stored_k), dtype=np.float32)
        padded[:, : data.shape[1]] = data
        data = padded
    if matrix.perm is not None:
        data = np.ascontiguousarray(data[:, matrix.perm])
    values = gpu.linear(data, matrix.packed, matrix.scales, kind=matrix.kind, bias=matrix.bias)
    if matrix.kind == "int8" and values.shape[1] != matrix.logical_n:
        return values[:, : matrix.logical_n]
    return values


def attend(
    gpu: Gpu,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    key_mask: np.ndarray | None = None,
    scaled: bool = True,
) -> np.ndarray:
    """One head at a time. ``query`` is ``[heads, tokens, dim]``.

    The diffusion transformer scales by ``1/sqrt(dim)``. umT5 does not: the
    relative bucket bias is added to the raw dot product.
    """
    heads, q_len, dim = query.shape
    scale = np.float32(dim**-0.5 if scaled else 1.0)
    out = np.empty((heads, q_len, value.shape[2]), dtype=np.float32)
    for head in range(heads):
        scores = gpu.gemm_half(
            query[head].astype(np.float16),
            key[head].astype(np.float16),
            transpose_right=True,
        )
        scores = scores * scale
        if bias is not None:
            scores = scores + bias[head].astype(np.float32)
        if key_mask is not None:
            scores[:, ~key_mask] = np.float32(-1.0e30)
        weights = softmax(scores).astype(np.float16)
        mixed = gpu.gemm_half(weights, value[head].astype(np.float16), transpose_right=False)
        out[head] = mixed
    return out
