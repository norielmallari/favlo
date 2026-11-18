"""One packed matrix and the attention scores around ``matmul2d``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wan22_ti2v_5b.metal41.activations import softmax
from wan22_ti2v_5b.metal41.blob import expect_f16, expect_i8
from wan22_ti2v_5b.metal41.gpu import Gpu


@dataclass
class Matrix:
    kind: str
    packed: np.ndarray
    scales: np.ndarray | None
    bias: np.ndarray | None
    perm: np.ndarray | None
    logical_n: int
    logical_k: int
    span: object | None = None
    packed_at: int | None = None
    scales_at: int | None = None
    bias_at: int | None = None
    stored_shape: tuple[int, ...] | None = None

    def stored_n(self) -> int:
        if self.stored_shape is not None:
            return int(self.stored_shape[0])
        return int(self.packed.shape[0])

    def stored_k(self) -> int:
        if self.stored_shape is not None:
            return int(self.stored_shape[1])
        return int(self.packed.shape[1])


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


def matrix_from_planes(
    handle: object,
    planes: dict[str, dict],
    name: str,
    *,
    kind: str,
    bias_path: Path | None = None,
) -> Matrix:
    """Bind one matrix at its plane offset. The packed codes stay in the span."""
    plane = planes.get(name)
    if plane is None:
        plane = planes.get(f"{name}.packed")
    if plane is None:
        message = f"weight pack is missing {name}"
        raise FileNotFoundError(message)
    shape = tuple(int(value) for value in plane["shape"])
    if kind == "f16":
        if "logical_n" not in plane or "logical_k" not in plane:
            message = f"{name} is missing its float16 extents"
            raise RuntimeError(message)
        expect_f16(plane)
    elif kind == "int8":
        expect_i8(plane)
    else:
        message = f"unknown kind {kind}"
        raise RuntimeError(message)
    scales_at = None
    bias_at = None
    bias = None
    if kind == "int8":
        scale = planes.get(f"{name}.scales")
        if scale is None:
            message = f"weight pack is missing {name}.scales"
            raise FileNotFoundError(message)
        if tuple(int(value) for value in scale["shape"]) != (shape[0],):
            message = f"{name}.scales length does not match the int8 rows"
            raise RuntimeError(message)
        if np.dtype(scale["dtype"]) != np.float32:
            message = f"{name}.scales is not float32"
            raise RuntimeError(message)
        scales_at = int(scale["offset"])
        bias_plane = planes.get(f"{name}.bias")
        if bias_plane is not None and tuple(int(value) for value in bias_plane["shape"]) == (shape[0],):
            if np.dtype(bias_plane["dtype"]) != np.float16:
                message = f"{name}.bias is not float16"
                raise RuntimeError(message)
            bias_at = int(bias_plane["offset"])
        elif bias_plane is not None:
            from wan22_ti2v_5b.metal41.blob import copy_plane

            bias = copy_plane(handle, bias_plane)
    if bias_path is not None and bias_path.is_file():
        bias = np.load(bias_path)
    logical_n = int(plane["logical_n"]) if "logical_n" in plane else int(shape[0])
    logical_k = int(plane["logical_k"]) if "logical_k" in plane else int(shape[1])
    if kind == "int8" and "logical_n" not in plane and bias is not None:
        bias_n = int(np.asarray(bias).reshape(-1).shape[0])
        if bias_n < logical_n:
            logical_n = bias_n
    return Matrix(
        kind,
        None,
        None,
        bias,
        None,
        logical_n,
        logical_k,
        handle,
        int(plane["offset"]),
        scales_at,
        bias_at,
        shape,
    )


def apply(gpu: Gpu, rows: np.ndarray, matrix: Matrix) -> np.ndarray:
    """``rows @ weight.T`` plus bias. ``rows`` is ``[M, K]``."""
    data = np.ascontiguousarray(rows, dtype=np.float32)
    stored_k = matrix.stored_k()
    if matrix.kind == "int8" and data.shape[1] != stored_k:
        if data.shape[1] > stored_k:
            message = f"K {data.shape[1]} is wider than the int8 weight"
            raise RuntimeError(message)
        padded = np.zeros((data.shape[0], stored_k), dtype=np.float32)
        padded[:, : data.shape[1]] = data
        data = padded
    if matrix.perm is not None:
        data = np.ascontiguousarray(data[:, matrix.perm])
    if matrix.span is not None:
        values = gpu.linear_span(data, matrix)
    else:
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
    """Every head, then one wait. ``query`` is ``[heads, tokens, dim]``.

    The diffusion transformer scales by ``1/sqrt(dim)``. umT5 does not: the
    relative bucket bias is added to the raw dot product.
    """
    dim = int(query.shape[2])
    scale = np.float32(dim**-0.5 if scaled else 1.0)
    scores = gpu.gemm_half_heads(
        query.astype(np.float16),
        key.astype(np.float16),
        transpose_right=True,
    )
    scores = scores * scale
    if bias is not None:
        scores = scores + bias.astype(np.float32)
    if key_mask is not None:
        scores[:, :, ~key_mask] = np.float32(-1.0e30)
    weights = softmax(scores).astype(np.float16)
    return gpu.gemm_half_heads(weights, value.astype(np.float16), transpose_right=False)
