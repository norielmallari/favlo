"""int8 and int4 images. The quantizer is ``favlo.quantize``."""

from __future__ import annotations

import math

import numpy as np

from favlo.quantize import quantize_int4, quantize_int8
from gemma4_26b_a4b.cuda.layout import EXPERT_K, GATE_ROWS, HIDDEN, gemm_bytes


def restore_int4(codes: np.ndarray, scale: np.ndarray, rows: int, inner: int) -> np.ndarray:
    """Undo ``quantize_int4``. The low nibble is the even column."""
    raw = np.ascontiguousarray(codes).reshape(rows, inner // 2)
    low = (raw & np.uint8(0x0F)).astype(np.int16)
    high = (raw >> np.uint8(4)).astype(np.int16)

    def _extend(nibble: np.ndarray) -> np.ndarray:
        return np.where(nibble >= 8, nibble - 16, nibble).astype(np.float32)

    values = np.empty((rows, inner), dtype=np.float32)
    values[:, 0::2] = _extend(low)
    values[:, 1::2] = _extend(high)
    return values * np.ascontiguousarray(scale, dtype=np.float32)[:, None]


def relative_error(weight: np.ndarray) -> float:
    """RMS of the int4 restoration divided by the source RMS."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    rows, inner = matrix.shape
    if inner % 2:
        message = f"int4 inner {inner} is odd"
        raise ValueError(message)
    codes, scale = quantize_int4(matrix)
    restored = restore_int4(codes, scale, rows, inner)
    diff = matrix.astype(np.float64) - restored.astype(np.float64)
    numerator = float(np.dot(diff.ravel(), diff.ravel()))
    flat = matrix.astype(np.float64).ravel()
    denominator = float(np.dot(flat, flat))
    if denominator == 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


def pack_matrix(weight: np.ndarray, plane: str) -> bytes:
    """Codes, then one fp32 scale per row. An inner of 704 is stored as 768."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        message = "a packed matrix is rank 2"
        raise ValueError(message)
    rows, inner = matrix.shape
    stored = inner
    if inner == 704:
        stored = EXPERT_K
        padded = np.zeros((rows, stored), dtype=np.float32)
        padded[:, :inner] = matrix
        matrix = padded
    if plane == "int8":
        codes, scale = quantize_int8(matrix)
    elif plane == "int4":
        codes, scale = quantize_int4(matrix)
    else:
        message = f"weight plane {plane} is not int8 or int4"
        raise ValueError(message)
    blob = np.ascontiguousarray(codes).tobytes() + np.ascontiguousarray(scale, dtype=np.float32).tobytes()
    expect = gemm_bytes(plane, rows, stored)
    if len(blob) != expect:
        message = f"{plane} matrix packed {len(blob)} bytes, expected {expect}"
        raise RuntimeError(message)
    return blob


def pack_expert(gate_up: np.ndarray, down: np.ndarray, plane: str) -> bytes:
    """One expert. Gate-up is ``[1408, 2816]`` and down is ``[2816, 704]``."""
    if gate_up.shape != (GATE_ROWS, HIDDEN):
        message = f"gate-up shape {gate_up.shape} is not {(GATE_ROWS, HIDDEN)}"
        raise ValueError(message)
    if down.shape != (HIDDEN, 704):
        message = f"down shape {down.shape} is not {(HIDDEN, 704)}"
        raise ValueError(message)
    return pack_matrix(gate_up, plane) + pack_matrix(down, plane)


def store_f16(values: np.ndarray) -> bytes:
    """IEEE fp16 bytes of a vector."""
    return np.ascontiguousarray(values, dtype=np.float32).astype(np.float16).tobytes()
