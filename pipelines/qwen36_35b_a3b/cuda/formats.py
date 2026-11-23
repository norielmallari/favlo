"""int8 and int4 expert images. The quantizer is ``favlo.quantize``."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from favlo.quantize import quantize_int4, quantize_int8


def _layout_module() -> Any:
    name = "qwen36_cuda_layout"
    found = sys.modules.get(name)
    if found is not None:
        return found
    path = Path(__file__).resolve().parent / "layout.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_layout = _layout_module()


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
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
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
    """Codes, then one fp32 scale per row."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        message = "a packed matrix is rank 2"
        raise ValueError(message)
    if plane == "int8":
        codes, scale = quantize_int8(matrix)
    elif plane == "int4":
        codes, scale = quantize_int4(matrix)
    else:
        message = f"weight plane {plane} is not int8 or int4"
        raise ValueError(message)
    blob = np.ascontiguousarray(codes).tobytes() + np.ascontiguousarray(scale,
        dtype=np.float32).tobytes()
    expect = _layout.gemm_bytes(plane, matrix.shape[0], matrix.shape[1])
    if len(blob) != expect:
        message = f"{plane} matrix packed {len(blob)} bytes, expected {expect}"
        raise RuntimeError(message)
    return blob


def pack_expert(gate_up: np.ndarray, down: np.ndarray, plane: str) -> bytes:
    """One expert image. Gate-up is ``[1024, 2048]`` and down is ``[2048, 512]``."""
    blob = pack_matrix(gate_up, plane) + pack_matrix(down, plane)
    expect = _layout.expert_image_bytes(plane)
    if len(blob) != expect:
        message = f"{plane} expert packed {len(blob)} bytes, expected {expect}"
        raise RuntimeError(message)
    return blob
