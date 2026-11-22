"""Int4 pack. One fp16 scale per output row. Low nibble is element 0."""

from __future__ import annotations

import numpy as np

from favlo.quantize import quantize_int4


def pack_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad K to a multiple of 256, then store codes and an fp16 row scale."""
    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        message = "int4 weight must be a matrix"
        raise ValueError(message)
    rows, cols = matrix.shape
    pad = (256 - (cols % 256)) % 256
    if pad:
        stored = np.zeros((rows, cols + pad), dtype=np.float32)
        stored[:, :cols] = matrix
        matrix = stored
    codes, scale = quantize_int4(matrix)
    return codes, np.ascontiguousarray(scale, dtype=np.float16)
