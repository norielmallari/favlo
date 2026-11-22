"""NVFP4 reconstruction used by the dequant kernel.

NVIDIA's rule, from the Transformer Engine NVFP4 note: an E2M1 code times an
E4M3 block scale times a fp32 outer scale. The outer scale is the tensor amax
divided by 448 * 6, and that divisor is the checkpoint's
``outer_scale_denominator`` (2688).
"""

from __future__ import annotations

import numpy as np

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e4m3_to_float(byte: int) -> float:
    """Decode one NVIDIA E4M3FN byte. The max finite value is 448."""
    value = int(byte) & 0xFF
    sign = (value >> 7) & 1
    exp = (value >> 3) & 0xF
    man = value & 0x7
    if exp == 0:
        mag = man * 2.0**-9
    elif exp == 15:
        mag = float("nan") if man == 7 else (1.0 + man / 8.0) * 256.0
    else:
        mag = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return -mag if sign else mag


def nvfp4_row(packed: np.ndarray, scales: np.ndarray, global_scale: float) -> np.ndarray:
    """Dequant one packed row. ``packed`` is K/2 bytes, ``scales`` is K/16 bytes."""
    width = int(packed.shape[0]) * 2
    out = np.empty(width, dtype=np.float32)
    for index in range(width):
        byte = int(packed[index // 2])
        nibble = (byte >> 4) if index & 1 else byte & 0xF
        sign = -1.0 if nibble >= 8 else 1.0
        code = E2M1[nibble & 7] * sign
        out[index] = code * e4m3_to_float(int(scales[index // 16])) * float(global_scale)
    return out
