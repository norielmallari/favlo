"""Finite scalar quantization bottleneck.

The checkpoint scale is ``scalar_quantization_scale``. A projected row is
passed through tanh, then rounded to that scale. Inference does not use the
straight-through estimator.
"""

from __future__ import annotations

import numpy as np


def quantize_bound(projected: np.ndarray, scale: int) -> np.ndarray:
    """``round(tanh(y) * scale) / scale`` on the last axis."""
    bounded = np.tanh(np.asarray(projected, dtype=np.float64))
    width = int(scale)
    return np.round(bounded * width) / width
