"""Shifted flow-matching knots for Z-Image-Turbo.

The checkpoint scheduler is FlowMatchEulerDiscreteScheduler with shift 3 and
1000 train timesteps. Eight velocity evaluations use nine knots. The shift is
``s·u / (1 + (s − 1)·u)``. Sigma 1 is noise and sigma 0 is the image.
"""

from __future__ import annotations

import numpy as np

SHIFT = 3.0
STEPS = 8
T_SCALE = 1000.0


def sigmas(steps: int = STEPS, shift: float = SHIFT) -> np.ndarray:
    if steps != STEPS:
        message = f"this pipeline evaluates {STEPS} velocities, got {steps}"
        raise ValueError(message)
    u = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    shifted = shift * u / (1.0 + (shift - 1.0) * u)
    return shifted.astype(np.float32)


def timestep(sigma: float) -> float:
    """Embedded time. The network sees ``(1 − sigma) · t_scale``."""
    return (1.0 - float(sigma)) * T_SCALE
