"""Shifted flow-matching knots.

The scheduler is FlowMatchEulerDiscreteScheduler. The shift is
``s·u / (1 + (s − 1)·u)``. Sigma 1 is noise and sigma 0 is the image.
The caller chooses the step count and the shift. The defaults are the
Turbo checkpoint: eight velocities and shift 3.
"""

from __future__ import annotations

import numpy as np

SHIFT = 3.0
STEPS = 8
T_SCALE = 1000.0


def sigmas(steps: int = STEPS, shift: float = SHIFT) -> np.ndarray:
    if steps < 1:
        message = f"steps must be positive, got {steps}"
        raise ValueError(message)
    if shift <= 0:
        message = f"shift must be positive, got {shift}"
        raise ValueError(message)
    u = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    shifted = shift * u / (1.0 + (shift - 1.0) * u)
    return shifted.astype(np.float32)


def denoise_start(steps: int, strength: float) -> int:
    """First knot of an image-to-image run.

    ``strength`` 1 starts on the noise knot. ``strength`` 0 starts on the image
    knot, so the loop does not run. The cut is the published one:
    ``int(steps − min(steps · strength, steps))``.
    """
    if steps < 1:
        message = f"steps must be positive, got {steps}"
        raise ValueError(message)
    if strength < 0.0 or strength > 1.0:
        message = f"strength must be from 0 to 1, got {strength}"
        raise ValueError(message)
    init = min(steps * float(strength), float(steps))
    return int(max(float(steps) - init, 0.0))


def timestep(sigma: float) -> float:
    """Embedded time. The network sees ``(1 − sigma) · t_scale``."""
    return (1.0 - float(sigma)) * T_SCALE
