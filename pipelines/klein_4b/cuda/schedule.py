"""Empirical rectified-flow shift used by FLUX.2 [klein]."""

from __future__ import annotations

import math

import numpy as np

_A1 = 8.73809524e-05
_B1 = 1.89833333
_A2 = 0.00016927
_B2 = 0.45666666


def empirical_mu(tokens: int, steps: int) -> float:
    """Blend the 10-step and 200-step lines. Above 4300 tokens, use the 200-step line."""
    if tokens > 4300:
        return _A2 * tokens + _B2
    low = _A1 * tokens + _B1
    high = _A2 * tokens + _B2
    slope = (high - low) / 190.0
    return slope * steps + (low - 10.0 * slope)


def shift_sigma(mu: float, sigma: float) -> float:
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0))


def sigmas(tokens: int, steps: int) -> np.ndarray:
    """``N`` shifted sigmas and a trailing 0."""
    mu = empirical_mu(tokens, steps)
    base = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float64)
    shifted = np.exp(mu) / (np.exp(mu) + (1.0 / base - 1.0))
    return np.concatenate([shifted, np.zeros(1, dtype=np.float64)]).astype(np.float32)
