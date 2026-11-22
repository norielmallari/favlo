"""Four-step flow schedule. μ follows the published empirical lines."""

from __future__ import annotations

import numpy as np


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Shift for one target token count and step count."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    slope = (m_200 - m_10) / 190.0
    offset = m_200 - 200.0 * slope
    return float(slope * num_steps + offset)


def sigmas(image_seq_len: int, num_steps: int) -> np.ndarray:
    """Shifted sigmas plus a trailing 0. The model sees ``sigma * 1000``."""
    if num_steps < 1:
        message = "steps must be positive"
        raise ValueError(message)
    mu = empirical_mu(image_seq_len, num_steps)
    base = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float64)
    shifted = np.exp(mu) / (np.exp(mu) + (1.0 / base - 1.0))
    full = np.concatenate([shifted, np.zeros(1, dtype=np.float64)])
    return full.astype(np.float32)
