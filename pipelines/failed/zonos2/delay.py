"""DAC delay pattern. Clark et al. 2026, equations 1 and 2."""

from __future__ import annotations

import numpy as np

from zonos2.layout import CODEBOOKS, PAD_ID


def shear(aligned: np.ndarray, pad: int = PAD_ID) -> np.ndarray:
    """Map aligned codes ``[frames, 9]`` to delayed codes ``[frames + 8, 9]``."""
    frames, books = aligned.shape
    if books != CODEBOOKS:
        message = f"expected {CODEBOOKS} codebooks"
        raise ValueError(message)
    delayed = np.full((frames + CODEBOOKS - 1, books), pad, dtype=np.int64)
    for book in range(books):
        delayed[book : book + frames, book] = aligned[:, book]
    return delayed


def unshear(delayed: np.ndarray) -> np.ndarray:
    """Invert the shear. The result has ``len(delayed) - 8`` aligned frames."""
    steps, books = delayed.shape
    if books != CODEBOOKS or steps < CODEBOOKS - 1:
        message = "delayed codes are shorter than the shear"
        raise ValueError(message)
    frames = steps - (CODEBOOKS - 1)
    aligned = np.empty((frames, books), dtype=np.int64)
    for book in range(books):
        aligned[:, book] = delayed[book : book + frames, book]
    return aligned
