"""Classifier-free guidance for the base checkpoint.

The mix is the published Z-Image update: ``pos + scale * (pos - neg)``.
``cfg_truncation`` at or below 1 skips the negative velocity once
``1 - sigma`` exceeds it. A truncation above 1 stays on for every knot.
``cfg_normalization`` above 0 caps the mixed vector at that multiple of the
positive velocity's L2 norm. Zero leaves the mix unchanged.
"""

from __future__ import annotations


def applies(sigma: float, guidance: float, truncation: float) -> bool:
    """Whether this knot runs the negative velocity."""
    if guidance <= 0:
        return False
    if truncation > 1.0:
        return True
    return not ((1.0 - float(sigma)) > float(truncation))


def normalization_limit(value: float) -> float:
    """The L2 cap multiple. Zero disables the rescale."""
    if value < 0:
        message = f"cfg_normalization must be non-negative, got {value}"
        raise ValueError(message)
    if value == 0:
        return 0.0
    return float(value)
