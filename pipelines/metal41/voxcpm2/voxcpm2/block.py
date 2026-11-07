"""MiniCPM-4 block equations used by the text-semantic language model.

``use_mup`` is false on this checkpoint, so the residual add is unscaled and
the token embedding scale is 1. LongRoPE follows the checkpoint factors.
"""

from __future__ import annotations

import math

import numpy as np

from voxcpm2.config import RopeScaling


def embedding_scale(use_mup: bool, scale_emb: float) -> float:
    """Token rows are multiplied by this before they enter the stack."""
    if use_mup:
        return float(scale_emb)
    return 1.0


def residual_add(
    residual: np.ndarray,
    update: np.ndarray,
    *,
    use_mup: bool,
    scale_depth: float,
    layers: int,
) -> np.ndarray:
    """Add a sublayer. The depth scale applies only when ``use_mup`` is set."""
    if use_mup:
        update = update * (float(scale_depth) / math.sqrt(int(layers)))
    return residual + update


def rmsnorm(hidden: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm over the last axis. The variance is float32."""
    values = np.asarray(hidden, dtype=np.float32)
    scale = np.asarray(weight, dtype=np.float32)
    variance = np.mean(values * values, axis=-1, keepdims=True)
    normed = values * np.reciprocal(np.sqrt(variance + float(eps)))
    return normed * scale


def silu(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    return data * (1.0 / (1.0 + np.exp(-data)))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The feed-forward activation before the down projection."""
    return silu(gate) * np.asarray(up, dtype=np.float32)


def rope_scale(max_position: int, original_max: int) -> float:
    """LongRoPE attention scale. It is 1 when the two caps match."""
    ratio = float(max_position) / float(original_max)
    return math.sqrt(1.0 + math.log(ratio) / math.log(float(original_max)))


def rope_angles(
    position: int,
    *,
    dim: int,
    base: float,
    factors: tuple[float, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine for one position. ``factors`` has ``dim / 2`` entries."""
    half = dim // 2
    if len(factors) != half:
        message = f"rope factors cover {len(factors)} pairs, the head covers {half}"
        raise ValueError(message)
    index = np.arange(0, dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (float(base) ** (index / float(dim)))
    freqs = (float(position) / np.asarray(factors, dtype=np.float64)) * inv_freq
    angles = np.concatenate((freqs, freqs))
    return np.cos(angles) * scale, np.sin(angles) * scale


def rope_factors(position_count: int, rope: RopeScaling, max_position: int) -> tuple[float, ...]:
    """Short factors inside the original cap, long factors past it."""
    if position_count > rope.original_max_position_embeddings:
        return rope.long_factor
    del max_position
    return rope.short_factor
