"""Host formulas for the kernel checks."""

from __future__ import annotations

import math

import numpy as np

from zonos2.layout import NORM_EPS, ROPE_THETA, SOFTCAP


def rms_norm(values: np.ndarray, weight: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    flat = values.astype(np.float64)
    scale = math.sqrt(float(np.dot(flat, flat)) / flat.size + eps)
    return (values.astype(np.float64) / scale * weight.astype(np.float64)).astype(np.float32)


def rope_pairs(values: np.ndarray, position: int, theta: float = ROPE_THETA) -> np.ndarray:
    """Su et al. rotation on adjacent pairs."""
    head = values.astype(np.float64).copy()
    half = head.size // 2
    for index in range(half):
        freq = position / (theta ** (2 * index / head.size))
        cos = math.cos(freq)
        sin = math.sin(freq)
        left = head[2 * index]
        right = head[2 * index + 1]
        head[2 * index] = left * cos - right * sin
        head[2 * index + 1] = left * sin + right * cos
    return head.astype(np.float32)


def silu(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float64)
    return (data / (1.0 + np.exp(-data))).astype(np.float32)


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return (silu(gate).astype(np.float64) * up.astype(np.float64)).astype(np.float32)


def softcap(values: np.ndarray, tau: float = SOFTCAP) -> np.ndarray:
    data = values.astype(np.float64)
    return (tau * np.tanh(data / tau)).astype(np.float32)
