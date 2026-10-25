"""Four-knot solver, token grid, and the image mask.

Shift 8 indexes the length-1001 grid at 0, 250, 500, and 750.
The last knot decodes ``x0``. It does not step to 0.
"""

from __future__ import annotations

import numpy as np

from wan22_ti2v_5b.spec import FPS, ROPE_H, ROPE_LEN, ROPE_T, ROPE_W

_INDICES = (0, 250, 500, 750)
_SHIFT = 8.0


def sigma_at(index: int) -> float:
    """Shift-8 sigma. ``u = 1 - index/1000`` and ``sigma = 8u / (1 + 7u)``."""
    u = 1.0 - (index / 1000.0)
    return (_SHIFT * u) / (1.0 + (_SHIFT - 1.0) * u)


def timesteps() -> tuple[float, float, float, float]:
    """The four printed knots. A run refuses any other row."""
    return tuple(1000.0 * sigma_at(index) for index in _INDICES)  # type: ignore[return-value]


def sigmas() -> tuple[float, float, float, float]:
    return tuple(sigma_at(index) for index in _INDICES)  # type: ignore[return-value]


def require_knots(values: tuple[float, ...]) -> None:
    expected = timesteps()
    if len(values) != len(expected):
        message = f"timesteps {values} are not the shift-8 row"
        raise RuntimeError(message)
    for got, want in zip(values, expected, strict=True):
        if abs(got - want) > 1e-3:
            message = f"timesteps {values} are not the shift-8 row"
            raise RuntimeError(message)


def snap_frames(seconds: float) -> int:
    """Nearest ``4n+1`` count at 24 fps, and at least 5 frames."""
    raw = max(int(round(seconds * FPS)), 5)
    remainder = (raw - 1) % 4
    snapped = raw - remainder
    if remainder >= 2:
        snapped += 4
    return max(snapped, 5)


def latent_grid(frames: int, height: int, width: int) -> tuple[int, int, int]:
    if frames < 1 or (frames - 1) % 4 != 0:
        message = f"frames must be 4n+1, got {frames}"
        raise ValueError(message)
    if height % 32 != 0 or width % 32 != 0:
        message = f"edges must be multiples of 32, got {width}x{height}"
        raise ValueError(message)
    time = (frames - 1) // 4 + 1
    return time, height // 16, width // 16


def token_grid(frames: int, height: int, width: int) -> tuple[int, int, int]:
    time, lat_h, lat_w = latent_grid(frames, height, width)
    return time, lat_h // 2, lat_w // 2


def token_count(frames: int, height: int, width: int) -> int:
    time, height_tokens, width_tokens = token_grid(frames, height, width)
    return time * height_tokens * width_tokens


def pad_multiple(count: int, multiple: int) -> int:
    if count < 1:
        return multiple
    return ((count + multiple - 1) // multiple) * multiple


def x0_from_velocity(latent: np.ndarray, sigma: float, velocity: np.ndarray) -> np.ndarray:
    return latent - np.float32(sigma) * velocity


def renoise(clean: np.ndarray, sigma_next: float, noise: np.ndarray) -> np.ndarray:
    keep = np.float32(1.0 - sigma_next)
    return keep * clean + np.float32(sigma_next) * noise


def image_mask(time: int, height: int, width: int) -> np.ndarray:
    """0 on latent frame 0 and 1 after that. Shape ``[1, T, H, W]``."""
    mask = np.ones((1, time, height, width), dtype=np.float32)
    mask[:, 0] = 0.0
    return mask


def text_mask(time: int, height: int, width: int) -> np.ndarray:
    return np.ones((1, time, height, width), dtype=np.float32)


def blend(latent: np.ndarray, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(1 - mask) * image + mask * latent``. ``image`` is frame 0, broadcast in time."""
    return (1.0 - mask) * image + mask * latent


def token_timesteps(mask: np.ndarray, timestep: float) -> np.ndarray:
    """Channel 0, subsampled ``::2`` on height and width, times the knot."""
    spatial = mask[0, :, ::2, ::2]
    return (spatial * np.float32(timestep)).astype(np.float32)


def rope_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine and sine pairs for time, height, and width. Each row is ``[cos | sin]``."""

    def axis(width: int) -> np.ndarray:
        freqs = 1.0 / (10000.0 ** (np.arange(0, width, 2, dtype=np.float64) / width))
        angles = np.arange(ROPE_LEN, dtype=np.float64)[:, None] * freqs[None, :]
        table = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        return table.astype(np.float32)

    if ROPE_T + ROPE_H + ROPE_W != 128:
        message = "rope axis split does not cover the head"
        raise RuntimeError(message)
    return axis(ROPE_T), axis(ROPE_H), axis(ROPE_W)


def relative_bucket(query: int, key: int, *, buckets: int = 32, max_distance: int = 128) -> int:
    """Bidirectional T5 bucket. ``key - query`` is the relative position."""
    relative = key - query
    half_buckets = buckets // 2
    bucket = half_buckets if relative > 0 else 0
    distance = abs(relative)
    exact = half_buckets // 2
    if distance < exact:
        return bucket + distance
    span = max_distance / exact
    large = exact + int(np.log(distance / exact) / np.log(span) * (half_buckets - exact))
    return bucket + min(large, half_buckets - 1)
