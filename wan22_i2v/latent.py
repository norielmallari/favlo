"""Stride latent used when ``Wan2.1_VAE.pth`` is not on disk.

The first pixel frame is the image. Later frames are zeros. Each 4×8×8 cell
averages into the latent grid, then the published channel affine is applied.
"""

from __future__ import annotations

import numpy as np

from wan22_i2v.schedule import LATENT_MEAN, LATENT_STD, geometry, mask_latent


def stride_encode(image: np.ndarray, frames: int) -> np.ndarray:
    """uint8 HWC image to the 16-channel latent ``[16, T, H, W]``."""
    height, width = image.shape[:2]
    lat_t, lat_h, lat_w, _ = geometry(frames, height, width)
    pixels = image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    video = np.zeros((3, frames, height, width), dtype=np.float32)
    video[:, 0] = np.transpose(pixels, (2, 0, 1))
    latent = np.zeros((16, lat_t, lat_h, lat_w), dtype=np.float32)
    for time in range(lat_t):
        if time == 0:
            chunk = video[:, :1]
        else:
            start = 1 + 4 * (time - 1)
            chunk = video[:, start : start + 4]
        pooled = chunk.reshape(3, chunk.shape[1], lat_h, 8, lat_w, 8).mean(axis=(1, 3, 5))
        latent[:3, time] = pooled
    mean = LATENT_MEAN[:, None, None, None]
    std = LATENT_STD[:, None, None, None]
    return (latent - mean) / std


def stride_decode(latent: np.ndarray, frames: int) -> np.ndarray:
    """16-channel latent to uint8 ``[frames, height, width, 3]``."""
    restored = latent * LATENT_STD[:, None, None, None] + LATENT_MEAN[:, None, None, None]
    restored = np.clip(restored, -1.0, 1.0)
    rgb = restored[:3]
    frames_out: list[np.ndarray] = []
    for time in range(rgb.shape[1]):
        tile = np.repeat(np.repeat(rgb[:, time], 8, axis=1), 8, axis=2)
        copies = 1 if time == 0 else 4
        frames_out.extend(tile for _ in range(copies))
    video = np.stack(frames_out[:frames], axis=0)
    uint8 = np.rint((video + 1.0) * 127.5).astype(np.uint8)
    return np.transpose(uint8, (0, 2, 3, 1))


def condition_volume(noise: np.ndarray, latent: np.ndarray, frames: int) -> np.ndarray:
    """36 channels: noise, the packed mask, then the VAE latent."""
    mask = mask_latent(frames)
    spatial = np.broadcast_to(mask[:, :, None, None], (4, *noise.shape[1:]))
    return np.concatenate((noise, np.ascontiguousarray(spatial), latent), axis=0).astype(np.float32)


def unpatchify(tokens: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Head rows ``[tokens, 64]`` back to ``[16, T, H, W]``.

    A row is one time patch, a 2×2 spatial patch, then 16 channels.
    """
    time, height, width = grid
    raw = tokens.reshape(time, height, width, 1, 2, 2, 16)
    ordered = np.transpose(raw, (6, 0, 3, 1, 4, 2, 5))
    return np.ascontiguousarray(ordered.reshape(16, time, height * 2, width * 2))
