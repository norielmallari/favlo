"""Stride latent used when ``Wan2.1_VAE.pth`` is not on disk.

The first pixel frame is the image. Later frames are zeros. A last still, when
one is passed, is the final pixel. Each 4×8×8 cell averages into the latent
grid, then the published channel affine is applied.
"""

from __future__ import annotations

import numpy as np

from wan22_i2v.portable.schedule import LATENT_MEAN, LATENT_STD, geometry, mask_latent


def stride_encode(
    image: np.ndarray, frames: int, end: np.ndarray | None = None
) -> np.ndarray:
    """uint8 HWC image to the 16-channel latent ``[16, T, H, W]``."""
    height, width = image.shape[:2]
    lat_t, lat_h, lat_w, _ = geometry(frames, height, width)
    pixels = image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    video = np.zeros((3, frames, height, width), dtype=np.float32)
    video[:, 0] = np.transpose(pixels, (2, 0, 1))
    if end is not None:
        last = end.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
        video[:, -1] = np.transpose(last, (2, 0, 1))
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


def stride_encode_clip(pixels: np.ndarray) -> np.ndarray:
    """uint8 ``[frames, height, width, 3]`` to the 16-channel latent.

    ``frames`` is ``4n+1``. Latent 0 is the first pixel only. Each later latent
    is the next four pixels. This is the stride stand-in for a causal re-encode.
    """
    frames, height, width = (int(pixels.shape[0]), int(pixels.shape[1]), int(pixels.shape[2]))
    if frames < 1 or (frames - 1) % 4:
        message = f"a clip encode needs 4n+1 frames, got {frames}"
        raise ValueError(message)
    lat_t, lat_h, lat_w, _ = geometry(frames, height, width)
    values = pixels.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    video = np.transpose(values, (3, 0, 1, 2))
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


def blend_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average two copies of one frame.

    Overlap 0 uses the last frame of a slice as the opening of the next slice,
    so that moment is decoded twice. The average is the one frame that remains.
    """
    if left.shape != right.shape:
        message = f"a blend needs two equal frames, got {left.shape} and {right.shape}"
        raise ValueError(message)
    mixed = (left.astype(np.float32) + right.astype(np.float32)) * np.float32(0.5)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def crossfade(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Linear fade from ``left`` to ``right`` across the shared frames.

    Both clips are already decoded. The first shared frame stays ``left``.
    The last shared frame is ``right``.
    """
    count = int(left.shape[0])
    if count != int(right.shape[0]):
        message = f"a crossfade needs two equal spans, got {count} and {right.shape[0]}"
        raise ValueError(message)
    if count == 1:
        return np.array(right, copy=True)
    out = np.empty(left.shape, dtype=np.float32)
    for index in range(count):
        weight = index / (count - 1)
        out[index] = (1.0 - weight) * left[index] + weight * right[index]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def pin_opening(sigma: float, clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Known frames at one flow knot: ``(1-σ) clean + σ noise``."""
    level = np.float32(sigma)
    return (np.float32(1.0) - level) * clean + level * noise


def match_opening_color(pixels: np.ndarray) -> np.ndarray:
    """Give every later frame the first frame's per-channel mean and spread."""
    if int(pixels.shape[0]) < 2:
        return pixels
    ref = pixels[0].astype(np.float32)
    ref_mean = ref.mean(axis=(0, 1))
    ref_std = ref.std(axis=(0, 1))
    matched = np.empty_like(pixels)
    matched[0] = pixels[0]
    for index in range(1, int(pixels.shape[0])):
        frame = pixels[index].astype(np.float32)
        mean = frame.mean(axis=(0, 1))
        std = np.maximum(frame.std(axis=(0, 1)), np.float32(1.0e-6))
        scaled = (frame - mean) / std * ref_std + ref_mean
        matched[index] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return matched


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


def condition_volume(
    noise: np.ndarray, latent: np.ndarray, frames: int, *, end: bool = False
) -> np.ndarray:
    """36 channels: noise, the packed mask, then the VAE latent."""
    mask = mask_latent(frames, end=end)
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
