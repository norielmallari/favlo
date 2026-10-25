"""One-second decoder tiles.

Each output sample of the Oobleck stack reads a fixed neighborhood of the
latent. A tile is that neighborhood for one second of audio. The convolution
padding supplies the zeros outside the clip, so the crop starts at the first
real latent frame the span reads.
"""

from __future__ import annotations

from ace_step_15_turbo.schedule import HOP

VALID_FRAMES = 25


def dependency(audio_lo: int, audio_hi: int) -> tuple[int, int]:
    """Inclusive latent indices the output span reads, before clipping to the clip."""
    if audio_hi < audio_lo:
        message = "the audio span is empty"
        raise ValueError(message)
    lo, hi = _conv_span(audio_lo, audio_hi, 7, 3, 1, 1)
    for stride in (2, 4, 4, 6, 10):
        for dilation in (9, 3, 1):
            lo, hi = _residual_back(lo, hi, dilation)
        lo, hi = _conv_transpose_span(lo, hi, stride * 2, stride // 2, stride)
    return _conv_span(lo, hi, 7, 3, 1, 1)


def plan(frames: int) -> list[tuple[int, int, int, int]]:
    """Crops ``(latent_lo, latent_hi, audio_lo, audio_hi)`` that cover the clip.

    ``latent_hi`` and ``audio_hi`` are exclusive. The latent crop is the real
    frames the audio span reads.
    """
    if frames < 1:
        message = "the clip has no latent frames"
        raise ValueError(message)
    total = frames * HOP
    step = VALID_FRAMES * HOP
    tiles: list[tuple[int, int, int, int]] = []
    audio = 0
    while audio < total:
        end = min(total, audio + step)
        lo, hi = dependency(audio, end - 1)
        crop_lo = min(frames - 1, max(0, lo))
        crop_hi = max(crop_lo + 1, min(frames, hi + 1))
        tiles.append((crop_lo, crop_hi, audio, end))
        audio = end
    return tiles


def tile_frames(frames: int) -> int:
    """Latent frames in the widest tile. The decoder peak uses this length."""
    return max(hi - lo for lo, hi, _audio_lo, _audio_hi in plan(frames))


def _conv_span(
    lo: int,
    hi: int,
    kernel: int,
    pad: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    return lo * stride - pad, hi * stride + (kernel - 1) * dilation - pad


def _residual_back(lo: int, hi: int, dilation: int) -> tuple[int, int]:
    lo, hi = _conv_span(lo, hi, 1, 0, 1, 1)
    pad = ((7 - 1) * dilation) // 2
    return _conv_span(lo, hi, 7, pad, 1, dilation)


def _conv_transpose_span(lo: int, hi: int, kernel: int, pad: int, stride: int) -> tuple[int, int]:
    low = max(0, lo + pad - kernel + 1)
    high = hi + pad
    if low > high:
        message = "the upsample reads no latent"
        raise RuntimeError(message)
    first = ((low + stride - 1) // stride) * stride
    last = (high // stride) * stride
    if first > last:
        message = "the upsample reads no latent"
        raise RuntimeError(message)
    return first // stride, last // stride
