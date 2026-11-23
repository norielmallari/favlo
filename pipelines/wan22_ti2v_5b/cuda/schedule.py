"""Frame grid shared with the Metal sampler. 24 fps, frames are 4n+1."""

from __future__ import annotations

FPS = 24


def frames_from_duration(seconds: float) -> int:
    if seconds <= 0:
        message = "duration must be positive"
        raise ValueError(message)
    counted = max(int(seconds * FPS), 5)
    return 1 + 4 * ((counted - 1) // 4)


def snap_frames(seconds: float) -> int:
    raw = max(int(round(seconds * FPS)), 5)
    remainder = (raw - 1) % 4
    snapped = raw - remainder
    if remainder >= 2:
        snapped += 4
    return max(snapped, 5)
