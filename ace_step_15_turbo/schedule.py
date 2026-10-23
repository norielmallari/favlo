"""Turbo schedule, clip length, and the published condition text."""

from __future__ import annotations

import math

import numpy as np

LATENT_RATE = 25
SAMPLE_RATE = 48_000
HOP = SAMPLE_RATE // LATENT_RATE
MIN_SECONDS = 10.0
MAX_SECONDS = 600.0
SHIFT = 3.0
STEPS = 8
TIMBRE_FRAMES = 30 * LATENT_RATE
TEXT_LIMIT = 256
LYRIC_LIMIT = 2048
INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


def latent_frames(seconds: float) -> int:
    """25 Hz frames for one clip. The count is what the VAE hop produces."""
    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
        message = f"seconds must be from {MIN_SECONDS:g} to {MAX_SECONDS:g}"
        raise ValueError(message)
    return int(math.ceil(seconds * LATENT_RATE))


def timestep_schedule(steps: int = STEPS, shift: float = SHIFT) -> np.ndarray:
    """Shifted nodes from 1 toward 0, without the terminal 0.

    ``t' = shift * t / (1 + (shift - 1) * t)`` on ``linspace(1, 0, steps + 1)``.
    """
    nodes = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
    if shift != 1.0:
        nodes = shift * nodes / (1.0 + (shift - 1.0) * nodes)
    return nodes[:-1]


def metadata_text(
    seconds: float,
    *,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
) -> str:
    bpm_text = str(bpm) if bpm is not None and bpm > 0 else "N/A"
    signature = timesignature.strip() if timesignature and timesignature.strip() else "N/A"
    key = keyscale.strip() if keyscale and keyscale.strip() else "N/A"
    duration = f"{int(seconds)} seconds" if seconds > 0 else "30 seconds"
    return (
        f"- bpm: {bpm_text}\n"
        f"- timesignature: {signature}\n"
        f"- keyscale: {key}\n"
        f"- duration: {duration}\n"
    )


def condition_text(
    prompt: str,
    lyrics: str,
    seconds: float,
    vocal_language: str,
    *,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    instruction: str | None = None,
) -> tuple[str, str]:
    """Caption string and lyric string the text encoder was trained on."""
    lead = INSTRUCTION if instruction is None else instruction
    if not lead.endswith(":"):
        lead = lead + ":"
    metas = metadata_text(seconds, bpm=bpm, keyscale=keyscale, timesignature=timesignature)
    caption = f"# Instruction\n{lead}\n\n# Caption\n{prompt}\n\n# Metas\n{metas}<|endoftext|>\n"
    lyric = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"
    return caption, lyric
