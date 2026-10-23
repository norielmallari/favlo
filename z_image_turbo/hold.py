"""Host checkpoint for an image the listener paused.

The caption embedding stays on the device, so a resume runs that stage again.
A finished Euler step keeps its latent here. A finished decode keeps the uint8
image. The runtime object does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ImagePaused(Exception):
    """The image stopped between stages. The job stays in progress."""

    def __init__(self, reason: str, hold: ImageHold) -> None:
        super().__init__(reason)
        self.reason = reason
        self.hold = hold


@dataclass
class ImageHold:
    stage: str
    step: int = 0
    latent: np.ndarray | None = None
    pixels: np.ndarray | None = None
    times: list[float] = field(default_factory=list)
    caption_s: float = 0.0
    vae_s: float = 0.0
    spent: float = 0.0


@dataclass
class ImageJob:
    id: str
    created: int
    prompt: str
    seed: int
    width: int
    height: int
    count: int
    quality: str
    index: int = 0
    images: list[dict[str, str]] = field(default_factory=list)
    timings: list[dict[str, Any]] = field(default_factory=list)
    hold: ImageHold | None = None


def ask(stop: Any) -> str | None:
    """``cancel`` or ``preempt`` when the listener has asked the stage to stop."""
    if not callable(stop):
        return None
    reason = stop()
    if reason in {"preempt", "cancel"}:
        return str(reason)
    return None
