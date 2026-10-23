"""Host checkpoint for a clip the listener paused.

The device is closed at the boundary. Arrays on this object are the finished
stages. A later call skips those stages and opens the device again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_STAGES = ("start", "text", "lyric", "timbre", "step", "decode", "encode")


class ClipPaused(Exception):
    """The clip stopped between stages. The job stays in progress."""

    def __init__(self, reason: str, hold: ClipHold) -> None:
        super().__init__(reason)
        self.reason = reason
        self.hold = hold


@dataclass
class ClipHold:
    stage: str
    step: int = 0
    done: int = 0
    text: np.ndarray | None = None
    lyric: np.ndarray | None = None
    timbre: np.ndarray | None = None
    noise: np.ndarray | None = None
    audio: np.ndarray | None = None
    wav: bytes | None = None

    def reached(self, stage: str) -> bool:
        """True when ``stage`` has finished. Its host output is on this object."""
        return _STAGES.index(self.stage) >= _STAGES.index(stage)


def interrupt(progress: Any) -> str | None:
    """``cancel`` or ``preempt`` when the listener has asked the stage to stop."""
    ask = getattr(progress, "interrupt", None)
    if not callable(ask):
        return None
    reason = ask()
    if reason in {"preempt", "cancel"}:
        return str(reason)
    return None
