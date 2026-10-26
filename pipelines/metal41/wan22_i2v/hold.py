"""Host checkpoint for a clip the listener paused.

The device is closed at the boundary. Arrays on this object are the finished
stages. A later call skips those stages and opens the device again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from favlo.media_pipeline import MediaPaused

_STAGES = ("compile", "text", "encode", "step", "decode", "pack")


class ClipPaused(MediaPaused):
    """The clip stopped between stages. The job stays in progress."""

    def __init__(self, reason: str, hold: ClipHold) -> None:
        super().__init__(reason, hold)


@dataclass
class ClipHold:
    stage: str
    step: int = 0
    expert: str | None = None
    done: int = 0
    table: np.ndarray | None = None
    table_null: np.ndarray | None = None
    noise: np.ndarray | None = None
    cond: np.ndarray | None = None
    mask: np.ndarray | None = None
    pixels: np.ndarray | None = None
    avi: bytes | None = None
    step_times: list[float] = field(default_factory=list)

    def reached(self, stage: str) -> bool:
        """True when ``stage`` has finished and its host outputs are on this object."""
        return _STAGES.index(self.stage) >= _STAGES.index(stage)


def reload_quietly(hold: ClipHold | None, expert: str, index: int) -> bool:
    """True when this expert was already counted for the step about to resume."""
    if hold is None or hold.stage != "step" or hold.expert != expert:
        return False
    return index == hold.step


def interrupt(progress: Any) -> str | None:
    """``cancel`` or ``preempt`` when the listener has asked the stage to stop."""
    ask = getattr(progress, "interrupt", None)
    if not callable(ask):
        return None
    reason = ask()
    if reason in {"preempt", "cancel"}:
        return str(reason)
    return None
