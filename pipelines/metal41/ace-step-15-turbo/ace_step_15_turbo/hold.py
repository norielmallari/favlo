"""Host checkpoint for a clip the listener paused.

The device is closed at the boundary. Arrays on this object are the finished
stages. A later call skips those stages and opens the device again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STAGES = ("start", "text", "lyric", "timbre", "step", "decode", "encode")


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
