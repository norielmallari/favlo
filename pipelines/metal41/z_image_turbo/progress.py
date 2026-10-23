"""Console rows for one image.

The load row counts bytes read from the packed record. The sample row counts
finished layers: each caption layer through the penultimate one, the caption
projection, each noise refiner, each context refiner, each main layer, the
final readout, and each decoder piece. A paused call closes that row. The next
call opens another for the work it still runs.
"""

from __future__ import annotations

from typing import Any

from z_image_turbo.dit import LAYERS as DIT_LAYERS
from z_image_turbo.text import LAYERS as TEXT_LAYERS

CAPTION_PIECES = TEXT_LAYERS - 1 + 1
REFINER_BLOCKS = 2
STEP_PIECES = REFINER_BLOCKS + REFINER_BLOCKS + DIT_LAYERS + 1
# conv in, two mid resnets, mid attention, four up blocks of three resnets,
# three upsamplers (nearest and conv), norm, conv out, and the uint8 clip.
DECODER_PIECES = 1 + 2 + 1 + 4 * 3 + 3 * 2 + 2 + 1


def sample_total(*, steps: int, done_steps: int) -> int:
    """Layers this call will finish. A resumed step runs the caption again."""
    if steps < 1:
        message = f"steps must be positive, got {steps}"
        raise ValueError(message)
    if done_steps < 0:
        message = f"done_steps must be non-negative, got {done_steps}"
        raise ValueError(message)
    remaining = max(0, steps - done_steps)
    caption = CAPTION_PIECES if done_steps < steps else 0
    return caption + remaining * STEP_PIECES + DECODER_PIECES


class LoadProgress:
    """One load row. ``open`` is false when the console has no bar."""

    def __init__(self, console: Any, detail: str) -> None:
        self.console = console
        self.detail = detail
        self.total = 0
        self.row: Any = None

    def open(self, total: int) -> bool:
        if total < 1:
            return False
        self.total = total
        console = self.console
        if console is None or not getattr(console, "shows_bars", False):
            return False
        self.row = console.begin("load", total=total, unit="bytes", detail=self.detail)
        return True

    def advance(self, nbytes: int) -> None:
        row = self.row
        if row is None or nbytes <= 0:
            return
        row.done = min(self.total, row.done + nbytes)

    def finish(self, summary: str) -> None:
        if self.row is not None:
            self.row.done = self.total
        self.close(summary)

    def close(self, summary: str) -> None:
        row = self.row
        self.row = None
        end = getattr(self.console, "end", None) if self.console is not None else None
        if row is not None and callable(end):
            end(row, summary)


class ImageProgress:
    """One sample row and the lines logged around it."""

    def __init__(self, console: Any, label: str) -> None:
        self.console = console
        self.label = label
        self.total = 0
        self.done = 0
        self.row: Any = None

    def open(self, total: int) -> None:
        if total < 1:
            message = f"layers must be positive, got {total}"
            raise ValueError(message)
        self.total = total
        if self.label:
            self.note(self.label)
            self.label = ""
        console = self.console
        if self.row is None and console is not None and getattr(console, "shows_bars", False):
            self.row = console.begin("sample", total=total, unit="layers", detail="")

    def stage(self, name: str) -> None:
        if self.row is not None:
            self.row.detail = name

    def advance(self) -> None:
        self.done += 1
        if self.total > 0:
            self.done = min(self.total, self.done)
        if self.row is not None:
            self.row.done = self.done

    def note(self, line: str) -> None:
        log = getattr(self.console, "log", None) if self.console is not None else None
        if callable(log):
            log(line)
            return
        print(line, flush=True)

    def summary(self, *, failed: bool = False) -> str:
        if failed:
            return "sample failed"
        return f"sample {self.done}/{self.total} layers"

    def close(self, summary: str) -> None:
        row = self.row
        self.row = None
        end = getattr(self.console, "end", None) if self.console is not None else None
        if row is not None and callable(end):
            end(row, summary)
