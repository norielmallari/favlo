"""Layer count for one clip.

The count is finished work: text layers, packed weights, shader pipelines,
DiT layers, and VAE blocks. It is not a fraction of the elapsed time.
"""

from __future__ import annotations

import time
from typing import Any


def clip_label(width: int, height: int, frames: int, fps: int) -> str:
    """Size, frame count, and duration for the sample row."""
    if fps < 1:
        message = f"fps must be positive, got {fps}"
        raise ValueError(message)
    text = f"{frames / fps:.4f}".rstrip("0").rstrip(".")
    seconds = text or "0"
    return f"{width}x{height} {frames} frames {seconds}s"


def sample_total(
    *,
    steps: int,
    dit_layers: int,
    loads: int,
    weights: int,
    text_layers: int,
    vae_blocks: int,
    pipelines: int,
    gpus: int,
) -> int:
    """Pieces the sample row will finish."""
    total = text_layers + loads * weights + steps * dit_layers + vae_blocks + gpus * pipelines
    if total < 1:
        message = "a clip has no measured work"
        raise ValueError(message)
    return total


class ClipProgress:
    """One sample row and the job fields a poll reads."""

    def __init__(
        self,
        *,
        job: dict[str, Any] | None = None,
        console: Any = None,
        clip: str = "",
    ) -> None:
        self.total = 0
        self.done = 0
        self.job = job
        self.console = console
        self.clip = clip
        self.row: Any = None
        self.started = time.perf_counter()
        self._count = True

    def open(self, total: int) -> None:
        """Open the row once the piece count is known. The clip label is logged once."""
        if total < 1:
            message = f"layers must be positive, got {total}"
            raise ValueError(message)
        self.total = total
        self.started = time.perf_counter()
        if self.clip:
            self.note(self.clip)
            self.clip = ""
        console = self.console
        if self.row is None and console is not None and getattr(console, "shows_bars", False):
            self.row = console.begin("sample", total=total, unit="layers", detail="")

    def stage(self, name: str) -> None:
        """Name the stage that is about to run."""
        if isinstance(self.job, dict) and self.job.get("status") == "in_progress":
            self.job["stage"] = name
        if self.row is not None:
            self.row.detail = name

    def summary(self, *, failed: bool = False) -> str:
        """The line printed when the row closes."""
        if failed:
            return "sample failed"
        return f"sample {self.done}/{self.total} layers"

    def counting(self, enabled: bool) -> None:
        """When false, ``advance`` ignores a piece that was already counted."""
        self._count = enabled

    def advance(self) -> None:
        """Count one finished piece. The job stays below 100 until the caller completes it."""
        if not self._count:
            return
        self.done += 1
        if self.total > 0:
            self.done = min(self.total, self.done)
            if isinstance(self.job, dict) and self.job.get("status") == "in_progress":
                self.job["progress"] = min(99, (100 * self.done) // self.total)
                self.job["eta"] = self._eta()
        if self.row is not None:
            self.row.done = self.done

    def note(self, line: str) -> None:
        """One measured line. A console records it. Otherwise it is printed."""
        log = getattr(self.console, "log", None) if self.console is not None else None
        if callable(log):
            log(line)
            return
        print(line, flush=True)

    def complete(self) -> None:
        """The MP4 is stored. The caller marks the job completed immediately after this."""
        self.done = self.total
        if isinstance(self.job, dict):
            self.job["progress"] = 100
            self.job["eta"] = 0
            self.job["stage"] = None
        if self.row is not None:
            self.row.done = self.total

    def _eta(self) -> int | None:
        """Remaining seconds from the finished-layer rate."""
        if self.done < 1 or self.total <= 0:
            return None
        if self.done >= self.total:
            return 0
        elapsed = time.perf_counter() - self.started
        if elapsed <= 0:
            return None
        return max(0, int((self.total - self.done) * elapsed / self.done))

    def close(self, summary: str) -> None:
        row = self.row
        self.row = None
        end = getattr(self.console, "end", None) if self.console is not None else None
        if row is not None and callable(end):
            end(row, summary)
