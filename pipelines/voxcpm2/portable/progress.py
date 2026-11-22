"""Console row for one spoken line.

The sample row counts each diffusion step of each patch, then the decoder.
Its detail names the patch and the Euler step before that step's estimates
run, and the row advances when those command buffers have finished. The
published patch ceiling is the row's total until the stop head ends the line.
A finished line fills that row. A paused call closes it. The next call opens
another.
"""

from __future__ import annotations

import time
from typing import Any

from favlo.console import publish_count

# Published ``generate`` default, then the vendor clamp ``tokens * 6 + 10``.
PATCH_CAP = 4096
LENGTH_RATIO = 6.0
LENGTH_PAD = 10


def patch_limit(text_tokens: int, max_len: int, context: int) -> int:
    """Patches this line may draw.

    ``text_tokens`` is the styled line without the audio-start id. The result
    is the smaller of ``max_len``, ``text_tokens * 6 + 10``, and the context
    left after those tokens and the audio-start id.
    """
    if text_tokens < 0 or max_len < 1 or context < 1:
        message = (
            "patch cap needs a text length, a positive max, and a context, "
            f"got {text_tokens}, {max_len}, {context}"
        )
        raise ValueError(message)
    room = context - (text_tokens + 1)
    if room < 1:
        message = "the line does not fit in the context"
        raise ValueError(message)
    span = int(text_tokens * LENGTH_RATIO) + LENGTH_PAD
    return min(max_len, span, room)


def sample_total(patches: int, steps: int) -> int:
    """Diffusion steps across the patch ceiling, plus the decoder."""
    if patches < 1 or steps < 1:
        message = f"patches and steps must be positive, got {patches}, {steps}"
        raise ValueError(message)
    return patches * steps + 1


class VoiceProgress:
    """One sample row, the job fields, and the lines logged around them."""

    def __init__(self, console: Any, job: dict[str, Any] | None = None) -> None:
        self.console = console
        self.job = job
        self.total = 0
        self.done = 0
        self.patches = 0
        self.decode_s = 0.0
        self.row: Any = None
        self.stage_name = ""
        self.started = 0.0

    def open(self, total: int) -> None:
        if total < 1:
            message = f"stages must be positive, got {total}"
            raise ValueError(message)
        self.total = total
        self.done = 0
        self.started = time.perf_counter()
        console = self.console
        if self.row is None and console is not None and getattr(console, "shows_bars", False):
            self.row = console.begin("sample", total=total, unit="stages", detail="")

    def stage(self, name: str) -> None:
        self.stage_name = name
        if self.row is not None:
            self.row.detail = name
        self._sync()

    def advance(self) -> None:
        self.done += 1
        if self.total > 0:
            self.done = min(self.total, self.done)
        if self.row is not None:
            publish_count(self.console, self.row, self.done)
        self._sync()

    def note(self, line: str) -> None:
        log = getattr(self.console, "log", None) if self.console is not None else None
        if callable(log):
            log(line)
            return
        print(line, flush=True)

    def complete(self) -> None:
        """The line has its patches. The row fills to the work that ran."""
        if self.done < 1:
            return
        self.total = self.done
        if self.row is not None:
            self.row.total = self.done
            publish_count(self.console, self.row, self.done)
        self._sync(final=True)

    def summary(self, *, failed: bool = False) -> str:
        if failed:
            return "sample failed"
        return f"sample {self.done}/{self.total} stages"

    def close(self, summary: str) -> None:
        row = self.row
        self.row = None
        end = getattr(self.console, "end", None) if self.console is not None else None
        if row is not None and callable(end):
            end(row, summary)

    def _sync(self, *, final: bool = False) -> None:
        job = self.job
        if not isinstance(job, dict) or job.get("status") != "in_progress":
            return
        job["stage"] = self.stage_name
        if self.total < 1 or self.done < 1:
            return
        job["progress"] = 99 if final else min(99, (100 * self.done) // self.total)
        if final or self.done >= self.total:
            job["eta"] = 0
            return
        elapsed = time.perf_counter() - self.started
        remain = self.total - self.done
        if elapsed <= 0:
            job["eta"] = 0
            return
        job["eta"] = max(0, int(remain * elapsed / self.done))
