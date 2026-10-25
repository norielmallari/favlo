"""Stage rows for one clip.

Each stage closes its own console row. The job percent is a share of the
clip: text, each expert load, the VAE, and the pack take fixed shares, and
the DiT walks take what remains.
"""

from __future__ import annotations

import time
from typing import Any

# Natural weights. DiT walks keep at least one point each, and these shrink to fit.
_TEXT = 8
_ENCODE = 8
_LOAD = 4
_DECODE = 12
_PACK = 2

_ROW = {
    "compile": ("sample", "layers", "compile", "text"),
    "text": ("sample", "layers", "text", "text"),
    "encode": ("sample", "blocks", "encode", "encode"),
    "load": ("load", "files", "expert", "load"),
    "high": ("sample", "layers", "high", "step"),
    "low": ("sample", "layers", "low", "step"),
    "decode": ("decode", "blocks", "vae", "decode"),
    "pack": ("", "", "pack", "pack"),
}


def clip_label(width: int, height: int, frames: int, fps: int) -> str:
    """Size, frame count, and duration for the first stage row."""
    if fps < 1:
        message = f"fps must be positive, got {fps}"
        raise ValueError(message)
    text = f"{frames / fps:.4f}".rstrip("0").rstrip(".")
    seconds = text or "0"
    return f"{width}x{height} {frames} frames {seconds}s"


def stage_shares(*, text: bool, vae: bool, loads: int, steps: int) -> dict[str, Any]:
    """Percent points for each stage. The values sum to 100."""
    if steps < 1 or loads < 0:
        message = f"steps must be positive and loads non-negative, got {steps}, {loads}"
        raise ValueError(message)
    labels: list[str] = []
    weights: list[int] = []
    if text:
        labels.append("text")
        weights.append(_TEXT)
    if vae:
        labels.append("encode")
        weights.append(_ENCODE)
    for _index in range(loads):
        labels.append("load")
        weights.append(_LOAD)
    if vae:
        labels.append("decode")
        weights.append(_DECODE)
    labels.append("pack")
    weights.append(_PACK)
    scaled = _fit(weights, max(0, 100 - steps))
    remain = 100 - sum(scaled)
    base, extra = divmod(remain, steps)
    shares: dict[str, Any] = {
        "text": 0,
        "encode": 0,
        "load": [],
        "decode": 0,
        "pack": 0,
        "step": [base + (1 if index < extra else 0) for index in range(steps)],
    }
    for label, value in zip(labels, scaled, strict=True):
        if label == "load":
            shares["load"].append(value)
        else:
            shares[label] = value
    return shares


def _fit(weights: list[int], cap: int) -> list[int]:
    """Scale ``weights`` so they sum to at most ``cap``."""
    total = sum(weights)
    if total <= cap:
        return list(weights)
    if cap <= 0 or total <= 0:
        return [0 for _weight in weights]
    raw = [cap * weight / total for weight in weights]
    out = [int(value) for value in raw]
    left = cap - sum(out)
    order = sorted(range(len(weights)), key=lambda index: raw[index] - out[index], reverse=True)
    for index in order[:left]:
        out[index] += 1
    return out


class ClipProgress:
    """One job percent and a console row for the stage that is running."""

    def __init__(
        self,
        *,
        job: dict[str, Any] | None = None,
        console: Any = None,
        clip: str = "",
    ) -> None:
        self.done = 0
        self.job = job
        self.console = console
        self.clip = clip
        self.row: Any = None
        self.started = time.perf_counter()
        self._count = True
        self.earned = 0
        self._share = 0
        self._stage_done = 0
        self._stage_total = 0
        self._kind = ""
        self._detail = ""
        self._unit = ""
        self._summary = ""
        self._text_taken = False
        self._text_remainder = 0
        self._text_share = 0
        self._encode_share = 0
        self._decode_share = 0
        self._pack_share = 0
        self._load_shares: list[int] = []
        self._step_shares: list[int] = []
        self.text_layers = 0
        self.pipelines = 0
        self.files = 0
        self.encode_blocks = 0
        self.decode_blocks = 0
        self.dit_layers = 0

    def open(
        self,
        *,
        text_layers: int,
        pipelines: int,
        loads: int,
        steps: int,
        encode_blocks: int,
        decode_blocks: int,
        dit_layers: int,
        files: int,
    ) -> None:
        """Store the stage totals. The first ``stage`` opens the console row."""
        shares = stage_shares(
            text=text_layers > 0 or pipelines > 0,
            vae=encode_blocks > 0 or decode_blocks > 0,
            loads=loads,
            steps=steps,
        )
        self.text_layers = text_layers
        self.pipelines = pipelines
        self.files = files
        self.encode_blocks = encode_blocks
        self.decode_blocks = decode_blocks
        self.dit_layers = dit_layers
        self._text_share = int(shares["text"])
        self._encode_share = int(shares["encode"])
        self._decode_share = int(shares["decode"])
        self._pack_share = int(shares["pack"])
        self._load_shares = [int(value) for value in shares["load"]]
        self._step_shares = [int(value) for value in shares["step"]]
        self.started = time.perf_counter()

    def restore(self, done: int, stage: str) -> None:
        """Keep the percent a paused clip already earned."""
        self.done = done
        self.earned = done
        if stage == "compile":
            self._text_taken = True
            self._text_remainder = max(0, self._text_share - done)
        self._sync()

    def stage(self, name: str) -> None:
        """Close the previous row and open the row for ``name``."""
        if isinstance(self.job, dict) and self.job.get("status") == "in_progress":
            self.job["stage"] = name
        if name == "text" and self._kind == "text" and self._count and self.row is not None:
            self._detail = "text"
            self.row.detail = "text"
            return
        self._finish_row(None)
        if not self._count:
            return
        self._open_row(name)

    def summary(self, *, failed: bool = False) -> str:
        """The line printed when the clip's row closes."""
        if failed:
            return "sample failed"
        if self._summary:
            return self._summary
        return "sample done"

    def counting(self, enabled: bool) -> None:
        """When false, ``advance`` ignores a piece that was already counted."""
        self._count = enabled

    def advance(self) -> None:
        """Count one finished piece of the open stage."""
        if not self._count or self._stage_total < 1:
            return
        self._stage_done = min(self._stage_total, self._stage_done + 1)
        filled = self._share * self._stage_done // self._stage_total
        self.done = min(99, self.earned + filled)
        self._summary = f"{self._detail} {self._stage_done}/{self._stage_total} {self._unit}"
        self._sync()
        if self.row is not None:
            self.row.done = self._stage_done

    def note(self, line: str) -> None:
        """One measured line. A console records it. Otherwise it is printed."""
        log = getattr(self.console, "log", None) if self.console is not None else None
        if callable(log):
            log(line)
            return
        print(line, flush=True)

    def complete(self) -> None:
        """The MP4 is stored. The caller marks the job completed immediately after this."""
        self._lock()
        self.done = 100
        self.earned = 100
        if isinstance(self.job, dict):
            self.job["progress"] = 100
            self.job["eta"] = 0
            self.job["stage"] = None
        if self.row is not None and self._stage_total:
            self.row.done = self._stage_total

    def close(self, summary: str) -> None:
        self._finish_row(summary)

    def _open_row(self, name: str) -> None:
        spec = _ROW.get(name)
        if spec is None:
            return
        phase, unit, detail, kind = spec
        total = self._total(name)
        if total < 1:
            return
        self._share = self._take(kind)
        self._stage_done = 0
        self._stage_total = total
        self._kind = kind
        self._detail = detail
        self._unit = unit
        if not phase:
            return
        if self.clip:
            self.note(self.clip)
            self.clip = ""
        console = self.console
        if console is not None and getattr(console, "shows_bars", False):
            self.row = console.begin(phase, total=total, unit=unit, detail=detail)

    def _total(self, name: str) -> int:
        if name == "compile":
            return self.pipelines + self.text_layers
        if name == "text":
            return self.text_layers
        if name == "encode":
            return self.encode_blocks
        if name == "load":
            return self.files
        if name in {"high", "low"}:
            return self.dit_layers
        if name == "decode":
            return self.decode_blocks
        if name == "pack":
            return 1
        return 0

    def _take(self, kind: str) -> int:
        if kind == "text":
            if self._text_taken:
                value = self._text_remainder
                self._text_remainder = 0
                return value
            self._text_taken = True
            return self._text_share
        if kind == "encode":
            value = self._encode_share
            self._encode_share = 0
            return value
        if kind == "load":
            if not self._load_shares:
                return 0
            return self._load_shares.pop(0)
        if kind == "step":
            if not self._step_shares:
                return 0
            return self._step_shares.pop(0)
        if kind == "decode":
            value = self._decode_share
            self._decode_share = 0
            return value
        if kind == "pack":
            value = self._pack_share
            self._pack_share = 0
            return value
        return 0

    def _lock(self) -> None:
        if self._share <= 0 or self._stage_total <= 0:
            return
        filled = self._share * min(self._stage_done, self._stage_total) // self._stage_total
        self.earned = min(99, self.earned + filled)
        self.done = self.earned
        self._share = 0
        self._sync()

    def _finish_row(self, summary: str | None) -> None:
        self._lock()
        row = self.row
        self.row = None
        self._kind = ""
        self._stage_done = 0
        self._stage_total = 0
        if row is None:
            return
        end = getattr(self.console, "end", None) if self.console is not None else None
        if callable(end):
            end(row, summary or self._summary or "sample done")

    def _sync(self) -> None:
        if not isinstance(self.job, dict) or self.job.get("status") != "in_progress":
            return
        self.job["progress"] = min(99, self.done)
        self.job["eta"] = self._eta()

    def _eta(self) -> int | None:
        """Remaining seconds from the finished percent of the clip."""
        if self.done < 1:
            return None
        if self.done >= 100:
            return 0
        elapsed = time.perf_counter() - self.started
        if elapsed <= 0:
            return None
        return max(0, int((100 - self.done) * elapsed / self.done))
