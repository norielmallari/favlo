"""Status row, durable lines, and the versioned event stream.

The process entry owns this module. It imports nothing from a pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from favlo.logs import RollingLog

PHASES = ("load", "download", "quantize", "prefill", "decode")
UNITS = ("tok", "layers", "bytes", "shards", "experts")
VERSION = "0.1.0"
SCHEMA = 1

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_FILL = "█"
_BAR_TRACK = "░"
_BAR_LEFT = "▕"
_BAR_RIGHT = "▏"
_PLATE_TOP = "▕███████████████████████▏"
_PLATE_MID = "▕██     F A V L O     ██▏"
_PLATE_BOT = "▕███████████████████████▏"
_TICK = 0.1
_PROMPT_QUIET = 0.25
_CSI_FINAL = set(
    "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)
_WINDOW = 5.0
_WINDOW_MIN = 0.5


@dataclass
class Progress:
    """One phase. The request thread is the only writer of these fields."""

    phase: str = ""
    done: int = 0
    total: int = 0
    unit: str = "tok"
    detail: str = ""
    gauge: tuple[str, int] | None = None
    request: int | None = None
    pipeline: str = ""
    started: float = 0.0
    summary: str = ""
    closed: bool = False


class Console:
    """Pretty row, plain lines, or NDJSON. ``auto`` follows the terminal."""

    def __init__(
        self,
        mode: str,
        *,
        pipeline: str,
        stream: TextIO | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        width: int | None = None,
        color: int | None = None,
        start_thread: bool = True,
    ) -> None:
        if mode not in {"pretty", "plain", "json"}:
            message = f"unknown console mode {mode}"
            raise ValueError(message)
        self.mode = mode
        self.pipeline = pipeline
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock or time.time
        self._sleep = sleep or time.sleep
        self._width = width
        self._color = self._resolve_color() if color is None else color
        self._lock = threading.RLock()
        self._handles: list[Progress] = []
        self._windows: dict[int, list[tuple[float, int]]] = {}
        self._drawn = 0
        self._inline = False
        self._wrap_off = False
        self._cursor_hidden = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_thread = start_thread and mode in {"pretty", "json"}
        self.fold_turns = False
        self.live_decode = mode == "pretty"
        self._recorder: Callable[[str], None] | None = None
        self._remote: dict[tuple[str, int | None], Progress] = {}
        self._draft: str | None = None
        self._status_lines = 0
        self._pending_summaries: list[str] = []
        self._column = 0
        self._activity = time.monotonic()
        self._holding = False
        self._held: list[dict[str, Any]] = []

    @classmethod
    def from_env(
        cls,
        choice: str | None,
        *,
        pipeline: str,
        stream: TextIO | None = None,
    ) -> Console:
        """``auto`` is pretty on a TTY. ``NO_COLOR`` and ``TERM=dumb`` force plain."""
        selected = choice or "auto"
        if selected not in {"auto", "pretty", "plain", "json"}:
            message = f"unknown console mode {selected}"
            raise ValueError(message)
        target = stream if stream is not None else sys.stderr
        mode = selected
        if selected == "auto":
            dumb = os.environ.get("TERM") == "dumb" or os.environ.get("NO_COLOR") is not None
            tty = _is_tty(target)
            mode = "plain" if dumb or not tty else "pretty"
        return cls(mode, pipeline=pipeline, stream=target)

    def attach_log(self, path: Path, max_bytes: int) -> None:
        """Copy durable lines into a rolling file. The row itself is not copied."""
        self._recorder = RollingLog(path, max_bytes).write

    def _record(self, text: str) -> None:
        recorder = self._recorder
        if recorder is None or not text:
            return
        recorder(text)

    @property
    def keeps_lines(self) -> bool:
        """Plain and json keep today's durable lines. Pretty replaces them with the row."""
        return self.mode != "pretty"

    @property
    def shows_bars(self) -> bool:
        """Pretty paints a row. JSON emits the same phase. Plain does not."""
        return self.mode != "plain"

    def plate(self) -> None:
        """The Ingot Plate, once, before any other output."""
        self._record(f"favlo {VERSION}")
        if self.mode == "plain":
            self._write(f"favlo {VERSION}\n")
            return
        if self.mode == "json":
            self._event({"type": "log", "line": f"favlo {VERSION}"})
            return
        rows = (_PLATE_TOP, _PLATE_MID, _PLATE_BOT)
        for row in rows:
            self._write(self._paint_plate(row) + "\n")

    def log(self, line: str) -> None:
        """A durable line. The listening line stays plain text in every mode."""
        self._record(line)
        if self.mode == "json":
            self._event({"type": "log", "line": line})
            return
        if self.mode == "pretty" and self._handles:
            return
        self._erase()
        self._write(line + "\n")
        self._redraw()

    def draft(self, text: str) -> None:
        """Show a reply that is still arriving. The terminal wraps it.

        An open prefill or decode row stays on the line under the reply and
        keeps its tok/s.
        """
        with self._lock:
            if self.mode != "pretty":
                self._draft = text
                return
            self._enable_wrap()
            if self._draft is None:
                self._erase()
                self._write(text)
            elif text.startswith(self._draft):
                self._clear_status_below()
                self._write(text[len(self._draft) :])
            else:
                self._clear_status_below()
                self._write("\n")
                self._write(text)
            self._draft = text
            self._paint_status_below(self._status_rows())

    def say(self, line: str) -> None:
        """A line that stays on screen while a row is open."""
        self._record(line)
        if self.mode == "json":
            self._event({"type": "log", "line": line})
            self._draft = None
            return
        with self._lock:
            if self.mode == "pretty" and self._draft is not None:
                self._clear_status_below()
                same = line == self._draft or line == self._draft.rstrip()
                if same:
                    if not self._draft.endswith("\n"):
                        self._write("\n\r")
                else:
                    if not self._draft.endswith("\n"):
                        self._write("\n\r")
                    self._enable_wrap()
                    self._write(line + "\n")
                self._draft = None
                self._flush_summaries()
                if self._handles:
                    self._redraw()
                else:
                    self._show_cursor()
                return
            self._draft = None
            self._erase()
            self._enable_wrap()
            self._write(line + "\n")
            self._redraw()

    def prompt(self, after_output: bool = False) -> None:
        """``you:`` at the start of a line, after this turn's telemetry is drawn."""
        if after_output:
            self._wait_quiet(_PROMPT_QUIET)
        with self._lock:
            if after_output and self._column != 0:
                self._write("\n")
            self._write("you: ")

    def hold(self, active: bool) -> None:
        """Keep telemetry until the reply is finished, then print those lines."""
        if active:
            with self._lock:
                self._holding = True
            return
        with self._lock:
            queued = self._held
            self._held = []
            self._holding = False
        for line in _telemetry_lines(queued):
            self.log(line)

    def apply_event(self, event: dict[str, Any]) -> None:
        """Mirror one daemon event onto this console."""
        with self._lock:
            if self._holding:
                self._held.append(event)
                return
        kind = event.get("type")
        if kind == "log":
            line = event.get("line")
            if isinstance(line, str) and line:
                self.log(line)
            return
        if kind == "begin":
            self._mirror_begin(event)
            return
        if kind == "tick":
            self._mirror_tick(event)
            return
        if kind == "end":
            self._mirror_end(event)

    def begin(
        self,
        phase: str,
        *,
        total: int = 0,
        unit: str = "tok",
        detail: str = "",
        request: int | None = None,
        gauge: tuple[str, int] | None = None,
    ) -> Progress:
        if phase not in PHASES:
            message = f"unknown console phase {phase}"
            raise ValueError(message)
        if unit not in UNITS:
            message = f"unknown console unit {unit}"
            raise ValueError(message)
        progress = Progress(
            phase=phase,
            total=total,
            unit=unit,
            detail=detail,
            request=request,
            gauge=gauge,
            pipeline=self.pipeline,
            started=self._clock(),
        )
        with self._lock:
            self._handles.append(progress)
        if self.mode == "json":
            self._event(self._boundary("begin", progress))
        if self.mode == "pretty":
            self._hide_cursor()
            self.sample()
        self._ensure_thread()
        return progress

    def end(self, progress: Progress, summary: str) -> None:
        if progress.closed:
            return
        if summary:
            self._record(summary)
        progress.summary = summary
        progress.closed = True
        with self._lock:
            if progress in self._handles:
                self._handles.remove(progress)
            self._windows.pop(id(progress), None)
            remaining = len(self._handles)
            if self.mode == "pretty" and self._draft is not None:
                if summary:
                    self._pending_summaries.append(summary)
                self._paint_status_below(self._status_rows())
                return
            if self.mode == "pretty":
                self._erase()
                if summary:
                    self._write(self._dim(summary) + "\n")
                if remaining == 0:
                    self._show_cursor()
                else:
                    self._redraw()
                return
        if self.mode == "json":
            event = self._boundary("end", progress)
            event["summary"] = summary
            self._event(event)
        elif summary:
            self._write(summary + "\n")

    def retarget(
        self,
        progress: Progress,
        *,
        total: int,
        unit: str,
        detail: str,
        done: int,
    ) -> None:
        """Move a quantize row to the next sub-phase without resetting a same-unit window."""
        if unit not in UNITS:
            message = f"unknown console unit {unit}"
            raise ValueError(message)
        if progress.unit != unit:
            self._windows.pop(id(progress), None)
        progress.total = total
        progress.unit = unit
        progress.detail = detail
        progress.done = done

    @contextmanager
    def phase(self, name: str, **kwargs: Any) -> Iterator[Progress]:
        progress = self.begin(name, **kwargs)
        try:
            yield progress
        finally:
            if not progress.closed:
                self.end(progress, progress.summary)

    def report(self, text: str) -> None:
        """Clear the live rows, then write ``text``.

        A traceback written straight to the terminal is painted over by the
        next row tick. This finishes the rows first.
        """
        if text:
            self._record(text)
        with self._lock:
            for progress in self._handles:
                progress.closed = True
            self._handles.clear()
            self._windows.clear()
            if self.mode == "pretty":
                self._clear_status_below()
                self._draft = None
                self._erase()
                self._show_cursor()
            if text:
                self._write(text if text.endswith("\n") else text + "\n")

    def sample(self) -> None:
        """One renderer tick. Tests call this with the thread left off."""
        now = self._clock()
        with self._lock:
            handles = list(self._handles)
            if self.mode == "pretty" and self._draft is not None:
                self._paint_status_below(self._status_rows())
                return
            if not handles:
                return
            if self.mode == "pretty":
                width = self._terminal_width()
                rows = [self._fit(self._row(progress, now), width) for progress in handles]
                self._paint_rows(rows)
                return
            if self.mode != "json":
                return
            events = [self._tick(progress, now) for progress in handles]
        for event in events:
            self._event(event)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        with self._lock:
            handles = list(self._handles)
        for progress in handles:
            self.end(progress, progress.summary)
        self._show_cursor()

    def _ensure_thread(self) -> None:
        if not self._start_thread:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="favlo-console", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(_TICK):
            self.sample()

    def _mirror_key(self, event: dict[str, Any]) -> tuple[str, int | None]:
        request = event.get("request")
        phase = event.get("phase")
        name = phase if isinstance(phase, str) else ""
        return (name, request if isinstance(request, int) else None)

    def _mirror_begin(self, event: dict[str, Any]) -> None:
        phase = event.get("phase")
        unit = event.get("unit")
        if phase not in PHASES or unit not in UNITS:
            return
        key = self._mirror_key(event)
        previous = self._remote.get(key)
        if previous is not None and not previous.closed:
            self.end(previous, "")
        total = event.get("total")
        request = event.get("request")
        detail = event.get("detail")
        progress = self.begin(
            phase,
            total=total if isinstance(total, int) else 0,
            unit=unit,
            detail=detail if isinstance(detail, str) else "",
            request=request if isinstance(request, int) else None,
            gauge=_event_gauge(event.get("gauge")),
        )
        self._remote[key] = progress

    def _mirror_tick(self, event: dict[str, Any]) -> None:
        progress = self._remote.get(self._mirror_key(event))
        if progress is None or progress.closed:
            return
        done = event.get("done")
        if isinstance(done, int):
            progress.done = done
        total = event.get("total")
        if isinstance(total, int):
            progress.total = total
        detail = event.get("detail")
        if isinstance(detail, str):
            progress.detail = detail
        gauge = _event_gauge(event.get("gauge"))
        if gauge is not None:
            progress.gauge = gauge

    def _mirror_end(self, event: dict[str, Any]) -> None:
        progress = self._remote.pop(self._mirror_key(event), None)
        summary = event.get("summary") if isinstance(event.get("summary"), str) else ""
        if progress is None or progress.closed:
            if summary:
                self.log(summary)
            return
        self.end(progress, summary)

    def _boundary(self, kind: str, progress: Progress) -> dict[str, Any]:
        event: dict[str, Any] = {
            "v": SCHEMA,
            "type": kind,
            "pipeline": progress.pipeline,
            "phase": progress.phase,
            "total": progress.total,
            "unit": progress.unit,
            "detail": progress.detail,
            "ts": self._clock(),
        }
        if progress.request is not None:
            event["request"] = progress.request
        return event

    def _tick(self, progress: Progress, now: float) -> dict[str, Any]:
        rate, _display_eta, eta = self._rates(progress, now)
        event: dict[str, Any] = {
            "v": SCHEMA,
            "type": "tick",
            "pipeline": progress.pipeline,
            "phase": progress.phase,
            "done": progress.done,
            "total": progress.total,
            "unit": progress.unit,
            "rate": rate,
            "eta": eta,
            "detail": progress.detail,
            "request": progress.request,
            "ts": now,
        }
        if progress.gauge is not None:
            event["gauge"] = [progress.gauge[0], progress.gauge[1]]
        if progress.request is None:
            event.pop("request")
        return event

    def _rates(
        self, progress: Progress, now: float
    ) -> tuple[float | None, float | None, float | None]:
        with self._lock:
            if progress.phase == "prefill" and progress.unit == "tok":
                rate, done, aged = self._cumulative_prefill(progress, now)
            else:
                rate, done, aged = self._window_rate(progress, now)
        display: float | None = None
        eta: float | None = None
        if rate is not None and rate > 0 and progress.total > done:
            display = (progress.total - done) / rate
            if aged:
                eta = display
        return rate, display, eta

    def _cumulative_prefill(
        self, progress: Progress, now: float
    ) -> tuple[float | None, int, bool]:
        """Tokens completed divided by the time since prefill started."""
        done = progress.done
        elapsed = now - progress.started
        if done <= 0 or elapsed <= 0:
            return None, done, False
        return done / elapsed, done, elapsed >= _WINDOW_MIN

    def _window_rate(
        self, progress: Progress, now: float
    ) -> tuple[float | None, int, bool]:
        samples = self._windows.setdefault(id(progress), [])
        samples.append((now, progress.done))
        cutoff = now - _WINDOW
        while len(samples) > 1 and samples[0][0] < cutoff:
            samples.pop(0)
        if len(samples) > 50:
            del samples[:-50]
        done = progress.done
        if len(samples) < 2:
            return None, done, False
        span = samples[-1][0] - samples[0][0]
        delta = samples[-1][1] - samples[0][1]
        elapsed = now - progress.started
        if span >= _WINDOW_MIN and span > 0:
            rate: float | None = delta / span
        elif elapsed > 0:
            rate = done / elapsed
        else:
            rate = None
        return rate, done, span >= _WINDOW_MIN

    def _row(self, progress: Progress, now: float) -> str:
        rate, display, _eta = self._rates(progress, now)
        if progress.total <= 0:
            return self._indeterminate(progress, now, rate)
        return self._determinate(progress, rate, display)

    def _determinate(self, progress: Progress, rate: float | None, eta: float | None) -> str:
        ratio = 0.0 if progress.total <= 0 else min(1.0, progress.done / progress.total)
        percent = f"{int(ratio * 100)}%"
        bar = _bar(ratio, 30)
        done = _count(progress.done, progress.unit)
        total = _count(progress.total, progress.unit)
        count = f"{done}/{total}" if progress.unit == "bytes" else f"{done}/{total} {progress.unit}"
        parts = [progress.phase, bar, percent, count]
        if progress.detail:
            parts.append(progress.detail)
        rate_text = _rate_text(rate, progress.unit)
        if rate_text:
            parts.append(rate_text)
        if eta is not None:
            parts.append(f"ETA {_eta_text(eta)}")
        if progress.gauge is not None:
            label, value = progress.gauge
            parts.append(f"{label} {value:,}")
        return _join_parts(parts)

    def _indeterminate(self, progress: Progress, now: float, rate: float | None) -> str:
        frame = _SPINNER[int(now * 10) % len(_SPINNER)]
        count = f"{progress.phase} {_count(progress.done, progress.unit)} {progress.unit}"
        parts = [frame]
        if progress.detail:
            parts.append(progress.detail)
        parts.append(count)
        rate_text = _rate_text(rate, progress.unit)
        if rate_text:
            parts.append(rate_text)
        if progress.gauge is not None:
            label, value = progress.gauge
            parts.append(f"{label} {value:,}")
        elapsed = now - progress.started
        if elapsed >= 0:
            parts.append(f"{elapsed:.1f}s")
        return _join_parts(parts)

    def _fit(self, line: str, width: int) -> str:
        if width <= 0 or len(line) <= width:
            return self._colorize(line)
        parts = line.split(" · ")
        while len(parts) > 1 and len(" · ".join(parts)) > width:
            parts.pop()
        fitted = " · ".join(parts)
        if len(fitted) > width:
            fitted = fitted[:width]
        return self._colorize(fitted)

    def _colorize(self, line: str) -> str:
        if self._color == 0 or " · " not in line and not line.startswith(_BAR_LEFT):
            if self._color == 0:
                return line
        phase, _, rest = line.partition(" ")
        if phase in PHASES or phase in _SPINNER:
            body = rest
            head = self._paint(phase, accent=True)
            if body.startswith(_BAR_LEFT):
                bar, sep, tail = body.partition(" ")
                body = self._paint_bar(bar) + (sep + self._dim(tail) if sep else "")
            else:
                body = self._dim(body)
            return f"{head} {body}" if rest else head
        return line

    def _paint_bar(self, bar: str) -> str:
        if self._color == 0:
            return bar
        fill, _, track = bar.partition(_BAR_TRACK)
        if not track and _BAR_TRACK not in bar:
            return self._paint(bar, accent=True)
        track_body = _BAR_TRACK + track
        return self._paint(fill, accent=True) + self._dim(track_body)

    def _paint_plate(self, row: str) -> str:
        if self._color == 0:
            return row
        return self._paint(row, accent=True)

    def _paint(self, text: str, *, accent: bool) -> str:
        if self._color == 0 or not text:
            return text
        if accent and self._color >= 2:
            return f"\x1b[38;5;75m{text}\x1b[0m"
        if accent:
            return f"\x1b[1;36m{text}\x1b[0m"
        return f"\x1b[1m{text}\x1b[0m"

    def _dim(self, text: str) -> str:
        if self._color == 0 or not text:
            return text
        return f"\x1b[2m{text}\x1b[0m"

    def _status_rows(self) -> list[str]:
        now = self._clock()
        width = self._terminal_width()
        return [self._fit(self._row(progress, now), width) for progress in self._handles]

    def _clear_status_below(self) -> None:
        """Erase the progress lines under the reply and return to the reply."""
        if self._status_lines <= 0:
            return
        self._write("\x1b7\n")
        for index in range(self._status_lines):
            self._write("\r\x1b[2K")
            if index + 1 < self._status_lines:
                self._write("\n")
        self._write("\x1b8")
        self._status_lines = 0

    def _paint_status_below(self, rows: list[str]) -> None:
        """Draw the open rows under the reply. The cursor stays on the reply."""
        self._clear_status_below()
        if not rows:
            return
        self._write("\x1b7\n")
        for index, row in enumerate(rows):
            self._write("\r\x1b[2K" + row)
            if index + 1 < len(rows):
                self._write("\n")
        self._status_lines = len(rows)
        self._write("\x1b8")

    def _paint_rows(self, rows: list[str]) -> None:
        """One open row stays on the current line. A second row uses the line below."""
        if len(rows) == 1:
            if self._drawn:
                self._write(f"\x1b[{self._drawn}F")
                self._drawn = 0
            if not self._wrap_off:
                self._write("\x1b[?7l")
                self._wrap_off = True
            self._write("\r\x1b[2K" + rows[0])
            self._inline = True
            return
        if self._inline:
            self._write("\n")
            self._inline = False
        if self._drawn:
            self._write(f"\x1b[{self._drawn}F")
        for row in rows:
            self._write(row + "\x1b[K\n")
        self._drawn = len(rows)

    def _enable_wrap(self) -> None:
        """Chat text uses the terminal's soft wrap. A progress row turns it off."""
        if self.mode != "pretty" or not self._wrap_off:
            return
        self._write("\x1b[?7h")
        self._wrap_off = False

    def _flush_summaries(self) -> None:
        pending = self._pending_summaries
        self._pending_summaries = []
        for summary in pending:
            self._write(self._dim(summary) + "\n")

    def _erase(self) -> None:
        if self.mode != "pretty":
            return
        if self._inline:
            self._write("\r\x1b[2K")
            self._inline = False
        elif self._drawn:
            self._write(f"\x1b[{self._drawn}F")
            for _ in range(self._drawn):
                self._write("\x1b[K\n")
            self._write(f"\x1b[{self._drawn}F")
            self._drawn = 0
        if self._wrap_off:
            self._write("\x1b[?7h")
            self._wrap_off = False

    def _redraw(self) -> None:
        if self.mode == "pretty" and self._handles:
            self.sample()

    def _hide_cursor(self) -> None:
        if self.mode == "pretty" and not self._cursor_hidden:
            self._write("\x1b[?25l")
            self._cursor_hidden = True

    def _show_cursor(self) -> None:
        if self._cursor_hidden:
            self._write("\x1b[?25h")
            self._cursor_hidden = False

    def _terminal_width(self) -> int:
        if self._width is not None:
            return self._width
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            return 80

    def _resolve_color(self) -> int:
        if self.mode != "pretty":
            return 0
        if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
            return 0
        term = os.environ.get("TERM", "")
        if os.environ.get("COLORTERM") or "256color" in term or term == "xterm-direct":
            return 2
        if term:
            return 1
        return 0

    def _event(self, payload: dict[str, Any]) -> None:
        payload.setdefault("v", SCHEMA)
        payload.setdefault("pipeline", self.pipeline)
        payload.setdefault("ts", self._clock())
        self._write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _wait_quiet(self, seconds: float) -> None:
        """Return once no row is open and nothing has been drawn for ``seconds``."""
        deadline = time.monotonic() + max(seconds * 4, 1.0)
        while time.monotonic() < deadline:
            with self._lock:
                idle_for = time.monotonic() - self._activity
                busy = bool(self._handles)
            if not busy and idle_for >= seconds:
                return
            self._sleep(0.02)

    def _write(self, text: str) -> None:
        if text:
            self._activity = time.monotonic()
            self._advance(text)
        self.stream.write(text)
        self.stream.flush()

    def _advance(self, text: str) -> None:
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\x1b":
                index += 1
                if index < len(text) and text[index] == "[":
                    index += 1
                    while index < len(text) and text[index] not in _CSI_FINAL:
                        index += 1
                index += 1
                continue
            if char in "\n\r":
                self._column = 0
            elif char >= " ":
                self._column += 1
            index += 1


def _telemetry_lines(events: list[dict[str, Any]]) -> list[str]:
    """The durable lines from one turn. Ticks stay off the reply."""
    lines: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind == "log":
            line = event.get("line")
        elif kind == "end":
            line = event.get("summary")
        else:
            continue
        if isinstance(line, str) and line:
            lines.append(line)
    return lines


def _event_gauge(value: Any) -> tuple[str, int] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    ):
        return (value[0], value[1])
    return None


def _is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


def _bar(ratio: float, cells: int) -> str:
    filled = int(round(ratio * cells))
    filled = max(0, min(cells, filled))
    return _BAR_LEFT + (_BAR_FILL * filled) + (_BAR_TRACK * (cells - filled)) + _BAR_RIGHT


def _count(value: int, unit: str) -> str:
    if unit == "bytes":
        return _bytes(value)
    if unit == "tok":
        return f"{value:,}"
    return str(value)


def _bytes(value: float) -> str:
    if abs(value) >= 1 << 30:
        return f"{value / (1 << 30):.1f} GiB"
    if abs(value) >= 1 << 20:
        return f"{value / (1 << 20):.1f} MiB"
    return f"{value:.0f} B"


def _rate_text(rate: float | None, unit: str) -> str:
    if rate is None or rate < 0:
        return ""
    shown = rate
    suffix = "/s"
    if unit in {"layers", "shards", "experts"} and rate < 0.5:
        shown = rate * 60
        suffix = "/min"
    if unit == "bytes":
        return f"{_bytes(shown)}{suffix}"
    if unit == "tok":
        text = f"{shown:.1f}" if shown < 100 else f"{shown:.0f}"
        return f"{text} tok/s"
    text = f"{shown:.1f}" if shown < 100 else f"{shown:.0f}"
    return f"{text} {unit}{suffix}"


def _eta_text(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    if seconds < 90 * 60:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}:{secs:02d}"
    hours, rem = divmod(int(seconds), 3600)
    return f"{hours}:{rem // 60:02d}"


def _join_parts(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    head, bar, *rest = parts
    if bar.startswith(_BAR_LEFT):
        tail = " · ".join(rest)
        return f"{head} {bar} {tail}" if tail else f"{head} {bar}"
    return " · ".join(parts)
