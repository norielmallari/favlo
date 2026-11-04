"""Status row, durable lines, and the versioned event stream.

The process entry owns this module. It imports nothing from a pipeline.
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from favlo.daemon import TracebackSkip
from favlo.logs import RollingLog
from favlo.progress import Sanitizer

PHASES = ("load", "download", "quantize", "prefill", "decode", "sample")
UNITS = ("tok", "layers", "bytes", "shards", "experts", "stages", "files", "blocks")
VERSION = "0.4.0"
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
_PAINT_CAP = 0.3
_FRAME_WAIT = 2.0
_TURN_IDLE = 0.4
_TURN_ORDER = ("prefill", "decode", "sample", "prefix")
_STASH_PHASES = frozenset({"prefill", "decode", "sample"})
_CSI_FINAL = set(
    "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)
_WINDOW_MIN = 0.5
_SMOOTH = 1.5


def publish_count(console: Any, progress: Any, done: int) -> None:
    """Enqueue ``done`` on ``console``. With no console, store it on the handle."""
    if progress is None or isinstance(done, bool):
        return
    report = getattr(console, "count", None) if console is not None else None
    if report is None:
        report = getattr(getattr(progress, "ui", None), "count", None)
    if callable(report):
        report(progress, int(done))
        return
    progress.done = int(done)


def format_rate(phase: str, count: int, seconds: float, *, unit: str = "tok") -> str:
    """Durable close line for a counted phase. The harness parses this form."""
    if count <= 0 or seconds <= 0:
        return f"{phase} 0 {unit}/s"
    return f"{phase} {count / seconds:.2f} {unit}/s ({count} {unit} in {seconds:.2f}s)"


def format_note(topic: str, **fields: Any) -> str:
    """Canonical diagnostic line. Pretty folds these. Plain and json keep them."""
    if topic == "prompt":
        return f"prompt tokens {int(fields['tokens'])}"
    if topic == "cache":
        hit = int(fields.get("hit") or 0)
        return "cache miss" if hit <= 0 else f"cache hit {hit}"
    if topic == "prefix_load":
        return f"prefix load {int(fields['tokens'])} tokens"
    if topic == "prefix_store":
        return f"prefix store {int(fields['tokens'])} tokens"
    if topic == "prefix_store_failed":
        return f"prefix store failed {fields['error']}"
    if topic == "mtp_depth":
        return f"mtp depth {int(fields['depth'])}"
    if topic == "mtp":
        return f"mtp {int(fields['accepts'])}/{int(fields['tries'])}"
    if topic == "incoming":
        return f"incoming tokens {int(fields['tokens'])}"
    if topic == "output":
        return f"output tokens {int(fields['tokens'])}"
    if topic == "context":
        atlas = fields.get("atlas")
        if atlas is None:
            return f"context {int(fields['tokens'])}"
        return f"context {int(fields['tokens'])} atlas {int(atlas)} images"
    message = f"unknown console note {topic}"
    raise ValueError(message)


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
    elapsed: float = 0.0
    elapsed_at: float = 0.0
    fed: bool = False
    ui: Any = None


class Console:
    """Pretty row, plain lines, or NDJSON. ``auto`` follows the terminal."""

    def __init__(
        self,
        mode: str,
        *,
        pipeline: str,
        stream: TextIO | None = None,
        clock: Callable[[], float] | None = None,
        wall: Callable[[], float] | None = None,
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
        self.load_detail = ""
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock or time.perf_counter
        self._wall = wall or time.time
        self._sleep = sleep or time.sleep
        self._width = width
        self._color = self._resolve_color() if color is None else color
        self._lock = threading.RLock()
        self._handles: list[Progress] = []
        self._display: dict[tuple[int, str], tuple[float, float]] = {}
        self._drawn = 0
        self._inline = False
        self._wrap_off = False
        self._cursor_hidden = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Every mode may enqueue begin/end/note so a slow pipe does not stall
        # the GPU thread. Tests pass start_thread=False and keep the sync path.
        self._start_thread = start_thread
        self._actions: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self.fold_turns = False
        self.live_decode = mode == "pretty"
        self._recorder: Callable[[str], None] | None = None
        self._remote: dict[tuple[Any, str, int | None], Progress] = {}
        self._sanitizer = Sanitizer(frozenset(PHASES), frozenset(UNITS))
        self._count_paint_at = 0.0
        self._behind = False
        self._draft: str | None = None
        self._status_lines = 0
        self._pending_summaries: list[str] = []
        self._column = 0
        self._holding = False
        self._reply: str | None = None
        self._opens = 0
        self._closes = 0
        self._follower = False
        self._caught_at = 0.0
        self._at_prompt = False
        self._restore_prompt = False
        self._traceback = TracebackSkip()
        self._turn_depth = 0
        self._turn_parts: list[str] = []
        self._turn_due: float | None = None

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
        ui = cls(mode, pipeline=pipeline, stream=target)
        ui.fold_turns = mode == "pretty"
        return ui

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
        if self.mode == "pretty" and (_folded_log(line) or self._traceback.drop(line)):
            return
        if self.mode == "pretty" and self.fold_turns and _turn_key(line) is not None:
            with self._lock:
                if self._turn_depth > 0:
                    self._stash_turn(line)
                    return
        self._note_prompt_displaced()
        self._erase()
        self._write(self._stamp(line) + "\n")
        self._redraw()

    def note(self, topic: str, **fields: Any) -> None:
        """A structured diagnostic. Pretty folds it. Plain and json keep the line.

        With the console thread running, the call enqueues and returns. The
        console thread formats and writes. Prefill and decode do not wait.
        """
        if self._start_thread:
            self._actions.put(("note", topic, dict(fields)))
            self._ensure_thread()
            return
        self.log(format_note(topic, **fields))

    def draft(self, text: str) -> None:
        """Show a reply that is still arriving. The terminal wraps it.

        While a turn is open, the reply waits until that turn's telemetry is drawn.
        """
        with self._lock:
            if self._holding:
                self._reply = text
                return
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
        with self._lock:
            if self._holding:
                self._reply = line
                return
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
        """A bare line. Telemetry already on screen stays above it."""
        with self._lock:
            if after_output and self._column != 0:
                self._write("\n")
            self._at_prompt = True
            self._restore_prompt = False

    def note_caught_up(self) -> None:
        """The follower has applied every complete line now in the log."""
        with self._lock:
            self._follower = True
            self._caught_at = time.monotonic()

    def hold(self, active: bool) -> None:
        """Paint telemetry now. Print the reply after every request has closed."""
        if active:
            with self._lock:
                self._holding = True
                self._opens = 0
                self._closes = 0
                self._reply = None
                self._at_prompt = False
                self._restore_prompt = False
            return
        self._release_frame()

    def _release_frame(self) -> None:
        started = time.monotonic()
        deadline = started + _FRAME_WAIT
        while True:
            with self._lock:
                if not self._holding:
                    return
                balanced = self._opens == self._closes
                caught = self._caught_at >= started
                if not self._follower:
                    ready = balanced
                else:
                    ready = balanced and self._closes >= 1 and caught
                if ready or time.monotonic() >= deadline:
                    reply = self._reply
                    self._reply = None
                    self._holding = False
                    self._opens = 0
                    self._closes = 0
                    break
            self._sleep(0.02)
        if reply:
            with self._lock:
                self._flush_turn(force=True)
            self.say(reply)

    def apply_event(self, event: dict[str, Any]) -> None:
        """Enqueue one daemon event. The console thread sanitizes it and paints."""
        if self._start_thread:
            self._actions.put(("remote", event))
            self._ensure_thread()
            return
        self._accept(event)

    def _accept(self, event: dict[str, Any]) -> None:
        for item in self._sanitizer.accept(event, pretty=self.mode == "pretty"):
            self._paint_remote(item)

    def _paint_remote(self, event: dict[str, Any]) -> None:
        """Apply one sanitized event. The caller is the console thread."""
        restore = False
        with self._lock:
            line = _log_line(event)
            if line == "request open":
                self._turn_depth += 1
                self._turn_due = None
                if self._holding:
                    self._opens += 1
                return
            if line == "request closed":
                self._turn_depth = max(0, self._turn_depth - 1)
                if self.fold_turns and self._turn_depth == 0:
                    self._turn_due = self._clock() + _TURN_IDLE
                if self._holding:
                    self._closes += 1
                elif self._restore_prompt and not self._handles:
                    self._restore_prompt = False
                    restore = True
                if not restore:
                    return
        if restore:
            self.prompt(after_output=True)
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
        started: float | None = None,
    ) -> Progress:
        """Open a phase row.

        With the console thread running, the call stores the handle and
        enqueues the paint. Prefill and decode do not wait for the write.
        """
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
            started=self._clock() if started is None else started,
        )
        progress.ui = self
        with self._lock:
            self._handles.append(progress)
            self._turn_due = None
        if self._start_thread:
            self._actions.put(("begin", progress))
            self._ensure_thread()
            return progress
        self._emit_begin(progress)
        return progress

    def end(self, progress: Progress, summary: str) -> None:
        """Close a phase row.

        With the console thread running, the call marks the row closed and
        enqueues the summary write. Prefill and decode do not wait for it.
        """
        if progress.closed:
            return
        progress.summary = summary
        progress.closed = True
        if self._start_thread:
            self._actions.put(("end", progress, summary))
            self._ensure_thread()
            return
        self._emit_end(progress, summary)

    def count(self, progress: Progress, done: int) -> None:
        """Enqueue a count and return. The console thread emits it.

        ``done`` is copied into the fact. The caller does not wait for the
        paint, and it does not leave the rate clock on a shared field.
        """
        if progress.closed or isinstance(done, bool) or not isinstance(done, int):
            return
        elapsed = self._clock() - progress.started
        if elapsed < 0 or not math.isfinite(elapsed):
            elapsed = 0.0
        fact = (
            done,
            progress.total,
            progress.unit,
            progress.detail,
            progress.gauge,
            elapsed,
        )
        progress.fed = True
        if self._start_thread:
            self._actions.put(("count", progress, fact))
            self._ensure_thread()
            return
        self._apply_count(progress, fact)

    def _apply_count(
        self,
        progress: Progress,
        fact: tuple[int, int, str, str, tuple[str, int] | None, float],
    ) -> None:
        if progress.closed:
            return
        done, total, unit, detail, gauge, elapsed = fact
        progress.done = done
        progress.total = total
        progress.unit = unit
        progress.detail = detail
        if gauge is not None:
            progress.gauge = gauge
        progress.elapsed = elapsed
        progress.elapsed_at = self._clock()
        progress.fed = True
        if self.mode == "json":
            self._event(self._tick(progress, self._clock()))
            return
        self._paint_count()

    def _paint_count(self) -> None:
        """Paint a live count. A backed-up queue redraws at most every 300 ms."""
        now = self._clock()
        behind = not self._actions.empty()
        if behind and now - self._count_paint_at < _PAINT_CAP:
            self._behind = True
            return
        self._behind = False
        self.sample()

    def _emit_begin(self, progress: Progress) -> None:
        if self.mode != "plain":
            self._note_prompt_displaced()
        if self.mode == "json":
            self._event(self._boundary("begin", progress))
        if self.mode == "pretty":
            self._hide_cursor()
            self.sample()

    def _emit_end(self, progress: Progress, summary: str) -> None:
        if summary:
            self._record(summary)
        with self._lock:
            if progress in self._handles:
                self._handles.remove(progress)
            self._forget(progress)
            remaining = len(self._handles)
            if self.mode == "pretty" and self._draft is not None:
                if summary:
                    if (
                        self.fold_turns
                        and progress.phase in _STASH_PHASES
                        and self._turn_depth > 0
                    ):
                        self._stash_turn(summary)
                    else:
                        self._pending_summaries.append(summary)
                self._paint_status_below(self._status_rows())
                return
            if self.mode == "pretty":
                self._note_prompt_displaced()
                self._erase()
                if summary:
                    if (
                        self.fold_turns
                        and progress.phase in _STASH_PHASES
                        and self._turn_depth > 0
                    ):
                        self._stash_turn(summary)
                    else:
                        self._write(self._stamp(summary, dim=True) + "\n")
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
            self._note_prompt_displaced()
            self._write(self._stamp(summary) + "\n")

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
            self._forget(progress)
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
            self._display.clear()
            if self.mode == "pretty":
                self._clear_status_below()
                self._draft = None
                self._erase()
                self._show_cursor()
                parts = self._turn_parts
                self._turn_parts = []
                self._turn_due = None
                if parts:
                    self._write(self._stamp("  ".join(parts), dim=True) + "\n")
            if text:
                self._write(text if text.endswith("\n") else text + "\n")

    def sample(self) -> None:
        """One renderer tick. Tests call this with the thread left off."""
        now = self._clock()
        self._count_paint_at = now
        with self._lock:
            self._flush_turn()
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
            events = [
                self._tick(progress, now)
                for progress in handles
                if not progress.fed and not progress.closed
            ]
        for event in events:
            self._event(event)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        self._drain_actions()
        with self._lock:
            handles = list(self._handles)
        for progress in handles:
            if not progress.closed:
                progress.closed = True
            self._emit_end(progress, progress.summary)
        with self._lock:
            self._flush_turn(force=True)
        self._show_cursor()

    def flush(self) -> None:
        """Drain queued begin/end/note work. Tests use this when the thread is off."""
        if self._start_thread:
            self._actions.put(("flush",))
            self._ensure_thread()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if self._actions.empty():
                    return
                time.sleep(0.01)
            return
        self._drain_actions()

    def _ensure_thread(self) -> None:
        if not self._start_thread:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="favlo-console", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain_actions()
            if self.mode == "pretty" and (
                not self._behind or self._clock() - self._count_paint_at >= _PAINT_CAP
            ):
                self.sample()
                self._behind = False
            elif self.mode == "json":
                self.sample()
            self._stop.wait(_TICK)
        self._drain_actions()

    def _drain_actions(self) -> None:
        while True:
            try:
                action = self._actions.get_nowait()
            except queue.Empty:
                return
            kind = action[0]
            if kind == "begin":
                self._emit_begin(action[1])
            elif kind == "end":
                self._emit_end(action[1], action[2])
            elif kind == "note":
                self.log(format_note(action[1], **action[2]))
            elif kind == "count":
                self._apply_count(action[1], action[2])
            elif kind == "remote":
                self._accept(action[1])
            elif kind == "flush":
                continue

    def _mirror_key(self, event: dict[str, Any]) -> tuple[Any, str, int | None]:
        request = event.get("request")
        phase = event.get("phase")
        name = phase if isinstance(phase, str) else ""
        source = event.get("source")
        if isinstance(source, bool) or not isinstance(source, (int, str)):
            source = ""
        request_id = request if isinstance(request, int) and not isinstance(request, bool) else None
        return (source, name, request_id)

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
            started=_event_stamp(event),
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
        self._take_elapsed(progress, event)
        self._paint_count()

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
            "ts": progress.started if kind == "begin" else self._clock(),
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
            "elapsed": self._elapsed(progress, now),
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
            rate, done, aged = self._cumulative(progress, now)
        display: float | None = None
        eta: float | None = None
        if rate is not None and rate > 0 and progress.total > done:
            display = (progress.total - done) / rate
            if aged:
                eta = display
        return rate, display, eta

    def _cumulative(
        self, progress: Progress, now: float
    ) -> tuple[float | None, int, bool]:
        """Work completed divided by the time since the phase started.

        A window shorter than ``_WINDOW_MIN`` does not publish a rate. One
        token recorded in the same instant as ``begin`` is not a throughput.
        """
        done = progress.done
        elapsed = self._elapsed(progress, now)
        if done <= 0 or elapsed < _WINDOW_MIN:
            return None, done, False
        return done / elapsed, done, True

    def _elapsed(self, progress: Progress, now: float) -> float:
        """Seconds since the phase began, on one clock.

        A count carries the producer's elapsed time. Time since that count
        was applied uses the painter's clock, so a child timestamp is never
        subtracted from this process's counter.
        """
        if progress.elapsed > 0 and progress.elapsed_at > 0:
            return progress.elapsed + max(0.0, now - progress.elapsed_at)
        return now - progress.started

    def _take_elapsed(self, progress: Progress, event: dict[str, Any]) -> None:
        elapsed = event.get("elapsed")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            return
        progress.elapsed = float(elapsed)
        progress.elapsed_at = self._clock()

    def _row(self, progress: Progress, now: float) -> str:
        rate, _display, eta = self._rates(progress, now)
        shown = self._smooth(progress, "rate", now, rate)
        shown_eta = None
        if eta is not None and shown is not None and shown > 0 and progress.total > progress.done:
            shown_eta = (progress.total - progress.done) / shown
        gauge = self._smooth_gauge(progress, now)
        if progress.total <= 0:
            return self._indeterminate(progress, now, shown, gauge)
        return self._determinate(progress, shown, shown_eta, gauge)

    def _forget(self, progress: Progress) -> None:
        key = id(progress)
        self._display.pop((key, "rate"), None)
        self._display.pop((key, "gauge"), None)

    def _smooth(
        self, progress: Progress, name: str, now: float, value: float | None
    ) -> float | None:
        """Exponential smooth of a painted number. The JSON tick stays raw."""
        key = (id(progress), name)
        if value is None:
            held = self._display.get(key)
            return None if held is None else held[1]
        previous = self._display.get(key)
        if previous is None:
            self._display[key] = (now, value)
            return value
        last, smoothed = previous
        dt = now - last
        if dt > 0:
            smoothed += (1.0 - math.exp(-dt / _SMOOTH)) * (value - smoothed)
        self._display[key] = (now, smoothed)
        return smoothed

    def _smooth_gauge(self, progress: Progress, now: float) -> tuple[str, int] | None:
        gauge = progress.gauge
        if gauge is None:
            self._display.pop((id(progress), "gauge"), None)
            return None
        label, value = gauge
        smoothed = self._smooth(progress, "gauge", now, float(value))
        if smoothed is None:
            return None
        return label, int(round(smoothed))

    def _determinate(
        self,
        progress: Progress,
        rate: float | None,
        eta: float | None,
        gauge: tuple[str, int] | None,
    ) -> str:
        ratio = 0.0 if progress.total <= 0 else min(1.0, progress.done / progress.total)
        percent = f"{int(ratio * 100)}%"
        bar = _bar(ratio, 30)
        done = _count(progress.done, progress.unit)
        total = _count(progress.total, progress.unit)
        count = f"{done}/{total}" if progress.unit == "bytes" else f"{done}/{total} {progress.unit}"
        label = _row_label(progress)
        parts = [label, bar, percent, count]
        if progress.detail and progress.detail != label:
            parts.append(progress.detail)
        rate_text = _rate_text(rate, progress.unit)
        if rate_text:
            parts.append(rate_text)
        if eta is not None:
            parts.append(f"ETA {_eta_text(eta)}")
        if gauge is not None:
            label, value = gauge
            parts.append(f"{label} {value:,}")
        return _join_parts(parts)

    def _indeterminate(
        self,
        progress: Progress,
        now: float,
        rate: float | None,
        gauge: tuple[str, int] | None,
    ) -> str:
        frame = _SPINNER[int(now * 10) % len(_SPINNER)]
        label = _row_label(progress)
        count = f"{label} {_count(progress.done, progress.unit)} {progress.unit}"
        parts = [frame]
        if progress.detail and progress.detail != label:
            parts.append(progress.detail)
        parts.append(count)
        rate_text = _rate_text(rate, progress.unit)
        if rate_text:
            parts.append(rate_text)
        if gauge is not None:
            label, value = gauge
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
        if phase in PHASES or phase in _SPINNER or rest.startswith(_BAR_LEFT):
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
            self._write(self._stamp(summary, dim=True) + "\n")

    def _stamp(self, line: str, *, dim: bool = False) -> str:
        """Local wall clock on a durable stderr line. The recorded body stays bare."""
        stamped = f"[{time.strftime('%H:%M:%S', time.localtime(self._wall()))}] {line}"
        return self._dim(stamped) if dim else stamped

    def _stash_turn(self, line: str) -> None:
        key = _turn_key(line)
        if key is None:
            return
        self._turn_parts = [item for item in self._turn_parts if _turn_key(item) != key]
        self._turn_parts.append(line)
        self._turn_parts.sort(
            key=lambda item: _TURN_ORDER.index(_turn_key(item) or "prefix")
        )

    def _flush_turn(self, *, force: bool = False) -> None:
        if not self.fold_turns or self.mode != "pretty":
            return
        if self._holding and not force:
            return
        if not force and (self._turn_due is None or self._clock() < self._turn_due):
            return
        if self._handles or self._turn_depth > 0:
            return
        parts = self._turn_parts
        self._turn_parts = []
        self._turn_due = None
        if not parts:
            return
        line = "  ".join(parts)
        self._note_prompt_displaced()
        self._erase()
        self._write(self._stamp(line, dim=True) + "\n")
        if self._handles:
            self._redraw()
        else:
            self._show_cursor()

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

    def _note_prompt_displaced(self) -> None:
        """A foreign request is painting over the prompt. Restore the bare line when it closes."""
        with self._lock:
            if not self._at_prompt or self._holding:
                return
            self._at_prompt = False
            self._restore_prompt = True
            if self.mode == "pretty" and not self._inline and self._drawn == 0:
                self._write("\r\x1b[2K")
            elif self.mode != "pretty" and self._column != 0:
                self._write("\n")

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

    def _write(self, text: str) -> None:
        if text:
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


def _log_line(event: dict[str, Any]) -> str | None:
    if event.get("type") != "log":
        return None
    line = event.get("line")
    if isinstance(line, str) and line:
        return line
    return None


def _folded_log(line: str) -> bool:
    """True when pretty should keep the line in the log and off the terminal."""
    if line == "cache miss":
        return True
    return line.startswith(
        (
            "prompt tokens ",
            "cache hit ",
            "prefix load ",
            "mtp ",
            "incoming tokens ",
            "output tokens ",
            "context ",
        )
    )


def _turn_key(line: str) -> str | None:
    """Kind used to collapse one request's durable lines."""
    if line.startswith("prefix store ") and "failed" not in line:
        return "prefix"
    first = line.split(" ", 1)[0]
    if first in _STASH_PHASES:
        return first
    return None


def _event_stamp(event: dict[str, Any]) -> float | None:
    """Producer clock on a mirrored event. The same counter paints the row."""
    stamp = event.get("ts")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return None
    value = float(stamp)
    if not math.isfinite(value):
        return None
    return value


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
    if unit in {"layers", "shards", "experts", "files", "blocks"} and rate < 0.5:
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


def _row_label(progress: Progress) -> str:
    """The word painted on the row. A sample row uses its stage detail."""
    if progress.phase == "sample" and progress.detail:
        return progress.detail
    return progress.phase


def _join_parts(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    head, bar, *rest = parts
    if bar.startswith(_BAR_LEFT):
        tail = " · ".join(rest)
        return f"{head} {bar} {tail}" if tail else f"{head} {bar}"
    return " · ".join(parts)
