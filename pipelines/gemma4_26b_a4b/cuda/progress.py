"""Console rows for load, prefill, and decode."""

from __future__ import annotations

from typing import Any

from favlo.console import format_note, format_rate, publish_count


def note(console: Any, kind: str, **fields: Any) -> None:
    if console is None:
        return
    log = getattr(console, "note", None)
    if callable(log):
        log(kind, **fields)
        return
    plain = getattr(console, "log", None)
    if callable(plain):
        plain(format_note(kind, **fields))


def begin_load(console: Any, total: int) -> Any:
    if console is None or total <= 0 or not getattr(console, "shows_bars", False):
        return None
    return console.begin("load", total=total, unit="bytes")


def note_load(progress: Any, step: int) -> None:
    if progress is None or step <= 0:
        return
    done = min(int(progress.total), int(getattr(progress, "_reported", 0)) + int(step))
    progress._reported = done
    publish_count(None, progress, done)


def begin_phase(console: Any, phase: str, total: int, *, live: bool = False) -> Any:
    if console is None:
        return None
    if live:
        if not getattr(console, "live_decode", False):
            return None
        total = 0
    elif total <= 0:
        return None
    return console.begin(phase, total=total, unit="tok")


def note_count(console: Any, progress: Any, done: int) -> None:
    if progress is None:
        return
    total = int(getattr(progress, "total", 0) or 0)
    capped = min(total, int(done)) if total else int(done)
    publish_count(console, progress, capped)


def end_phase(console: Any, progress: Any, summary: str) -> None:
    if console is None or progress is None:
        return
    end = getattr(console, "end", None)
    if callable(end):
        end(progress, summary)


def rate(phase: str, count: int, seconds: float) -> str:
    return format_rate(phase, count, seconds)
