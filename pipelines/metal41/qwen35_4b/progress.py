"""Console progress helpers. Writes stay off the hot path via ``favlo.console``."""

from __future__ import annotations

from typing import Any

from favlo.console import format_note, format_rate, publish_count


def begin_load(console: Any, total: int, detail: str) -> Any:
    if console is None or total <= 0 or not getattr(console, "shows_bars", False):
        return None
    return console.begin("load", total=total, unit="bytes", detail=detail)


def note_load(progress: Any, step: int) -> None:
    if progress is None or step <= 0:
        return
    done = min(int(progress.total), int(getattr(progress, "_reported", 0)) + int(step))
    progress._reported = done
    publish_count(None, progress, done)


def begin_phase(
    console: Any,
    phase: str,
    total: int,
    *,
    request: int | None = None,
    live: bool = False,
) -> Any:
    if console is None:
        return None
    if live:
        if not getattr(console, "live_decode", False):
            return None
        total = 0
    elif total <= 0:
        return None
    detail = f"req #{request}" if request else ""
    return console.begin(
        phase,
        total=total,
        unit="tok",
        detail=detail,
        request=request,
    )


def note_prefill(progress: Any, done: int) -> None:
    if progress is None:
        return
    capped = min(int(progress.total), int(done))
    if capped <= int(getattr(progress, "_reported", 0)):
        return
    progress._reported = capped
    publish_count(None, progress, capped)


def note_decode(progress: Any, done: int) -> None:
    if progress is None:
        return
    publish_count(None, progress, int(done))


def end_phase(console: Any, progress: Any, summary: str) -> None:
    if progress is None:
        if summary and console is None:
            print(summary, flush=True)
        elif summary and console is not None:
            log = getattr(console, "log", None)
            if callable(log):
                log(summary)
            else:
                print(summary, flush=True)
        return
    if getattr(progress, "closed", False):
        return
    if console is None:
        if summary:
            print(summary, flush=True)
        return
    console.end(progress, summary)


def note(console: Any, topic: str, **fields: Any) -> None:
    if console is None:
        print(format_note(topic, **fields), flush=True)
        return
    writer = getattr(console, "note", None)
    if callable(writer):
        writer(topic, **fields)
        return
    log = getattr(console, "log", None)
    if callable(log):
        log(format_note(topic, **fields))
        return
    print(format_note(topic, **fields), flush=True)


def rate(phase: str, count: int, seconds: float) -> str:
    return format_rate(phase, count, seconds)
