"""Whether the reply reports the passkey from a read that contained it."""

from __future__ import annotations

from typing import Any


def problems(
    reply: str,
    number: str,
    needle_line: int,
    reads: list[dict[str, Any]],
    *,
    retrieve: bool = False,
) -> list[str]:
    """Empty when the reply contains the number and one read returned its line.

    ``retrieve`` also requires the file to be read to the end and the last
    window to start after the needle line.
    """
    found: list[str] = []
    if number not in reply:
        found.append("reply missed the number")
    if not covered(needle_line, reads):
        found.append("no read covered the needle line")
    if retrieve and not finished(reads):
        found.append("the file was not read to the end")
    if retrieve and not last_window_after(needle_line, reads):
        found.append("the last window still held the needle line")
    return found


def _windows(reads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reads that returned at least one line. An empty page is not a window."""
    kept: list[dict[str, Any]] = []
    for body in reads:
        rows = body.get("lines")
        if isinstance(rows, list) and rows:
            kept.append(body)
    return kept


def finished(reads: list[dict[str, Any]]) -> bool:
    """True when the last window has no ``next_offset``."""
    windows = _windows(reads)
    if not windows:
        return False
    return "next_offset" not in windows[-1]


def last_window_after(needle_line: int, reads: list[dict[str, Any]]) -> bool:
    """True when the last window starts on a line after the needle."""
    windows = _windows(reads)
    if not windows:
        return False
    rows = windows[-1].get("lines")
    if not isinstance(rows, list) or not rows:
        return False
    first = rows[0]
    if not isinstance(first, dict):
        return False
    line = first.get("line")
    return isinstance(line, int) and line > needle_line


def covered(needle_line: int, reads: list[dict[str, Any]]) -> bool:
    """True when a tool result includes the one-based needle line."""
    for body in reads:
        rows = body.get("lines")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("line") == needle_line:
                return True
    return False
