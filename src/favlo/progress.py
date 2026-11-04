"""Turn one raw progress event into the paint operations the console applies.

The pipeline process enqueues facts. A reader forwards each fact here. This
module drops a count for a phase that has not begun or has already ended, and
it leaves a shutdown traceback out of the pretty console. It does not compute
a rate and it does not write the terminal.
"""

from __future__ import annotations

import math
from typing import Any

from favlo.daemon import TracebackSkip


class Sanitizer:
    """Open rows, keyed by the pipeline process, the phase, and the request."""

    def __init__(self, phases: frozenset[str], units: frozenset[str]) -> None:
        self._phases = phases
        self._units = units
        self._open: set[tuple[Any, str, int | None]] = set()
        self._skip = TracebackSkip()

    def accept(self, event: dict[str, Any], *, pretty: bool = False) -> list[dict[str, Any]]:
        """Paint operations for one raw event, in the order the painter applies them."""
        kind = event.get("type")
        if kind == "log":
            line = event.get("line")
            if not isinstance(line, str) or not line:
                return []
            if pretty and self._skip.drop(line):
                return []
            return [event]
        if kind == "begin":
            return self._begin(event)
        if kind in {"tick", "count"}:
            return self._count(event)
        if kind == "end":
            return self._end(event)
        return []

    def _begin(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        phase = event.get("phase")
        unit = event.get("unit")
        if phase not in self._phases or unit not in self._units:
            return []
        self._open.add(_key(event))
        return [event]

    def _count(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        if _key(event) not in self._open:
            return []
        done = event.get("done")
        if isinstance(done, bool) or not isinstance(done, int):
            return []
        cleaned = dict(event)
        cleaned["type"] = "tick"
        cleaned["done"] = done
        total = event.get("total")
        if isinstance(total, bool) or not isinstance(total, int):
            cleaned["total"] = 0
        elapsed = event.get("elapsed")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            cleaned.pop("elapsed", None)
        else:
            cleaned["elapsed"] = float(elapsed)
        return [cleaned]

    def _end(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        key = _key(event)
        summary = event.get("summary") if isinstance(event.get("summary"), str) else ""
        if key not in self._open:
            if summary:
                return [{"type": "log", "line": summary}]
            return []
        self._open.discard(key)
        return [event]


def _key(event: dict[str, Any]) -> tuple[Any, str, int | None]:
    phase = event.get("phase") if isinstance(event.get("phase"), str) else ""
    request = event.get("request")
    request_id = request if isinstance(request, int) and not isinstance(request, bool) else None
    source = event.get("source")
    if isinstance(source, bool) or not isinstance(source, (int, str)):
        source = ""
    return (source, phase, request_id)
