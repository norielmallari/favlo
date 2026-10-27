"""One line per tool call: session, run, and tool-call identity."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(
        self,
        *,
        session_id: str,
        run_id: str,
        tool_call_id: str,
        tool: str,
        error: str | None,
        message: str | None = None,
    ) -> None:
        row = {
            "error": error,
            "run_id": run_id,
            "session_id": session_id,
            "tool": tool,
            "tool_call_id": tool_call_id,
        }
        if message:
            row["message"] = message
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
