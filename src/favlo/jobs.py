"""Media job queue.

The daemon owns this queue. One job runs at a time. The MCP task transport
enqueues work here, and the call that created the task has already returned.
A pipeline crash fails that job. The next job starts when it reaches the head.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

_running: str | None = None
_gate = threading.Lock()


def running() -> str | None:
    with _gate:
        return _running


def run(task_id: str, work: Callable[[], Any]) -> Any:
    """Run ``work`` for ``task_id``. A second call waits until this one returns."""
    global _running
    with _gate:
        _running = task_id
        try:
            return work()
        finally:
            if _running == task_id:
                _running = None


async def execute(task_id: str, work: Callable[[], Any]) -> Any:
    """The Docket worker's entry. The tools/call that queued the job has returned."""
    return await asyncio.to_thread(run, task_id, work)
