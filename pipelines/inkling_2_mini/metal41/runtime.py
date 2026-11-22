"""Open the checkpoint for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inkling_2_mini.metal41.chat import handle_responses
from inkling_2_mini.metal41.forward import InklingRuntime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "half",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
) -> InklingRuntime:
    """Load the safetensors checkpoint under the serve ledger."""
    del resident_percent
    from favlo.memory import memory

    memory.set_max_memory(max_memory)
    return InklingRuntime(path, kv_cache=kv_cache, console=console, owner=owner)


__all__ = ["handle_responses", "open_runtime"]
