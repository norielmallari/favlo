"""Open the packed record for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from favlo.memory import memory
from qwen36_35b_a3b.cuda.chat import handle_responses
from qwen36_35b_a3b.cuda.forward import Qwen36Runtime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "int8",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
    mtp: bool = False,
) -> Qwen36Runtime:
    """Load the language record. The expert atlas stays in host RAM."""
    del resident_percent
    memory.set_max_memory(max_memory)
    return Qwen36Runtime(path, kv_cache=kv_cache, console=console, owner=owner, mtp=mtp)


__all__ = ["handle_responses", "open_runtime"]
