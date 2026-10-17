"""Open the packed record for ``madmetal serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen36_35b_a3b.chat import handle_chat
from qwen36_35b_a3b.forward import Qwen36Runtime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
) -> Qwen36Runtime:
    """Load the resident matrices and the expert atlas under ``max_memory``."""
    del resident_percent
    return Qwen36Runtime(
        path,
        kv_cache=kv_cache,
        console=console,
        max_memory=max_memory,
    )


__all__ = ["handle_chat", "open_runtime"]
