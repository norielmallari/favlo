"""Open the packed record for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen38_27b.chat import handle_chat
from qwen38_27b.forward import Qwen38Runtime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
) -> Qwen38Runtime:
    """Load the resident fp4 record and the fp8 key/value cache."""
    del resident_percent
    return Qwen38Runtime(
        path,
        kv_cache=kv_cache,
        console=console,
        max_memory=max_memory,
    )


__all__ = ["handle_chat", "open_runtime"]
