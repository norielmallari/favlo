"""Open the packed record for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spark_x26_1_7b.chat import handle_chat, handle_responses
from spark_x26_1_7b.forward import SparkRuntime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "f16",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
    mtp: bool = False,
) -> SparkRuntime:
    """Load the resident fp8 record and the key/value cache (``f16`` or ``e4m3``)."""
    del resident_percent, mtp
    return SparkRuntime(
        path,
        kv_cache=kv_cache,
        console=console,
        max_memory=max_memory,
        owner=owner,
    )


__all__ = ["handle_chat", "handle_responses", "open_runtime"]
