"""Open the packed record for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen38_27b.chat import handle_responses
from qwen38_27b.forward import Qwen38Runtime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "half",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
    mtp: bool = False,
) -> Qwen38Runtime:
    """Load the resident fp4 record and the key/value cache (``half`` or ``fp8_e4m3``).

    ``max_memory`` is the serve ledger's ceiling. ``mtp`` false leaves the MTP block in the record file.
    """
    del resident_percent
    from favlo.memory import memory

    memory.set_max_memory(max_memory)
    return Qwen38Runtime(
        path,
        kv_cache=kv_cache,
        console=console,
        owner=owner,
        mtp=mtp,
    )


__all__ = ["handle_responses", "open_runtime"]
