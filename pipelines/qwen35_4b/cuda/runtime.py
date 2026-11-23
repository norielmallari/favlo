"""Open the packed record for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen35_4b.cuda.forward import Qwen35Runtime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "int8",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
    mtp: bool = False,
) -> Qwen35Runtime:
    """Load the resident integer record under the serve ledger."""
    del resident_percent
    from favlo.memory import memory

    memory.set_max_memory(max_memory)
    return Qwen35Runtime(
        path,
        kv_cache=kv_cache,
        console=console,
        owner=owner,
        mtp=mtp,
    )


__all__ = ["open_runtime"]
