"""Open the packed record for ``favlo serve``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gemma4_26b_a4b.metal41.chat import handle_responses
from gemma4_26b_a4b.metal41.forward import Gemma4Runtime


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "fp8_e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    owner: str | None = None,
    max_memory: str | int | None = None,
    mtp: bool = False,
) -> Gemma4Runtime:
    """Load the resident matrices. Routed experts stay in the layer files."""
    del resident_percent, mtp
    from favlo.memory import memory
    from gemma4_26b_a4b.metal41.atlas import kv_mode

    memory.set_max_memory(max_memory)
    return Gemma4Runtime(
        path,
        kv_cache=kv_mode(kv_cache),
        console=console,
        owner=owner or "gemma4_26b_a4b",
    )


__all__ = ["handle_responses", "open_runtime"]
