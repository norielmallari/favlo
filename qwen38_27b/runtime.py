"""Load a packed Qwen3.8-27B record. The language forward is this package's."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
) -> Any:
    """Open the record at ``path``. One pipeline is resident in the process."""
    del kv_cache, console, resident_percent
    root = Path(path)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        message = f"{root.name} has no manifest.json"
        raise FileNotFoundError(message)
    message = "qwen38-27b forward is not available for this record"
    raise RuntimeError(message)


def handle_chat(*_args: Any, **_kwargs: Any) -> None:
    """Chat completion. The language forward is not available yet."""
    message = "qwen38-27b forward is not available for this record"
    raise RuntimeError(message)
