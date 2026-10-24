"""Safetensors header. The tensor bytes stay on disk."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def tensor_header(path: Path) -> dict[str, dict[str, object]]:
    """Name to ``dtype``, ``shape``, and ``data_offsets`` from the file header."""
    with path.open("rb") as handle:
        raw_count = handle.read(8)
        if len(raw_count) != 8:
            message = f"{path.name} has no safetensors header"
            raise ValueError(message)
        count = struct.unpack("<Q", raw_count)[0]
        header = json.loads(handle.read(count))
    header.pop("__metadata__", None)
    return header


def summarize(header: dict[str, dict[str, object]]) -> str:
    """One line per dtype, then the tensor count."""
    counts: dict[str, int] = {}
    for row in header.values():
        dtype = str(row.get("dtype") or "")
        counts[dtype] = counts.get(dtype, 0) + 1
    lines = [f"{name} {counts[name]}" for name in sorted(counts)]
    lines.append(f"tensors {len(header)}")
    return "\n".join(lines)
