"""Disk cache for one pipeline's vision rows.

The arrays are opaque. This module does not keep a copy in the process.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def cache_dir(pipeline: str) -> Path:
    """``~/.favlo/vision-cache/<pipeline>/``. The directory is created."""
    from favlo.logs import favlo_home

    path = favlo_home() / "vision-cache" / pipeline
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(
    pipeline: str,
    root: str,
    digest: bytes,
    min_pixels: int,
    max_pixels: int,
) -> Path:
    """Hash of the checkpoint path, the image digest, and the pixel budget."""
    material = hashlib.sha256()
    material.update(str(root).encode())
    material.update(digest if isinstance(digest, bytes) else bytes(digest))
    material.update(struct.pack("<ii", int(min_pixels), int(max_pixels)))
    return cache_dir(pipeline) / f"{material.hexdigest()}.npz"


def load_cache(path: Path) -> dict[str, np.ndarray] | None:
    """Arrays stored at ``path``. A missing or unreadable file is ``None``."""
    if not path.is_file():
        return None
    try:
        payload = np.load(path)
        try:
            loaded: dict[str, np.ndarray] = {}
            for name in payload.files:
                value = np.array(payload[name], copy=True)
                loaded[name] = value if value.ndim == 0 else np.ascontiguousarray(value)
            return loaded
        finally:
            payload.close()
    except (OSError, ValueError, KeyError):
        return None


def store_cache(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write ``arrays`` and replace ``path``. A failed write leaves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        stored = {}
        for name, value in arrays.items():
            array = np.array(value, copy=True)
            stored[name] = array if array.ndim == 0 else np.ascontiguousarray(array)
        np.savez(tmp, **stored)
        tmp.replace(path)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
