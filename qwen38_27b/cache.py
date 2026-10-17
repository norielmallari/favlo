"""Context length and the fp8 key/value plane for this pipeline.

Keys are ``[head, token, channel]`` with an E8M0 scale every 32 channels.
Values are ``[head, channel, token]`` with an E8M0 scale every 32 tokens, so
both the score product and the value mix are ``matmul2d``. A head at one
position is 264 bytes either way. The width is fp8 E4M3.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from qwen38_27b.layout import CONTEXT, FULL_LAYERS, HEAD, KV_HEADS

BLOCK = 32
RECORD = HEAD + HEAD // BLOCK


def kv_mode(name: str | None) -> str:
    """The cache width. Empty means fp8 E4M3."""
    text = "e4m3" if name is None or name == "" else str(name)
    if text in {"fp8", "e4m3"}:
        return "e4m3"
    message = f"kv cache {name} is not e4m3"
    raise ValueError(message)


def plane_bytes(positions: int) -> int:
    """One key plane or one value plane, data then scales, for every head."""
    return KV_HEADS * positions * RECORD


def layer_cache_bytes(positions: int) -> int:
    """Key and value planes of one full-attention layer."""
    return 2 * plane_bytes(positions)


def cache_bytes(positions: int) -> int:
    """The 16 language full-attention layers."""
    return FULL_LAYERS * layer_cache_bytes(positions)


def sequence_cache_bytes(positions: int, slots: int = FULL_LAYERS) -> int:
    """Key and value planes for ``slots`` full-attention layers, including MTP when present."""
    return slots * layer_cache_bytes(positions)


def fit_positions(budget: int, resident: int, scratch: int, slots: int = FULL_LAYERS) -> int:
    """Largest context, a multiple of 2048, whose cache fits in ``budget``.

    ``cache_bytes(1)`` is 33,792: one token across the 16 language layers. A drafted
    record adds the MTP plane and passes ``slots`` 17, which is 35,904 bytes per token.
    """
    room = budget - resident - scratch
    per_token = sequence_cache_bytes(1, slots)
    if room < per_token * 2048:
        message = f"maximum memory leaves {room} bytes for the key/value cache"
        raise RuntimeError(message)
    positions = min(CONTEXT, (room // per_token) // 2048 * 2048)
    if positions < 2048:
        message = "the key/value cache does not fit a 2048-token chunk"
        raise RuntimeError(message)
    return positions


def parse_memory(text: str | int | None) -> int | None:
    """``32GiB``, ``512MiB``, or a byte count. Empty means the caller picks the cap."""
    if text is None or text == "":
        return None
    if isinstance(text, int):
        if text < 1:
            message = f"maximum memory {text} is not positive"
            raise ValueError(message)
        return text
    raw = str(text).strip().replace("_", "")
    suffix = ""
    digits = raw
    for ending, name in (
        ("gib", "GiB"),
        ("mib", "MiB"),
        ("gb", "GiB"),
        ("mb", "MiB"),
        ("g", "GiB"),
        ("m", "MiB"),
    ):
        if raw.lower().endswith(ending):
            suffix = name
            digits = raw[: -len(ending)]
            break
    try:
        number = int(digits)
    except ValueError as exc:
        message = f"maximum memory {text} is not a byte count"
        raise ValueError(message) from exc
    if number < 1:
        message = f"maximum memory {text} is not positive"
        raise ValueError(message)
    if suffix == "GiB":
        return number << 30
    if suffix == "MiB":
        return number << 20
    return number


def available_bytes() -> int:
    """Reclaimable pages: free, inactive, and speculative."""
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)

    class Stats(ctypes.Structure):
        _fields_ = [
            ("free_count", ctypes.c_uint),
            ("active_count", ctypes.c_uint),
            ("inactive_count", ctypes.c_uint),
            ("wire_count", ctypes.c_uint),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", ctypes.c_uint),
            ("speculative_count", ctypes.c_uint),
        ]

    count = ctypes.c_uint(ctypes.sizeof(Stats) // 4)
    stats = Stats()
    kern = libc.host_statistics64(
        libc.mach_host_self(),
        ctypes.c_int(4),
        ctypes.byref(stats),
        ctypes.byref(count),
    )
    if kern != 0:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page) * int(pages)
    page = int(os.sysconf("SC_PAGE_SIZE"))
    reclaimable = stats.free_count + stats.inactive_count + stats.speculative_count
    return int(reclaimable) * page


def max_position(root: Path) -> int:
    """The text config's ``max_position_embeddings``, capped at 262144."""
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config") if isinstance(config, dict) else None
    value = None
    if isinstance(text, dict):
        value = text.get("max_position_embeddings")
    if value is None and isinstance(config, dict):
        value = config.get("max_position_embeddings")
    if not isinstance(value, int) or value < 1:
        message = f"{root.name} has no max_position_embeddings"
        raise RuntimeError(message)
    return min(value, CONTEXT)
