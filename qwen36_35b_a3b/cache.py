"""Context length and key/value width for this pipeline.

The position limit is the checkpoint's ``max_position_embeddings``. The cache
is E4M3 unless the caller asks for fp4 or for the float32 values with no
quantization. Keys scale every 32 channels. Values lie along the token axis and
scale every 32 tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

HEAD = 256
KV_HEADS = 2
BLOCK = 32
SCALES = HEAD // BLOCK
E4M3_RECORD = HEAD + SCALES
E2M1_RECORD = HEAD // 2 + SCALES
F32_RECORD = HEAD * 4
MODES = ("e4m3", "e2m1", "f32")
_ALIASES = {"fp8": "e4m3", "fp4": "e2m1", "float32": "f32"}


def kv_mode(name: str | None) -> str:
    """``e4m3``, ``e2m1``, or ``f32``. Empty means the fp8 default."""
    text = "e4m3" if name is None or name == "" else str(name)
    text = _ALIASES.get(text, text)
    if text not in MODES:
        message = f"kv cache {name} is not e4m3, e2m1, or f32"
        raise ValueError(message)
    return text


def kv_record_bytes(mode: str) -> int:
    """Bytes of one head at one position, data plane plus its scales."""
    chosen = kv_mode(mode)
    if chosen == "f32":
        return F32_RECORD
    if chosen == "e2m1":
        return E2M1_RECORD
    return E4M3_RECORD


def kv_buffer_bytes(mode: str, positions: int) -> int:
    """One layer's K or V buffer: two heads across ``positions``."""
    return KV_HEADS * positions * kv_record_bytes(mode)


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


def max_position(root: Path) -> int:
    """The text config's ``max_position_embeddings``."""
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
    return value
