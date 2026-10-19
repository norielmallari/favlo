"""Context length and key/value width for this pipeline.

The checkpoint cap is ``max_position_embeddings``. The live cache starts at one
2048-token chunk and grows with the prompt and with decode, up to the longest
multiple of 2048 that still leaves eight expert images. The atlas then keeps
as many images as the remaining bytes allow. The cache is E4M3 unless the
caller asks for fp4 or for the float32 values with no quantization. Keys scale
every 32 channels. Values lie along the token axis and scale every 32 tokens.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

HEAD = 256
KV_HEADS = 2
BLOCK = 32
SCALES = HEAD // BLOCK
E4M3_RECORD = HEAD + SCALES
E2M1_RECORD = HEAD // 2 + SCALES
F32_RECORD = HEAD * 4
MODES = ("e4m3", "e2m1", "f32")
_ALIASES = {"fp8": "e4m3", "fp4": "e2m1", "float32": "f32"}
_CONTEXT_STEP = 2048
CONTEXT_STEP = _CONTEXT_STEP
_KV_BUFFERS = 20


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


def cache_bytes(mode: str, positions: int) -> int:
    """Key and value buffers for the 10 full-attention layers."""
    return _KV_BUFFERS * kv_buffer_bytes(mode, positions)


def fit_positions(
    budget: int,
    fixed: int,
    mode: str,
    limit: int,
    extra: Callable[[int], int],
    *,
    keep: int = 0,
) -> int | None:
    """Largest multiple of 2048 whose cache fits, or None when 2048 does not.

    ``fixed`` is every byte that does not grow with the context. ``extra`` is
    the plane scratch at a candidate length. ``keep`` is left for the expert
    atlas. ``limit`` is the checkpoint cap.
    """
    if limit < _CONTEXT_STEP:
        return None
    top = (limit // _CONTEXT_STEP) * _CONTEXT_STEP
    chosen = 0
    low = 1
    high = top // _CONTEXT_STEP
    while low <= high:
        mid = (low + high) // 2
        positions = mid * _CONTEXT_STEP
        need = fixed + cache_bytes(mode, positions) + extra(positions) + keep
        if need <= budget:
            chosen = positions
            low = mid + 1
        else:
            high = mid - 1
    if chosen < _CONTEXT_STEP:
        return None
    return chosen


def repack_plane(
    src: np.ndarray,
    old_pos: int,
    new_pos: int,
    count: int,
    mode: str,
    kind: str,
) -> np.ndarray:
    """Copy ``count`` live tokens from a cache of ``old_pos`` into ``new_pos``."""
    dst = np.zeros(kv_buffer_bytes(mode, new_pos), dtype=np.uint8)
    if count < 1:
        return dst
    if mode == "f32":
        row = F32_RECORD
        dst.reshape(KV_HEADS, new_pos, row)[:, :count, :] = src.reshape(KV_HEADS, old_pos, row)[
            :, :count, :
        ]
        return dst
    width = 256 if mode == "e4m3" else 128
    old_base = KV_HEADS * old_pos * width
    new_base = KV_HEADS * new_pos * width
    if kind == "key":
        dst[:new_base].reshape(KV_HEADS, new_pos, width)[:, :count, :] = src[:old_base].reshape(
            KV_HEADS, old_pos, width
        )[:, :count, :]
        dst[new_base:].reshape(KV_HEADS, new_pos, 8)[:, :count, :] = src[old_base:].reshape(
            KV_HEADS, old_pos, 8
        )[:, :count, :]
        return dst
    if kind != "value":
        message = f"unknown plane {kind}"
        raise ValueError(message)
    blocks = (count + 31) // 32
    if mode == "e4m3":
        dst[:new_base].reshape(KV_HEADS, HEAD, new_pos)[:, :, :count] = src[:old_base].reshape(
            KV_HEADS, HEAD, old_pos
        )[:, :, :count]
    else:
        half = (count + 1) // 2
        dst[:new_base].reshape(KV_HEADS, HEAD, new_pos // 2)[:, :, :half] = src[:old_base].reshape(
            KV_HEADS, HEAD, old_pos // 2
        )[:, :, :half]
    dst[new_base:].reshape(KV_HEADS, HEAD, new_pos // 32)[:, :, :blocks] = src[old_base:].reshape(
        KV_HEADS, HEAD, old_pos // 32
    )[:, :, :blocks]
    return dst


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
