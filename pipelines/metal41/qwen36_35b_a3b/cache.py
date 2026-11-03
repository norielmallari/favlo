"""Context length and key/value width for this pipeline.

The checkpoint cap is ``max_position_embeddings``. The served window stops at
``DEFAULT_CONTEXT_LENGTH``. The live cache starts at one 2048-token chunk and
grows with the prompt and with decode, up to the longest multiple of 2048 that
still leaves eight expert images and still fits that window. The atlas then keeps
as many images as the remaining bytes allow. The cache is ``fp8_e4m3`` unless
the caller asks for half, ``e2m1``, or the float32 values. half and float32
store raw values, ``[head, token, channel]``, with no scale plane.
``fp8_e4m3`` and ``e2m1`` keys scale every 32 channels. Their values lie
along the token axis and scale every 32 tokens.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

DEFAULT_CONTEXT_LENGTH = 131072
HEAD = 256
KV_HEADS = 2
BLOCK = 32
SCALES = HEAD // BLOCK
E4M3_RECORD = HEAD + SCALES
E2M1_RECORD = HEAD // 2 + SCALES
F16_RECORD = HEAD * 2
F32_RECORD = HEAD * 4
MODES = ("half", "fp8_e4m3", "e2m1", "f32")
_ALIASES = {
    "f16": "half",
    "fp16": "half",
    "e4m3": "fp8_e4m3",
    "fp8": "fp8_e4m3",
    "fp4": "e2m1",
    "float32": "f32",
}
_CONTEXT_STEP = 2048
CONTEXT_STEP = _CONTEXT_STEP
# Ten language full-attention layers, plus the MTP layer.
FULL_ATTENTION = 11
KV_BUFFERS = FULL_ATTENTION * 2


def kv_mode(name: str | None) -> str:
    """``half``, ``fp8_e4m3``, ``e2m1``, or ``f32``. Empty means ``fp8_e4m3``.

    ``f16`` and ``fp16`` are half. ``e4m3`` and ``fp8`` are ``fp8_e4m3``.
    """
    text = "fp8_e4m3" if name is None or name == "" else str(name)
    text = _ALIASES.get(text, text)
    if text not in MODES:
        message = f"kv cache {name} is not half, fp8_e4m3, e2m1, or f32"
        raise ValueError(message)
    return text


def plain_cache(mode: str) -> bool:
    """Raw half or float32. Those modes skip the quantized score plane."""
    return kv_mode(mode) in ("half", "f32")


def kv_io(mode: str, *, rows: bool) -> tuple[str, str]:
    """Store and attend kernel names for a raw cache."""
    if kv_mode(mode) == "half":
        if rows:
            return "store_kv_rows_f16", "attend_prefill2_f16"
        return "store_kv_f16", "attend_gqa_heads_f16"
    if rows:
        return "store_kv_rows", "attend_prefill2"
    return "store_kv", "attend_gqa_heads"


def kv_record_bytes(mode: str) -> int:
    """Bytes of one head at one position, data plane plus its scales."""
    chosen = kv_mode(mode)
    if chosen == "f32":
        return F32_RECORD
    if chosen == "half":
        return F16_RECORD
    if chosen == "e2m1":
        return E2M1_RECORD
    return E4M3_RECORD


def kv_buffer_bytes(mode: str, positions: int) -> int:
    """One layer's K or V buffer: two heads across ``positions``."""
    return KV_HEADS * positions * kv_record_bytes(mode)


def cache_bytes(mode: str, positions: int) -> int:
    """Key and value buffers for the language full-attention layers and the MTP layer."""
    return KV_BUFFERS * kv_buffer_bytes(mode, positions)


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
    mode = kv_mode(mode)
    if mode in ("f32", "half"):
        row = F32_RECORD if mode == "f32" else F16_RECORD
        dst.reshape(KV_HEADS, new_pos, row)[:, :count, :] = src.reshape(KV_HEADS, old_pos, row)[
            :, :count, :
        ]
        return dst
    width = 256 if mode == "fp8_e4m3" else 128
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
    if mode == "fp8_e4m3":
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
