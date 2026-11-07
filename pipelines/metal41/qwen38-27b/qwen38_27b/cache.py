"""Context length and the key/value plane for this pipeline.

``half`` stores Metal halves at ``[head, token, channel]`` with no scale plane:
512 bytes per head per token. ``fp8_e4m3`` keeps the quantized layout (Metal
``metal_fp8_e4m3``): keys ``[head, token, channel]`` with an E8M0 scale every
32 channels, values ``[head, channel, token]`` with an E8M0 scale every 32
tokens, 264 bytes per head per token. The served window stops at
``DEFAULT_CONTEXT_LENGTH``.
"""

from __future__ import annotations

import json
from pathlib import Path

from qwen38_27b.layout import CONTEXT, FULL_LAYERS, HEAD, KV_HEADS

DEFAULT_CONTEXT_LENGTH = 131072
BLOCK = 32
E4M3_RECORD = HEAD + HEAD // BLOCK
HALF_RECORD = HEAD * 2
MODES = ("half", "fp8_e4m3")
_ALIASES = {
    "e4m3": "fp8_e4m3",
    "fp8": "fp8_e4m3",
}


def kv_mode(name: str | None) -> str:
    """``half`` or ``fp8_e4m3``. Empty means ``fp8_e4m3``.

    ``e4m3`` and ``fp8`` are ``fp8_e4m3``.
    """
    text = "fp8_e4m3" if name is None or name == "" else str(name)
    text = _ALIASES.get(text, text)
    if text not in MODES:
        message = f"kv cache {name} is not half or fp8_e4m3"
        raise ValueError(message)
    return text


def plain_cache(mode: str | None) -> bool:
    """Raw half. That mode skips the quantized score plane."""
    return kv_mode(mode) == "half"


def kv_record_bytes(mode: str | None) -> int:
    """Bytes of one head at one position."""
    if plain_cache(mode):
        return HALF_RECORD
    return E4M3_RECORD


def plane_bytes(positions: int, mode: str | None = "fp8_e4m3") -> int:
    """One key plane or one value plane for every head."""
    return KV_HEADS * positions * kv_record_bytes(mode)


def layer_cache_bytes(positions: int, mode: str | None = "fp8_e4m3") -> int:
    """Key and value planes of one full-attention layer."""
    return 2 * plane_bytes(positions, mode)


def cache_bytes(positions: int, mode: str | None = "fp8_e4m3") -> int:
    """The 16 language full-attention layers."""
    return FULL_LAYERS * layer_cache_bytes(positions, mode)


def sequence_cache_bytes(
    positions: int, slots: int = FULL_LAYERS, mode: str | None = "fp8_e4m3"
) -> int:
    """Key and value planes for ``slots`` full-attention layers, including MTP when present."""
    return slots * layer_cache_bytes(positions, mode)


def fit_positions(
    budget: int,
    resident: int,
    scratch: int,
    slots: int = FULL_LAYERS,
    mode: str | None = "fp8_e4m3",
) -> int:
    """Largest context, a multiple of 2048, whose cache fits in ``budget``.

    ``cache_bytes(1)`` is 33,792 for ``fp8_e4m3`` across the 16 language layers
    (65,536 for half). A drafted record adds the MTP plane and passes ``slots`` 17,
    which is 35,904 bytes per token on ``fp8_e4m3`` (69,632 on half).
    """
    room = budget - resident - scratch
    per_token = sequence_cache_bytes(1, slots, mode)
    if room < per_token * 2048:
        message = f"maximum memory leaves {room} bytes for the key/value cache"
        raise RuntimeError(message)
    positions = min(CONTEXT, (room // per_token) // 2048 * 2048)
    if positions < 2048:
        message = "the key/value cache does not fit a 2048-token chunk"
        raise RuntimeError(message)
    return positions


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


# Back-compat for callers that still read the quantized-plane constant.
RECORD = E4M3_RECORD
