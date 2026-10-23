"""What stays on the device when the prompt and the next token need room.

The prefill workspace, the decode buffers, the DeltaNet state, and the key/value
cache are claimed first. Vision tensors stay in the record. The MTP head is
dropped as a whole before any language feed-forward layer. Unwired language
layers are read into one stage, from the last index downward.
"""

from __future__ import annotations

from dataclasses import dataclass

from qwen38_27b.attn import scratch_bytes
from qwen38_27b.cache import sequence_cache_bytes
from qwen38_27b.layout import (
    AB,
    ATTN_OUT,
    FULL_LAYERS,
    HIDDEN,
    INTERMEDIATE,
    KV_PROJ,
    LAYERS,
    LINEAR_LAYERS,
    LINEAR_V,
    MIX,
    Q_PROJ,
    VOCAB,
    Z_WIDTH,
)
from qwen38_27b.prefill import workspace_bytes

_ARGMAX_GROUPS = (VOCAB + 255) // 256
_STATE_BYTES = LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4
_HISTORY_BYTES = LINEAR_LAYERS * 3 * MIX * 2
_FFN_PARTS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
_STEP = 2048


@dataclass(frozen=True)
class Group:
    """Tensor spans the fit can keep, drop, or leave in the file."""

    text: tuple[tuple[str, int, int], ...]
    mtp: tuple[tuple[str, int, int], ...]
    layers: tuple[tuple[tuple[str, int, int], ...], ...]


@dataclass(frozen=True)
class Wire:
    """One device image and the context length it leaves room for."""

    positions: int
    mtp: bool
    wired: tuple[int, ...]
    stage: int
    weights: int
    slots: int


def startup_bytes(
    resident: int, positions: int, *, slots: int = FULL_LAYERS, drafts: bool = False
) -> int:
    """Bytes copied or cleared before the first token: weights, cache, scratch, and the workspace."""
    decode = (
        HIDDEN * 4
        + HIDDEN * 4
        + max(MIX, Q_PROJ, INTERMEDIATE) * 4
        + max(Z_WIDTH, INTERMEDIATE) * 4
        + max(Z_WIDTH, ATTN_OUT, INTERMEDIATE) * 4
        + ATTN_OUT * 4
        + ATTN_OUT * 4
        + KV_PROJ * 4
        + KV_PROJ * 4
        + AB * 4
        + AB * 4
        + VOCAB * 4
        + _ARGMAX_GROUPS * 8
    )
    hold = HIDDEN * 4 if drafts else 0
    return (
        resident
        + sequence_cache_bytes(positions, slots)
        + _STATE_BYTES
        + _HISTORY_BYTES
        + hold
        + decode
        + scratch_bytes(positions, slots)
        + workspace_bytes(drafts=drafts)
    )


def tensor_span(entry: dict) -> int:
    """Bytes of one packed tensor, data then pad then scales."""
    if "data_nbytes" in entry:
        return (
            int(entry["data_nbytes"])
            + int(entry.get("scale_pad") or 0)
            + int(entry.get("scale_nbytes") or 0)
        )
    if str(entry.get("dtype")) != "f16":
        message = "tensor span is not fp4, fp8, or f16"
        raise ValueError(message)
    count = 1
    for dim in entry["stored_shape"]:
        count *= int(dim)
    return count * 2


def language_ffn_layer(name: str) -> int | None:
    """The language feed-forward layer of ``name``, or None."""
    if name.startswith("mtp.") or ".visual." in f".{name}." or name.startswith("visual."):
        return None
    if not any(name.endswith(part) for part in _FFN_PARTS):
        return None
    parts = name.split(".")
    if "layers" not in parts or "mlp" not in parts:
        return None
    index = parts.index("layers") + 1
    if index >= len(parts):
        return None
    try:
        return int(parts[index])
    except ValueError:
        return None


def group_tensors(tensors: dict[str, dict], layers: int = LAYERS) -> Group:
    """Split a manifest into text tensors, the MTP head, and language feed-forward layers."""
    text: list[tuple[str, int, int]] = []
    mtp: list[tuple[str, int, int]] = []
    per: list[list[tuple[str, int, int]]] = [[] for _ in range(layers)]
    for name, entry in tensors.items():
        item = (name, int(entry["offset"]), tensor_span(entry))
        if name.startswith("mtp."):
            mtp.append(item)
            continue
        if ".visual." in f".{name}." or name.startswith("visual."):
            continue
        layer = language_ffn_layer(name)
        if layer is None:
            text.append(item)
            continue
        if layer < 0 or layer >= layers:
            message = f"{name} is outside {layers} layers"
            raise RuntimeError(message)
        per[layer].append(item)
    text.sort()
    mtp.sort()
    return Group(
        tuple(text),
        tuple(mtp),
        tuple(tuple(sorted(items)) for items in per),
    )


def layout_copies(
    group: Group, *, mtp: bool, wired: tuple[int, ...]
) -> tuple[list[tuple[str, int, int, int]], int, int]:
    """Packed ``(name, file offset, device offset, span)``, then the stage base and the image size."""
    rows: list[tuple[str, int, int, int]] = []
    cursor = 0

    def add(items: tuple[tuple[str, int, int], ...] | list[tuple[str, int, int]]) -> None:
        nonlocal cursor
        for name, file_offset, span in items:
            rows.append((name, file_offset, cursor, span))
            cursor += span

    add(group.text)
    if mtp:
        add(group.mtp)
    for layer in wired:
        add(_ordered_layer(group, layer))
    stage = _stage_bytes(group, wired)
    return rows, cursor, cursor + stage


def same_weight_image(previous: Wire | None, plan: Wire) -> bool:
    """True when ``plan`` keeps the resident weights and only the cache length changes."""
    if previous is None:
        return False
    return (
        previous.mtp == plan.mtp
        and previous.wired == plan.wired
        and previous.slots == plan.slots
    )


def growth_budget(reclaimable: int, held: int, packed: int) -> int:
    """Device bytes for the next image while ``packed`` cache bytes stay on the host.

    ``held`` is the current image and is released before the allocation.
    ``packed`` is the host copy of the live cache. It stays until the new
    cache is filled, so it is reserved out of the sum.
    """
    return max(0, reclaimable + held - packed)


def choose_wire(
    budget: int,
    needed: int,
    group: Group,
    limit: int,
    *,
    extra_unwire: int = 0,
    mtp_available: bool = True,
) -> Wire:
    """The cheapest image that covers ``needed``, else the longest length with every feed-forward unwired.

    MTP leaves before the first language layer. ``extra_unwire`` steps further down that
    list when a device allocation of the first image fails.
    """
    if extra_unwire < 0:
        message = f"extra unwire {extra_unwire} is negative"
        raise ValueError(message)
    target = _align(needed, limit)
    layers = len(group.layers)
    candidates: list[tuple[bool, tuple[int, ...]]] = []
    if mtp_available and group.mtp:
        candidates.append((True, tuple(range(layers))))
    for keep in range(layers, -1, -1):
        candidates.append((False, tuple(range(keep))))
    fitting: list[Wire] = []
    for mtp, wired in candidates:
        wire = _wire(group, target, mtp, wired)
        if startup_bytes(wire.weights, target, slots=wire.slots, drafts=mtp) <= budget:
            fitting.append(wire)
    if fitting:
        return fitting[min(extra_unwire, len(fitting) - 1)]
    wired: tuple[int, ...] = ()
    weights, stage = _weight_bytes(group, mtp=False, wired=wired)
    positions = _longest(budget, limit, weights, slots=FULL_LAYERS, drafts=False)
    if positions < _STEP:
        message = "the record does not fit a 2048-token cache"
        raise RuntimeError(message)
    return Wire(positions, False, wired, stage, weights, FULL_LAYERS)


def _wire(group: Group, positions: int, mtp: bool, wired: tuple[int, ...]) -> Wire:
    weights, stage = _weight_bytes(group, mtp=mtp, wired=wired)
    slots = FULL_LAYERS + (1 if mtp else 0)
    return Wire(positions, mtp, wired, stage, weights, slots)


def _weight_bytes(group: Group, *, mtp: bool, wired: tuple[int, ...]) -> tuple[int, int]:
    _copies, stage_base, total = layout_copies(group, mtp=mtp, wired=wired)
    return total, total - stage_base


def _stage_bytes(group: Group, wired: tuple[int, ...]) -> int:
    kept = set(wired)
    dropped = [
        _layer_span(group, layer)
        for layer in range(len(group.layers))
        if layer not in kept
    ]
    if not dropped:
        return 0
    return max(dropped)


def _layer_span(group: Group, layer: int) -> int:
    return sum(span for _name, _offset, span in group.layers[layer])


def _ordered_layer(group: Group, layer: int) -> list[tuple[str, int, int]]:
    found: dict[str, tuple[str, int, int]] = {}
    for name, file_offset, span in group.layers[layer]:
        for part in _FFN_PARTS:
            if name.endswith(part):
                found[part] = (name, file_offset, span)
    missing = [part for part in _FFN_PARTS if part not in found]
    if missing:
        message = f"layer {layer} is missing {missing[0]}"
        raise RuntimeError(message)
    return [found[part] for part in _FFN_PARTS]


def _align(needed: int, limit: int) -> int:
    cap = (limit // _STEP) * _STEP
    if cap < _STEP:
        message = "the record does not fit a 2048-token cache"
        raise RuntimeError(message)
    step = max(_STEP, ((max(needed, 1) + _STEP - 1) // _STEP) * _STEP)
    return min(step, cap)


def _longest(budget: int, limit: int, weights: int, *, slots: int, drafts: bool) -> int:
    cap = (limit // _STEP) * _STEP
    lo = 0
    hi = cap // _STEP
    while lo < hi:
        mid = (lo + hi + 1) // 2
        positions = mid * _STEP
        if startup_bytes(weights, positions, slots=slots, drafts=drafts) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo * _STEP
