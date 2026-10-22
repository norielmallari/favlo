"""Run the vision tower on Metal and keep the merged tokens for prefill.

The record stores the tower. A text forward leaves it there. An image request
copies that span once, then each block is a layer norm, an fp4 projection,
and a residual. The merger writes one language vector per 2×2 patch block.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from qwen38_27b.budget import tensor_span
from qwen38_27b.vision import IMAGE_PAD, prepare_image

_HIDDEN = 1152
_HEADS = 16
_HEAD = 72
_INTER = 4304
_INTER_STORED = 4352
_MERGE = 4608
_LANG = 5120
_DEPTH = 27
_POS_SIDE = 48
_POS_ROWS = _POS_SIDE * _POS_SIDE
_EPS = 1e-6


def bind_request(runtime: Any, prompt: list[int], hit: int) -> None:
    """Build RoPE ids and, when the image is not already cached, the embeddings."""
    images = list(getattr(runtime, "_vision_images", None) or [])
    runtime._vision_rows = None
    runtime._vision_map = None
    if not images:
        runtime._rope_ids = None
        runtime._rope_next = None
        return
    from qwen38_27b.vision import language_positions, merged_grid

    prepared = getattr(runtime, "_vision_prepared", None)
    if not prepared:
        prepared = [prepare_image(blob) for blob in images]
    merged = [merged_grid(grid) for _patches, grid in prepared]
    positions, nxt = language_positions(prompt, merged)
    runtime._rope_ids = positions
    runtime._rope_next = nxt
    slots = [index for index, token in enumerate(prompt) if token == IMAGE_PAD]
    expected = sum(height * width for height, width in merged)
    if len(slots) != expected:
        message = f"{len(slots)} image tokens, the grids need {expected}"
        raise RuntimeError(message)
    if not any(index >= hit for index in slots):
        return
    rows = _encode(runtime, prepared)
    runtime._vision_rows = rows
    runtime._vision_map = {slot: row for row, slot in enumerate(slots)}


def scatter(runtime: Any, hidden: Any, pos0: int, count: int) -> None:
    """Overwrite image rows in this prefill chunk with merger outputs."""
    mapping = getattr(runtime, "_vision_map", None)
    rows = getattr(runtime, "_vision_rows", None)
    if not mapping or rows is None:
        return
    local: list[int] = []
    source: list[int] = []
    for offset in range(count):
        slot = mapping.get(pos0 + offset)
        if slot is None:
            continue
        local.append(offset)
        source.append(slot)
    if not local:
        return
    session = runtime.session
    session.encode(
        "scatter_hidden_rows",
        [
            (rows, 0),
            (hidden, 0),
            (session.upload(np.asarray(local, dtype=np.uint32)), 0),
            (session.upload(np.asarray(source, dtype=np.uint32)), 0),
        ],
        [_u32(len(local)), _u32(_LANG)],
        (len(local), 1, 1),
        (256, 1, 1),
    )


def _encode(runtime: Any, prepared: list[tuple[np.ndarray, tuple[int, int]]]) -> Any:
    patches = np.concatenate([item[0] for item in prepared], axis=0)
    count = int(patches.shape[0])
    if count % 4:
        message = "vision patches are not a whole merge block"
        raise RuntimeError(message)
    session = runtime.session
    weights = _record(runtime)
    row_pad = (count + 63) // 64 * 64
    merged = count // 4
    merged_pad = (merged + 63) // 64 * 64
    hidden = session.zeros(row_pad * _HIDDEN * 4)
    work = session.zeros(row_pad * _HIDDEN * 4)
    wide = max(row_pad * _INTER_STORED, merged_pad * _MERGE)
    mixed = session.zeros(wide * 4)
    query = session.zeros(count * _HIDDEN * 4)
    key = session.zeros(count * _HIDDEN * 4)
    value = session.zeros(count * _HIDDEN * 4)
    half = session.zeros(wide * 2)
    coords = _coords(session, prepared)
    session.begin()
    try:
        _project(
            runtime, weights, half, patches, hidden, count, row_pad, 1536, 1536, "model.visual.patch_embed.proj"
        )
        _bias(runtime, weights, hidden, count, _HIDDEN, _HIDDEN, "model.visual.patch_embed.proj.bias")
        _positions(runtime, weights, hidden, prepared)
        for layer in range(_DEPTH):
            _block(
                runtime,
                weights,
                hidden,
                work,
                mixed,
                query,
                key,
                value,
                half,
                coords,
                count,
                row_pad,
                layer,
            )
            session.submit()
        language = session.zeros(merged_pad * _LANG * 4)
        normed = work
        _norm(runtime, weights, hidden, normed, count, _HIDDEN, "model.visual.merger.norm")
        packed = session.zeros(merged_pad * _MERGE * 4)
        session.encode(
            "vision_merge_pack",
            [(normed, 0), (packed, 0)],
            [_u32(merged), _u32(_HIDDEN)],
            (merged, 1, 1),
            (256, 1, 1),
        )
        merged_out = mixed
        _project(
            runtime,
            weights,
            half,
            None,
            merged_out,
            merged,
            merged_pad,
            _MERGE,
            _MERGE,
            "model.visual.merger.linear_fc1",
            source=packed,
            source_stride=_MERGE,
        )
        _bias(runtime, weights, merged_out, merged, _MERGE, _MERGE, "model.visual.merger.linear_fc1.bias")
        _gelu(runtime, merged_out, merged * _MERGE, exact=True)
        _project(
            runtime,
            weights,
            half,
            None,
            language,
            merged,
            merged_pad,
            _MERGE,
            _LANG,
            "model.visual.merger.linear_fc2",
            source=merged_out,
            source_stride=_MERGE,
        )
        _bias(runtime, weights, language, merged, _LANG, _LANG, "model.visual.merger.linear_fc2.bias")
        session.end()
    except Exception:
        session.finish()
        raise
    return language


def _block(
    runtime: Any,
    weights: dict[str, tuple[Any, int, dict]],
    hidden: Any,
    work: Any,
    mixed: Any,
    query: Any,
    key: Any,
    value: Any,
    half: Any,
    coords: Any,
    count: int,
    row_pad: int,
    layer: int,
) -> None:
    prefix = f"model.visual.blocks.{layer}"
    normed = work
    _norm(runtime, weights, hidden, normed, count, _HIDDEN, f"{prefix}.norm1")
    _project(
        runtime,
        weights,
        half,
        None,
        mixed,
        count,
        row_pad,
        _HIDDEN,
        _HIDDEN * 3,
        f"{prefix}.attn.qkv",
        source=normed,
        source_stride=_HIDDEN,
    )
    _bias(runtime, weights, mixed, count, _HIDDEN * 3, _HIDDEN * 3, f"{prefix}.attn.qkv.bias")
    runtime.session.encode(
        "vision_split_qkv",
        [(mixed, 0), (query, 0), (key, 0), (value, 0)],
        [_u32(count), _u32(_HIDDEN), _u32(_HIDDEN * 3)],
        ((count * _HIDDEN * 3 + 255) // 256, 1, 1),
        (256, 1, 1),
    )
    runtime.session.encode(
        "vision_rope",
        [(query, 0), (key, 0), (coords, 0)],
        [_u32(count), _u32(_HEADS), _u32(_HEAD)],
        (count, _HEADS, 1),
        (32, 1, 1),
    )
    runtime.session.encode(
        "vision_attend",
        [(query, 0), (key, 0), (value, 0), (normed, 0)],
        [_u32(count), _u32(_HEADS), _u32(_HEAD)],
        (count, _HEADS, 1),
        (32, 1, 1),
    )
    _project(
        runtime,
        weights,
        half,
        None,
        mixed,
        count,
        row_pad,
        _HIDDEN,
        _HIDDEN,
        f"{prefix}.attn.proj",
        source=normed,
        source_stride=_HIDDEN,
    )
    _bias(runtime, weights, mixed, count, _HIDDEN, _HIDDEN, f"{prefix}.attn.proj.bias")
    _residual(runtime, hidden, mixed, count * _HIDDEN)
    _norm(runtime, weights, hidden, normed, count, _HIDDEN, f"{prefix}.norm2")
    _project(
        runtime,
        weights,
        half,
        None,
        mixed,
        count,
        row_pad,
        _HIDDEN,
        _INTER_STORED,
        f"{prefix}.mlp.linear_fc1",
        source=normed,
        source_stride=_HIDDEN,
    )
    _bias(runtime, weights, mixed, count, _INTER_STORED, _INTER, f"{prefix}.mlp.linear_fc1.bias")
    _gelu(runtime, mixed, count * _INTER_STORED)
    _project(
        runtime,
        weights,
        half,
        None,
        work,
        count,
        row_pad,
        _INTER_STORED,
        _HIDDEN,
        f"{prefix}.mlp.linear_fc2",
        source=mixed,
        source_stride=_INTER_STORED,
    )
    _bias(runtime, weights, work, count, _HIDDEN, _HIDDEN, f"{prefix}.mlp.linear_fc2.bias")
    _residual(runtime, hidden, work, count * _HIDDEN)


def _project(
    runtime: Any,
    weights: dict[str, tuple[Any, int, dict]],
    half: Any,
    host: np.ndarray | None,
    dest: Any,
    count: int,
    row_pad: int,
    logical: int,
    cols_logical: int,
    name: str,
    *,
    source: Any = None,
    source_stride: int = 0,
    dest_stride: int = 0,
) -> None:
    _buf, offset, entry = weights[f"{name}.weight"]
    stored_cols, inner = (int(value) for value in entry["stored_shape"])
    if stored_cols % 64 or inner % 32 or row_pad % 64:
        message = f"{name} tile {row_pad}x{stored_cols}x{inner} is not a 64-row fp4 tile"
        raise RuntimeError(message)
    session = runtime.session
    if host is not None:
        padded = np.zeros((row_pad, inner), dtype=np.float32)
        padded[:count, :logical] = host.reshape(count, logical)
        src = session.upload(padded.astype(np.float16))
        src_stride = inner
    else:
        _to_half(runtime, source, half, count, row_pad, logical, inner, source_stride or logical)
        src = half
        src_stride = inner
    del src_stride
    kernel = "prefill_fp4_n64"
    width = int(session.pipes[kernel].threadExecutionWidth())
    session.encode(
        kernel,
        [
            (src, 0),
            (_buf, offset),
            (_buf, offset + int(entry["data_nbytes"]) + int(entry.get("scale_pad") or 0)),
            (dest, 0),
        ],
        [_u32(row_pad), _u32(stored_cols), _u32(inner)],
        (stored_cols // 64, row_pad // 64, 1),
        (width * 4, 1, 1),
    )
    if dest_stride and dest_stride != stored_cols:
        message = f"{name} destination stride {dest_stride} does not match {stored_cols}"
        raise RuntimeError(message)
    if cols_logical != stored_cols and name.endswith("linear_fc2"):
        return


def _positions(
    runtime: Any,
    weights: dict[str, tuple[Any, int, dict]],
    hidden: Any,
    prepared: list[tuple[np.ndarray, tuple[int, int]]],
) -> None:
    buf, offset, entry = weights["model.visual.pos_embed.weight"]
    _rows, inner = (int(value) for value in entry["stored_shape"])
    table = runtime.session.zeros(_POS_ROWS * _HIDDEN * 4)
    runtime.session.encode(
        "dequant_fp4_rows",
        [
            (buf, offset),
            (buf, offset + int(entry["data_nbytes"]) + int(entry.get("scale_pad") or 0)),
            (table, 0),
        ],
        [_u32(_POS_ROWS), _u32(inner), _u32(_HIDDEN)],
        (_POS_ROWS, 1, 1),
        (32, 1, 1),
    )
    cursor = 0
    for _patches, (grid_h, grid_w) in prepared:
        count = grid_h * grid_w
        runtime.session.encode(
            "vision_pos_add",
            [(table, 0), (hidden, cursor * _HIDDEN * 4)],
            [_u32(_POS_SIDE), _u32(grid_h), _u32(grid_w), _u32(_HIDDEN)],
            (count, 1, 1),
            (256, 1, 1),
        )
        cursor += count


def _coords(session: Any, prepared: list[tuple[np.ndarray, tuple[int, int]]]) -> Any:
    pairs: list[int] = []
    for _patches, (grid_h, grid_w) in prepared:
        blocks_w = grid_w // 2
        for block in range((grid_h // 2) * blocks_w):
            bh = block // blocks_w
            bw = block - bh * blocks_w
            for intra in range(4):
                ih = intra // 2
                iw = intra % 2
                pairs.extend((bh * 2 + ih, bw * 2 + iw))
    return session.upload(np.asarray(pairs, dtype=np.uint32))


def _norm(runtime: Any, weights: dict, src: Any, dest: Any, rows: int, width: int, name: str) -> None:
    buf, offset, entry = weights[f"{name}.weight"]
    bias_buf, bias_off, _bias = weights[f"{name}.bias"]
    del entry
    runtime.session.encode(
        "vision_layernorm",
        [(src, 0), (buf, offset), (bias_buf, bias_off), (dest, 0)],
        [_u32(width), _f32(_EPS), _u32(rows)],
        (rows, 1, 1),
        (256, 1, 1),
    )


def _bias(
    runtime: Any,
    weights: dict,
    rows: Any,
    count: int,
    stored: int,
    logical: int,
    name: str,
) -> None:
    buf, offset, _entry = weights[name]
    runtime.session.encode(
        "vision_bias",
        [(rows, 0), (buf, offset)],
        [_u32(count), _u32(stored), _u32(logical)],
        ((count * stored + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _gelu(runtime: Any, data: Any, count: int, *, exact: bool = False) -> None:
    runtime.session.encode(
        "vision_gelu_erf" if exact else "vision_gelu_tanh",
        [(data, 0), (data, 0)],
        [_u32(count)],
        ((count + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _residual(runtime: Any, left: Any, right: Any, count: int) -> None:
    runtime.session.encode(
        "residual_rows",
        [(left, 0), (right, 0), (left, 0)],
        [_u32(count)],
        ((count + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _to_half(
    runtime: Any,
    src: Any,
    dst: Any,
    rows: int,
    row_pad: int,
    logical: int,
    inner: int,
    src_stride: int,
) -> None:
    runtime.session.encode(
        "vision_to_half",
        [(src, 0), (dst, 0)],
        [_u32(rows), _u32(row_pad), _u32(logical), _u32(inner), _u32(src_stride)],
        ((row_pad * inner + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _record(runtime: Any) -> dict[str, tuple[Any, int, dict]]:
    cached = getattr(runtime, "_vision_record", None)
    if cached is not None:
        return cached
    tensors = runtime.tensors
    names = [name for name in tensors if name.startswith("model.visual.")]
    if not names:
        message = "this record has no vision tower"
        raise RuntimeError(message)
    spans = []
    for name in names:
        entry = tensors[name]
        start = int(entry["offset"])
        spans.append((start, start + tensor_span(entry)))
    origin = min(start for start, _end in spans)
    end = max(stop for _start, stop in spans)
    blob = runtime.weights.load_span(runtime.session, origin, end - origin)
    table = {
        name: (blob, int(tensors[name]["offset"]) - origin, tensors[name])
        for name in names
    }
    runtime._vision_record = table
    runtime._vision_blob = blob
    return table


def _u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _f32(value: float) -> bytes:
    return struct.pack("<f", float(value))
