"""Vision tower for one still image.

The patch embed is the flattened Conv3d. Position rows are a bilinear sample of
the 48 by 48 table, then reordered into 2 by 2 merge blocks. Each block is
LayerNorm, full attention, and a tanh GELU MLP. The merger LayerNorms each
patch, concatenates the block, and applies erf GELU. The prompt expands each still to its merged grid before the prefix check.
Prefill overwrites only the image-pad rows past that hit. A span already in
the loaded prefix logs ``vision cached`` and uploads no rows. A span past the
hit reads ``~/.favlo/vision-cache/qwen35_4b/`` and logs ``vision cache``,
or runs the tower and logs ``vision patches`` and ``vision merged``.
The process does not keep those rows. The turn releases the device copy
when it finishes.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from favlo.memory import memory
from qwen35_4b.metal41.layout import (
    HIDDEN,
    IMAGE_TOKEN,
    MERGER_DIM,
    PATCH,
    PATCH_DIM,
    SPATIAL_MERGE,
    TEMPORAL_PATCH,
    VISION_CHANNELS,
    VISION_DEPTH,
    VISION_HEADS,
    VISION_HIDDEN,
    VISION_INTERMEDIATE,
    VISION_OUT,
    VISION_POSITIONS,
)
from qwen35_4b.metal41.session import f32, u32

MERGE = SPATIAL_MERGE
HEAD = VISION_HIDDEN // VISION_HEADS
GRID = 48
QKV = VISION_HIDDEN * 3
WIDE = VISION_INTERMEDIATE
FACTOR = PATCH * MERGE
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
MAX_RATIO = 200
MAX_PATCHES = 8192
MAX_IMAGE_BYTES = 64 * 1024 * 1024
_FETCH_SECONDS = 30.0
EPS = np.float32(1e-6)
THETA = 10000.0
_EMPTY = 0xFFFFFFFF
_PIPELINE = "qwen35_4b"


def smart_size(
    height: int,
    width: int,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Nearest factor-aligned size inside the processor pixel bounds."""
    if height < 1 or width < 1:
        message = f"image size {width}x{height} is empty"
        raise RuntimeError(message)
    if min_pixels < 1 or max_pixels < min_pixels:
        message = f"image pixels {min_pixels}..{max_pixels} are empty"
        raise RuntimeError(message)
    if max(height, width) / min(height, width) > MAX_RATIO:
        message = f"image aspect {width}x{height} exceeds {MAX_RATIO}"
        raise RuntimeError(message)
    h_bar = max(FACTOR, round(height / FACTOR) * FACTOR)
    w_bar = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h_bar * w_bar > max_pixels:
        beta = ((height * width) / max_pixels) ** 0.5
        h_bar = max(FACTOR, int(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, int(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = int(np.ceil(height * beta / FACTOR) * FACTOR)
        w_bar = int(np.ceil(width * beta / FACTOR) * FACTOR)
    return int(h_bar), int(w_bar)


def merged_grid(
    rgb: np.ndarray,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Merged height and width for one still. This does not run the tower."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        message = f"image is {getattr(rgb, 'shape', None)}"
        raise RuntimeError(message)
    height, width = smart_size(
        int(rgb.shape[0]),
        int(rgb.shape[1]),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    merged_h = height // FACTOR
    merged_w = width // FACTOR
    if merged_h < 1 or merged_w < 1:
        message = f"image grid {height}x{width} has no merged tokens"
        raise RuntimeError(message)
    return merged_h, merged_w


def pack_image(
    rgb: np.ndarray,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[np.ndarray, int, int]:
    """Normalized patches ``[count, 1536]`` and the patch grid."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        message = f"image is {rgb.shape} {rgb.dtype}"
        raise RuntimeError(message)
    height, width = smart_size(
        int(rgb.shape[0]),
        int(rgb.shape[1]),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = _resize(rgb, height, width)
    values = rgb.astype(np.float32) / np.float32(255.0)
    values = (values - np.float32(0.5)) / np.float32(0.5)
    grid_h = height // PATCH
    grid_w = width // PATCH
    frames = np.stack([values] * TEMPORAL_PATCH, axis=0)
    blocks_h = grid_h // MERGE
    blocks_w = grid_w // MERGE
    frames = frames.reshape(
        1,
        TEMPORAL_PATCH,
        blocks_h,
        MERGE,
        PATCH,
        blocks_w,
        MERGE,
        PATCH,
        VISION_CHANNELS,
    )
    frames = frames.transpose(0, 2, 5, 3, 6, 8, 1, 4, 7)
    packed = np.ascontiguousarray(frames.reshape(-1, PATCH_DIM), dtype=np.float32)
    if packed.shape[0] > MAX_PATCHES:
        message = f"vision patches {packed.shape[0]} exceed {MAX_PATCHES}"
        raise RuntimeError(message)
    return packed, grid_h, grid_w


def interpolate_positions(table: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Bilinear sample of ``[48*48, hidden]``, in merge-block order."""
    side = GRID
    if table.shape != (side * side, VISION_HIDDEN):
        message = f"position table is {table.shape}"
        raise RuntimeError(message)
    if VISION_POSITIONS != side * side:
        message = f"position count {VISION_POSITIONS} is not {side * side}"
        raise RuntimeError(message)
    h_idx = np.linspace(0, side - 1, grid_h, dtype=np.float64)
    w_idx = np.linspace(0, side - 1, grid_w, dtype=np.float64)
    h_floor = np.floor(h_idx).astype(np.int64)
    w_floor = np.floor(w_idx).astype(np.int64)
    h_ceil = np.minimum(h_floor + 1, side - 1)
    w_ceil = np.minimum(w_floor + 1, side - 1)
    dh = (h_idx - h_floor)[:, None]
    dw = (w_idx - w_floor)[None, :]
    hf = h_floor[:, None]
    wf = w_floor[None, :]
    hc = h_ceil[:, None]
    wc = w_ceil[None, :]
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01
    acc = (
        table[hf * side + wf] * w00[..., None]
        + table[hf * side + wc] * w01[..., None]
        + table[hc * side + wf] * w10[..., None]
        + table[hc * side + wc] * w11[..., None]
    )
    acc = acc.reshape(grid_h // MERGE, MERGE, grid_w // MERGE, MERGE, VISION_HIDDEN)
    acc = acc.transpose(0, 2, 1, 3, 4).reshape(-1, VISION_HIDDEN)
    return np.ascontiguousarray(acc, dtype=np.float32)


def vision_rotary(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine for the duplicated 32-wide vision frequencies."""
    h_ids, w_ids = _merge_ids(grid_h, grid_w)
    inv = 1.0 / (THETA ** (np.arange(0, HEAD // 2, 2, dtype=np.float64) / (HEAD // 2)))
    freq = np.concatenate(
        (h_ids.astype(np.float64)[:, None] * inv, w_ids.astype(np.float64)[:, None] * inv),
        axis=1,
    )
    emb = np.concatenate((freq, freq), axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rope_positions(
    length: int, images: list[tuple[int, int, int]]
) -> tuple[np.ndarray, int]:
    """Three mRoPE axes, and the delta decode adds to the sequence index."""
    if length < 0:
        message = f"prompt length {length} is negative"
        raise ValueError(message)
    positions = np.empty((3, length), dtype=np.uint32)
    cursor = 0
    index = 0
    previous = -1
    for start, height, width in images:
        if height < 1 or width < 1:
            message = f"image grid {height}x{width} is empty"
            raise ValueError(message)
        count = height * width
        if start < index or start + count > length:
            message = f"image span {start}+{count} does not fit {length} tokens"
            raise ValueError(message)
        if start < previous:
            message = "image spans are out of order"
            raise ValueError(message)
        previous = start
        text = start - index
        if text:
            span = np.arange(cursor, cursor + text, dtype=np.uint32)
            positions[:, index:start] = span
            cursor += text
        height_ids = np.repeat(np.arange(height, dtype=np.uint32), width) + np.uint32(cursor)
        width_ids = np.tile(np.arange(width, dtype=np.uint32), height) + np.uint32(cursor)
        positions[0, start : start + count] = np.uint32(cursor)
        positions[1, start : start + count] = height_ids
        positions[2, start : start + count] = width_ids
        cursor += max(height, width)
        index = start + count
    rest = length - index
    if rest:
        span = np.arange(cursor, cursor + rest, dtype=np.uint32)
        positions[:, index:] = span
        cursor += rest
    if length == 0:
        return positions, 0
    delta = int(positions.max()) + 1 - length
    return positions, delta


def expand_image_tokens(
    ids: list[int], lengths: list[int], image_id: int
) -> tuple[list[int], list[int]]:
    """Replace each image pad with ``lengths[i]`` copies. Spans are absolute starts."""
    out: list[int] = []
    spans: list[int] = []
    index = 0
    for token in ids:
        if token != image_id:
            out.append(int(token))
            continue
        if index >= len(lengths):
            message = "the prompt has more image pads than images"
            raise RuntimeError(message)
        spans.append(len(out))
        out.extend([image_id] * lengths[index])
        index += 1
    if index != len(lengths):
        message = "the prompt has fewer image pads than images"
        raise RuntimeError(message)
    return out, spans


def prepare_prompt(runtime: Any, messages: list[dict[str, Any]], ids: list[int]) -> list[int]:
    """Expand each image to its merged grid and install mRoPE. The tower waits for the prefix hit."""
    _reject_video(messages)
    blobs = _image_blobs(messages)
    if not blobs:
        runtime._vision_rows = None
        runtime._vision_index = None
        runtime._rope_device = None
        runtime._rope_length = 0
        runtime._rope_delta = 0
        runtime._vision_plan = None
        return ids
    lengths: list[int] = []
    grids: list[tuple[int, int]] = []
    for blob, low, high in blobs:
        merged_h, merged_w = merged_grid(
            _decode_bitmap(blob), min_pixels=low, max_pixels=high
        )
        lengths.append(merged_h * merged_w)
        grids.append((merged_h, merged_w))
    expanded, spans = expand_image_tokens(ids, lengths, IMAGE_TOKEN)
    if spans and lengths and spans[-1] + lengths[-1] == len(expanded):
        message = "the last prompt token is an image pad"
        raise RuntimeError(message)
    positions, delta = rope_positions(
        len(expanded),
        [(start, height, width) for start, (height, width) in zip(spans, grids, strict=True)],
    )
    _install_rope(runtime, positions)
    runtime._rope_delta = delta
    runtime._vision_plan = [
        (blob, low, high, start, height, width)
        for (blob, low, high), start, (height, width) in zip(blobs, spans, grids, strict=True)
    ]
    runtime._vision_rows = None
    runtime._vision_index = None
    return expanded


def bind_vision(runtime: Any, hit: int) -> None:
    """Embed images that sit past ``hit``. A covered image is already in the stored KV."""
    plan = getattr(runtime, "_vision_plan", None) or []
    if not plan:
        return
    covered: list[int] = []
    pending: list[tuple[Any, ...]] = []
    for item in plan:
        _blob, _low, _high, start, height, width = item
        count = height * width
        if start + count <= hit:
            covered.append(count)
        else:
            pending.append(item)
    for count in covered:
        _note(runtime, f"vision cached {count}")
    if not pending:
        rows = getattr(runtime, "_vision_rows", None)
        index = getattr(runtime, "_vision_index", None)
        runtime._vision_rows = None
        runtime._vision_index = None
        _drop_device([rows, index])
        return
    rows: list[np.ndarray] = []
    spans: list[int] = []
    for blob, low, high, start, height, width in pending:
        embedded, grid_h, grid_w = _vision_row(
            runtime, blob, min_pixels=low, max_pixels=high
        )
        merged_h = grid_h // MERGE
        merged_w = grid_w // MERGE
        if (merged_h, merged_w) != (height, width) or embedded.shape[0] != height * width:
            message = f"vision merged {embedded.shape[0]} is not {height}x{width}"
            raise RuntimeError(message)
        rows.append(embedded)
        spans.append(start)
    length = int(getattr(runtime, "_rope_length", 0))
    if length < 1:
        message = "vision rows have no position table"
        raise RuntimeError(message)
    _install_rows(runtime, length, spans, rows)


def embed_image(
    runtime: Any,
    blob: bytes,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[np.ndarray, int, int]:
    """Merged rows ``[tokens, 2560]`` and the patch grid for one still image."""
    pixels, grid_h, grid_w = pack_image(
        _decode_bitmap(blob), min_pixels=min_pixels, max_pixels=max_pixels
    )
    patches = int(pixels.shape[0])
    if patches % (MERGE * MERGE) != 0:
        message = f"vision patches {patches} are not a merge block"
        raise RuntimeError(message)
    _note(runtime, f"vision patches {patches}")
    ws = _workspace(runtime, patches)
    memory.store(ws.pixels, 0, np.ascontiguousarray(pixels, dtype=np.float32).tobytes())
    _dequant_positions(runtime, ws)
    count = GRID * GRID * VISION_HIDDEN
    table = np.frombuffer(memory.load(ws.table, 0, count * 4), dtype=np.float32).reshape(
        GRID * GRID, VISION_HIDDEN
    )
    pos = interpolate_positions(np.array(table, copy=True), grid_h, grid_w)
    cos, sin = vision_rotary(grid_h, grid_w)
    memory.store(ws.pos, 0, pos.tobytes())
    memory.store(ws.cos, 0, np.ascontiguousarray(cos, dtype=np.float32).tobytes())
    memory.store(ws.sin, 0, np.ascontiguousarray(sin, dtype=np.float32).tobytes())
    session = runtime.session
    session.begin()
    _linear(runtime, ws, ws.pixels, patches, PATCH_DIM, "visual.patch", "visual.patch_bias", ws.hidden, VISION_HIDDEN)
    _residual(runtime, ws.hidden, ws.pos, patches, VISION_HIDDEN)
    for layer in range(VISION_DEPTH):
        _block(runtime, ws, layer, patches)
    _layernorm(runtime, ws.hidden, "visual.merger_norm", "visual.merger_norm_bias", ws.normed, patches)
    merged = patches // (MERGE * MERGE)
    _linear(
        runtime,
        ws,
        ws.normed,
        merged,
        MERGER_DIM,
        "visual.merger_fc1",
        "visual.merger_fc1_bias",
        ws.packed,
        MERGER_DIM,
    )
    _gelu(runtime, ws.packed, merged * MERGER_DIM, erf=True)
    _linear(
        runtime,
        ws,
        ws.packed,
        merged,
        MERGER_DIM,
        "visual.merger_fc2",
        "visual.merger_fc2_bias",
        ws.out,
        VISION_OUT,
    )
    session.end(wait=True)
    embedded = np.frombuffer(memory.load(ws.out, 0, merged * VISION_OUT * 4), dtype=np.float32).copy()
    embedded = embedded.reshape(merged, VISION_OUT)
    if not np.isfinite(embedded).all():
        message = "vision embedding is not finite"
        raise RuntimeError(message)
    if VISION_OUT != HIDDEN:
        message = f"vision output {VISION_OUT} is not hidden {HIDDEN}"
        raise RuntimeError(message)
    _note(runtime, f"vision merged {merged}")
    return embedded, grid_h, grid_w


class _Workspace:
    def __init__(self, session: Any, patches: int) -> None:
        pad = (patches + 63) // 64 * 64
        self.patches = patches
        self.pad = pad
        self.hidden = session.allocate(patches * VISION_HIDDEN * 4)
        self.normed = session.allocate(patches * VISION_HIDDEN * 4)
        self.wide = session.allocate(pad * WIDE * 4)
        self.packed = session.allocate(pad * WIDE * 4)
        self.half = session.allocate(pad * WIDE * 2)
        self.q = session.allocate(patches * VISION_HIDDEN * 4)
        self.k = session.allocate(patches * VISION_HIDDEN * 4)
        self.v = session.allocate(patches * VISION_HIDDEN * 4)
        self.y = session.allocate(patches * VISION_HIDDEN * 4)
        self.cos = session.allocate(patches * HEAD * 4)
        self.sin = session.allocate(patches * HEAD * 4)
        self.table = session.allocate(GRID * GRID * VISION_HIDDEN * 4)
        self.pos = session.allocate(patches * VISION_HIDDEN * 4)
        self.pixels = session.allocate(patches * PATCH_DIM * 4)
        self.out = session.allocate((patches // (MERGE * MERGE)) * VISION_OUT * 4)


def _workspace(runtime: Any, patches: int) -> _Workspace:
    current = getattr(runtime, "_vision_ws", None)
    if current is not None and current.patches >= patches:
        return current
    if patches > MAX_PATCHES:
        message = f"vision patches {patches} exceed {MAX_PATCHES}"
        raise RuntimeError(message)
    ws = _Workspace(runtime.session, patches)
    runtime._vision_ws = ws
    return ws


def _block(runtime: Any, ws: _Workspace, layer: int, patches: int) -> None:
    stem = f"visual.{layer}."
    _layernorm(runtime, ws.hidden, stem + "norm1", stem + "norm1_bias", ws.normed, patches)
    _linear(runtime, ws, ws.normed, patches, VISION_HIDDEN, stem + "qkv", stem + "qkv_bias", ws.packed, QKV)
    runtime.session.launch(
        "split_qkv1024",
        [ws.packed, ws.q, ws.k, ws.v],
        [u32(patches)],
        ((1024 + 31) // 32, patches, 1),
        (32, 1, 1),
    )
    _rope(runtime, ws, ws.q, patches)
    _rope(runtime, ws, ws.k, patches)
    runtime.session.launch(
        "vision_attend64",
        [ws.q, ws.k, ws.v, ws.y],
        [u32(patches), u32(VISION_HEADS)],
        (VISION_HEADS, patches, 1),
        (64, 1, 1),
    )
    _linear(
        runtime,
        ws,
        ws.y,
        patches,
        VISION_HIDDEN,
        stem + "proj",
        stem + "proj_bias",
        ws.normed,
        VISION_HIDDEN,
    )
    _residual(runtime, ws.hidden, ws.normed, patches, VISION_HIDDEN)
    _layernorm(runtime, ws.hidden, stem + "norm2", stem + "norm2_bias", ws.normed, patches)
    _linear(
        runtime,
        ws,
        ws.normed,
        patches,
        VISION_HIDDEN,
        stem + "fc1",
        stem + "fc1_bias",
        ws.packed,
        VISION_INTERMEDIATE,
    )
    _gelu(runtime, ws.packed, patches * VISION_INTERMEDIATE, erf=False)
    _linear(
        runtime,
        ws,
        ws.packed,
        patches,
        VISION_INTERMEDIATE,
        stem + "fc2",
        stem + "fc2_bias",
        ws.normed,
        VISION_HIDDEN,
    )
    _residual(runtime, ws.hidden, ws.normed, patches, VISION_HIDDEN)


def _linear(
    runtime: Any,
    ws: _Workspace,
    source: Any,
    rows: int,
    inner: int,
    weight: str,
    bias: str,
    dest: Any,
    cols: int,
) -> None:
    if rows > ws.patches or cols > WIDE or inner > WIDE or cols % 64 or inner % 32:
        message = f"{weight} {rows}x{cols}x{inner} does not fit the vision tile"
        raise RuntimeError(message)
    pad = (rows + 63) // 64 * 64
    if pad > ws.pad:
        message = f"vision rows {rows} exceed the patch buffer"
        raise RuntimeError(message)
    plane = runtime._planes[weight]
    bias_plane = runtime._planes[bias]
    session = runtime.session
    session.launch(
        "cast_pad_half",
        [source, ws.half],
        [u32(rows), u32(inner), u32(pad)],
        ((inner + 31) // 32, pad, 1),
        (32, 1, 1),
    )
    session.launch(
        "matmul_fp8",
        [ws.half, runtime.weights, runtime.weights, ws.wide],
        [u32(pad), u32(cols), u32(inner)],
        (cols // 64, pad // 64, 1),
        (128, 1, 1),
        offsets=[0, plane.data, plane.scale, 0],
    )
    session.launch(
        "copy_bias_f32",
        [ws.wide, runtime.weights, dest],
        [u32(rows), u32(cols)],
        ((cols + 31) // 32, rows, 1),
        (32, 1, 1),
        offsets=[0, bias_plane.data, 0],
    )


def _layernorm(
    runtime: Any, source: Any, weight: str, bias: str, dest: Any, rows: int
) -> None:
    gain = runtime._planes[weight]
    shift = runtime._planes[bias]
    runtime.session.launch(
        "layernorm_bias_rows",
        [source, runtime.weights, runtime.weights, dest],
        [u32(VISION_HIDDEN), f32(float(EPS)), u32(rows)],
        (rows, 1, 1),
        (256, 1, 1),
        offsets=[0, gain.data, shift.data, 0],
    )


def _rope(runtime: Any, ws: _Workspace, values: Any, patches: int) -> None:
    runtime.session.launch(
        "vision_rope64",
        [values, ws.cos, ws.sin],
        [u32(patches), u32(VISION_HEADS)],
        (patches, 1, 1),
        (256, 1, 1),
    )


def _gelu(runtime: Any, values: Any, count: int, *, erf: bool) -> None:
    runtime.session.launch(
        "gelu_erf_n" if erf else "gelu_tanh_n",
        [values],
        [u32(count)],
        ((count + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def _residual(runtime: Any, left: Any, right: Any, rows: int, width: int) -> None:
    runtime.session.launch(
        "residual_add_rows",
        [left, right],
        [u32(width), u32(rows)],
        ((width + 255) // 256, rows, 1),
        (256, 1, 1),
    )


def _dequant_positions(runtime: Any, ws: _Workspace) -> None:
    plane = runtime._planes["visual.pos"]
    session = runtime.session
    session.begin()
    session.launch(
        "dequant_fp8_rows",
        [runtime.weights, runtime.weights, ws.table],
        [u32(VISION_HIDDEN), u32(VISION_HIDDEN), u32(GRID * GRID)],
        (GRID * GRID, 1, 1),
        (32, 1, 1),
        offsets=[plane.data, plane.scale, 0],
    )
    session.end(wait=True)


def release_vision(runtime: Any) -> None:
    """Drop the device copy of the last image rows and position table."""
    rows = getattr(runtime, "_vision_rows", None)
    index = getattr(runtime, "_vision_index", None)
    rope = getattr(runtime, "_rope_device", None)
    runtime._vision_rows = None
    runtime._vision_index = None
    runtime._rope_device = None
    runtime._rope_length = 0
    runtime._rope_delta = 0
    runtime._vision_plan = None
    _drop_device([rows, index, rope])


def _drop_device(buffers: list[Any]) -> None:
    from favlo.memory import memory

    for buf in buffers:
        if buf is not None:
            memory.release(buf)


def _vision_row(
    runtime: Any,
    blob: bytes,
    *,
    min_pixels: int,
    max_pixels: int,
) -> tuple[np.ndarray, int, int]:
    """The merged rows for these bytes. A repeat reads the stored file."""
    key = (
        str(getattr(runtime, "root", "")),
        hashlib.sha256(blob).digest(),
        int(min_pixels),
        int(max_pixels),
    )
    cached = _load_vision_disk(key)
    if cached is not None:
        embedded, grid_h, grid_w = cached
        _note(runtime, f"vision cache {embedded.shape[0]}")
        return embedded, grid_h, grid_w
    embedded, grid_h, grid_w = embed_image(
        runtime, blob, min_pixels=min_pixels, max_pixels=max_pixels
    )
    _store_vision_disk(key, embedded, grid_h, grid_w)
    return embedded, grid_h, grid_w


def _vision_cache_path(key: tuple[Any, ...]) -> Path:
    from favlo.vision_cache import cache_path

    root, digest, min_pixels, max_pixels = key
    return cache_path(_PIPELINE, str(root), digest, int(min_pixels), int(max_pixels))


def _load_vision_disk(
    key: tuple[Any, ...],
) -> tuple[np.ndarray, int, int] | None:
    from favlo.vision_cache import load_cache

    body = load_cache(_vision_cache_path(key))
    if body is None or "embedded" not in body or "grid_h" not in body or "grid_w" not in body:
        return None
    return body["embedded"], int(body["grid_h"]), int(body["grid_w"])


def _store_vision_disk(
    key: tuple[Any, ...], embedded: np.ndarray, grid_h: int, grid_w: int
) -> None:
    from favlo.vision_cache import store_cache

    store_cache(
        _vision_cache_path(key),
        {
            "embedded": np.ascontiguousarray(embedded, dtype=np.float32),
            "grid_h": np.int32(grid_h),
            "grid_w": np.int32(grid_w),
        },
    )


def _install_rope(runtime: Any, positions: np.ndarray) -> None:
    previous = getattr(runtime, "_rope_device", None)
    packed = np.ascontiguousarray(positions, dtype=np.uint32)
    runtime._rope_device = runtime.session.upload(packed.tobytes())
    runtime._rope_length = int(packed.shape[1])
    if previous is not None:
        _drop_device([previous])


def _install_rows(
    runtime: Any,
    length: int,
    spans: list[int],
    rows: list[np.ndarray],
) -> None:
    """Upload rows for image tokens past the prefix hit. Covered positions stay empty."""
    blob = bytearray()
    index = np.full(length, _EMPTY, dtype=np.uint32)
    slot = 0
    for start, embedded in zip(spans, rows, strict=True):
        flat = np.ascontiguousarray(embedded, dtype=np.float32)
        blob += flat.tobytes()
        index[start : start + int(flat.shape[0])] = np.arange(
            slot, slot + int(flat.shape[0]), dtype=np.uint32
        )
        slot += int(flat.shape[0])
    previous = (
        getattr(runtime, "_vision_rows", None),
        getattr(runtime, "_vision_index", None),
    )
    runtime._vision_rows = runtime.session.upload(bytes(blob))
    runtime._vision_index = runtime.session.upload(index.tobytes())
    _drop_device([buf for buf in previous if buf is not None])


def _reject_video(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and (
                item.get("type") in {"video", "video_url", "input_video"}
                or "video" in item
                or "video_url" in item
            ):
                message_text = "video is not served"
                raise RuntimeError(message_text)


def _image_blobs(messages: list[dict[str, Any]]) -> list[tuple[bytes, int, int]]:
    blobs: list[tuple[bytes, int, int]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and _is_image(item):
                low, high = _pixel_bounds(item)
                blobs.append((_image_bytes(item), low, high))
    return blobs


def _is_image(item: dict[str, Any]) -> bool:
    return "image" in item or "image_url" in item or item.get("type") in {"image", "image_url", "input_image"}


def _pixel_bounds(item: dict[str, Any]) -> tuple[int, int]:
    nested = item.get("image_url")
    if not isinstance(nested, dict):
        nested = {}
    low = _pixel_field(item, nested, "min_pixels")
    high = _pixel_field(item, nested, "max_pixels")
    detail = nested.get("detail", item.get("detail"))
    if detail is not None and detail not in {"auto", "low", "high", "original"}:
        message = f"image detail {detail} is unsupported"
        raise RuntimeError(message)
    if low is None and high is None:
        if detail == "low":
            return MIN_PIXELS, MIN_PIXELS
        return MIN_PIXELS, MAX_PIXELS
    low = MIN_PIXELS if low is None else low
    high = MAX_PIXELS if high is None else high
    if low < MIN_PIXELS or high > MAX_PIXELS or low > high:
        message = f"image pixels {low}..{high} are outside {MIN_PIXELS}..{MAX_PIXELS}"
        raise RuntimeError(message)
    return low, high


def _pixel_field(item: dict[str, Any], nested: dict[str, Any], key: str) -> int | None:
    value = item[key] if key in item else nested.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"image {key} is not an integer"
        raise RuntimeError(message)
    return value


def _image_bytes(item: dict[str, Any]) -> bytes:
    raw = item.get("image_url", item.get("image"))
    if isinstance(raw, dict):
        raw = raw.get("url")
    if not isinstance(raw, str) or not raw:
        message = "image part has no data"
        raise RuntimeError(message)
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        if "base64" not in header:
            message = "image data url is not base64"
            raise RuntimeError(message)
        try:
            return base64.b64decode("".join(payload.split()), validate=True)
        except (ValueError, TypeError) as exc:
            message = "image data url is not base64"
            raise RuntimeError(message) from exc
    if "://" in raw:
        if not raw.startswith(("http://", "https://")):
            message = "image url must be http or https"
            raise RuntimeError(message)
        return _fetch_image(raw)
    try:
        return base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        message = "image url must be http, https, or base64"
        raise RuntimeError(message) from exc


def _fetch_image(url: str) -> bytes:
    _check_url(url)
    opener = urllib.request.build_opener(_PublicRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "favlo"})
    try:
        with opener.open(request, timeout=_FETCH_SECONDS) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(65536)
                if not block:
                    break
                total += len(block)
                if total > MAX_IMAGE_BYTES:
                    message = "image url is too large"
                    raise RuntimeError(message)
                chunks.append(block)
    except urllib.error.URLError as exc:
        message = "image url could not be fetched"
        raise RuntimeError(message) from exc
    if not chunks:
        message = "image url is empty"
        raise RuntimeError(message)
    return b"".join(chunks)


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        message = "image url must be http or https"
        raise RuntimeError(message)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        message = "image url did not resolve"
        raise RuntimeError(message) from exc
    for info in found:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            message = "image url is not a public address"
            raise RuntimeError(message)


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode_bitmap(blob: bytes) -> np.ndarray:
    from AppKit import NSBitmapImageRep
    from Foundation import NSData

    data = NSData.dataWithBytes_length_(blob, len(blob))
    rep = NSBitmapImageRep.imageRepWithData_(data)
    if rep is None:
        message = "image format is unsupported"
        raise RuntimeError(message)
    width = int(rep.pixelsWide())
    height = int(rep.pixelsHigh())
    channels = int(rep.bitsPerPixel()) // 8
    row_bytes = int(rep.bytesPerRow())
    if (
        width < 1
        or height < 1
        or channels < 3
        or int(rep.bitsPerSample()) != 8
        or bool(rep.isPlanar())
        or row_bytes < width * channels
    ):
        message = "image format is unsupported"
        raise RuntimeError(message)
    raw = np.frombuffer(rep.bitmapData(), dtype=np.uint8, count=row_bytes * height)
    view = raw.reshape(height, row_bytes)[:, : width * channels]
    rgb = view.reshape(height, width, channels)
    return np.ascontiguousarray(rgb[:, :, :3])


def _merge_ids(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    h_ids = np.arange(grid_h, dtype=np.int64)[:, None].repeat(grid_w, axis=1)
    w_ids = np.arange(grid_w, dtype=np.int64)[None, :].repeat(grid_h, axis=0)
    h_ids = h_ids.reshape(grid_h // MERGE, MERGE, grid_w // MERGE, MERGE)
    w_ids = w_ids.reshape(grid_h // MERGE, MERGE, grid_w // MERGE, MERGE)
    h_ids = h_ids.transpose(0, 2, 1, 3).reshape(-1)
    w_ids = w_ids.transpose(0, 2, 1, 3).reshape(-1)
    return h_ids, w_ids


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w, _channels = image.shape
    y = np.linspace(0, src_h - 1, height)
    x = np.linspace(0, src_w - 1, width)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0).astype(np.float32)[:, None, None]
    wx = (x - x0).astype(np.float32)[None, :, None]
    top = image[y0[:, None], x0[None, :]].astype(np.float32) * (1.0 - wx)
    top += image[y0[:, None], x1[None, :]].astype(np.float32) * wx
    bot = image[y1[:, None], x0[None, :]].astype(np.float32) * (1.0 - wx)
    bot += image[y1[:, None], x1[None, :]].astype(np.float32) * wx
    out = top * (1.0 - wy) + bot * wy
    return np.rint(np.clip(out, 0, 255)).astype(np.uint8)


def _note(runtime: Any, text: str) -> None:
    console = getattr(runtime, "console", None)
    if console is None:
        print(text, flush=True)
        return
    log = getattr(console, "log", None)
    if callable(log):
        log(text)
        return
    print(text, flush=True)
