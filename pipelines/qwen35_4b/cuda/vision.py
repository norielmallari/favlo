"""Vision tower for one still image.

Matrices are int8 with one fp32 scale per row. LayerNorm weights, LayerNorm
biases, and the other biases stay half. The patch embed is the flattened
Conv3d. Position rows are a bilinear sample of the 48 by 48 table, then
reordered into 2 by 2 merge blocks. Each block is LayerNorm, full attention,
and a tanh GELU MLP. The merger LayerNorms each patch, concatenates the
block, and applies erf GELU.

The prompt expands each still to its merged grid before the prefix check.
Prefill overwrites only the image-pad rows past that hit and rotates with the
three-axis table. A span already in the loaded prefix logs ``vision cached``.
A span past the hit reads ``~/.favlo/vision-cache/qwen35_4b/`` and logs
``vision cache``, or runs the tower and logs ``vision patches`` and
``vision merged``. Video is rejected.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from qwen35_4b.cuda.layout import (
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

MERGE = SPATIAL_MERGE
HEAD = VISION_HIDDEN // VISION_HEADS
GRID = 48
QKV = VISION_HIDDEN * 3
FACTOR = PATCH * MERGE
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
MAX_RATIO = 200
MAX_PATCHES = 8192
MAX_IMAGE_BYTES = 64 * 1024 * 1024
_FETCH_SECONDS = 30.0
EPS = 1e-6
THETA = 10000.0
_EMPTY = np.uint32(0xFFFFFFFF)
_PIPELINE = "qwen35_4b"
_GDIPLUS = None


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


def rope_positions(length: int, images: list[tuple[int, int, int]]) -> tuple[np.ndarray, int]:
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
    runtime._image_lengths = []
    runtime._image_token = None
    blobs = _image_blobs(messages)
    if not blobs:
        release_vision(runtime)
        runtime._vision_plan = None
        return ids
    if "visual.patch" not in getattr(runtime, "_tensors", {}):
        message = "qwen35_4b cuda record has no vision tower"
        raise RuntimeError(message)
    lengths: list[int] = []
    grids: list[tuple[int, int]] = []
    for blob, low, high in blobs:
        merged_h, merged_w = merged_grid(_decode_bitmap(blob), min_pixels=low, max_pixels=high)
        lengths.append(merged_h * merged_w)
        grids.append((merged_h, merged_w))
    runtime._image_token = IMAGE_TOKEN
    runtime._image_lengths = lengths
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
    _free_rows(runtime)
    return expanded


def bind_vision(runtime: Any, hit: int) -> None:
    """Embed images that sit past ``hit``. A covered image is already in the stored KV."""
    plan = getattr(runtime, "_vision_plan", None) or []
    if not plan:
        return
    rows: list[np.ndarray] = []
    spans: list[int] = []
    for blob, low, high, start, height, width in plan:
        count = height * width
        if start + count <= hit:
            _note(runtime, f"vision cached {count}")
            continue
        embedded, grid_h, grid_w = _vision_row(runtime, blob, min_pixels=low, max_pixels=high)
        if embedded.shape[0] != count or grid_h // MERGE != height or grid_w // MERGE != width:
            message = f"vision merged {embedded.shape[0]} is not {height}x{width}"
            raise RuntimeError(message)
        rows.append(embedded)
        spans.append(start)
    length = int(getattr(runtime, "_rope_length", 0))
    if rows and length < 1:
        message = "vision rows have no position table"
        raise RuntimeError(message)
    if rows:
        _install_rows(runtime, length, spans, rows)


def release_vision(runtime: Any) -> None:
    """Drop the device copy of the last image rows and position table."""
    _free_rows(runtime)
    rope = getattr(runtime, "_rope_device", None)
    runtime._rope_device = None
    runtime._rope_length = 0
    runtime._rope_delta = 0
    runtime._vision_plan = None
    _free(runtime, rope)


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
    runtime.engine.h2d(ws.pixels, np.ascontiguousarray(pixels, dtype=np.float32))
    pos = interpolate_positions(_position_table(runtime, ws), grid_h, grid_w)
    cos, sin = vision_rotary(grid_h, grid_w)
    runtime.engine.h2d(ws.pos, pos)
    runtime.engine.h2d(ws.cos, np.ascontiguousarray(cos, dtype=np.float32))
    runtime.engine.h2d(ws.sin, np.ascontiguousarray(sin, dtype=np.float32))
    _linear(runtime, ws.pixels, patches, PATCH_DIM, "visual.patch", "visual.patch_bias", ws.hidden, VISION_HIDDEN)
    runtime.engine.add(ws.hidden, ws.pos, VISION_HIDDEN, patches)
    for layer in range(VISION_DEPTH):
        _block(runtime, ws, layer, patches)
        _check(runtime)
    _layernorm(runtime, ws.hidden, "visual.merger_norm", "visual.merger_norm_bias", ws.normed, patches)
    merged = patches // (MERGE * MERGE)
    _linear(
        runtime,
        ws.normed,
        merged,
        MERGER_DIM,
        "visual.merger_fc1",
        "visual.merger_fc1_bias",
        ws.packed,
        MERGER_DIM,
    )
    runtime.engine.vision_gelu(ws.packed, merged * MERGER_DIM, erf=True)
    _linear(
        runtime,
        ws.packed,
        merged,
        MERGER_DIM,
        "visual.merger_fc2",
        "visual.merger_fc2_bias",
        ws.out,
        VISION_OUT,
    )
    runtime.engine.sync()
    host = np.empty((merged, VISION_OUT), dtype=np.float32)
    runtime.engine.d2h(host, ws.out)
    if not np.isfinite(host).all():
        message = "vision embedding is not finite"
        raise RuntimeError(message)
    if VISION_OUT != HIDDEN:
        message = f"vision output {VISION_OUT} is not hidden {HIDDEN}"
        raise RuntimeError(message)
    _note(runtime, f"vision merged {merged}")
    return host, grid_h, grid_w


class _Workspace:
    def __init__(self, runtime: Any, patches: int) -> None:
        self.patches = patches
        self.hidden = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.normed = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.packed = _alloc(runtime, patches * QKV * 4)
        self.q = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.k = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.v = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.y = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.cos = _alloc(runtime, patches * HEAD * 4)
        self.sin = _alloc(runtime, patches * HEAD * 4)
        self.table = _alloc(runtime, GRID * GRID * VISION_HIDDEN * 4)
        self.pos = _alloc(runtime, patches * VISION_HIDDEN * 4)
        self.pixels = _alloc(runtime, patches * PATCH_DIM * 4)
        self.out = _alloc(runtime, (patches // (MERGE * MERGE)) * VISION_OUT * 4)
        self._pointers = [
            self.hidden,
            self.normed,
            self.packed,
            self.q,
            self.k,
            self.v,
            self.y,
            self.cos,
            self.sin,
            self.table,
            self.pos,
            self.pixels,
            self.out,
        ]


def _workspace(runtime: Any, patches: int) -> _Workspace:
    current = getattr(runtime, "_vision_ws", None)
    if current is not None and current.patches >= patches:
        return current
    if patches > MAX_PATCHES:
        message = f"vision patches {patches} exceed {MAX_PATCHES}"
        raise RuntimeError(message)
    if current is not None:
        for pointer in current._pointers:
            _free(runtime, pointer)
    ws = _Workspace(runtime, patches)
    runtime._vision_ws = ws
    return ws


def _block(runtime: Any, ws: _Workspace, layer: int, patches: int) -> None:
    stem = f"visual.{layer}."
    _layernorm(runtime, ws.hidden, stem + "norm1", stem + "norm1_bias", ws.normed, patches)
    _linear(runtime, ws.normed, patches, VISION_HIDDEN, stem + "qkv", stem + "qkv_bias", ws.packed, QKV)
    runtime.engine.vision_split_qkv(ws.packed, ws.q, ws.k, ws.v, patches)
    runtime.engine.vision_rope(ws.q, ws.cos, ws.sin, patches, VISION_HEADS)
    runtime.engine.vision_rope(ws.k, ws.cos, ws.sin, patches, VISION_HEADS)
    runtime.engine.vision_attend(ws.q, ws.k, ws.v, ws.y, patches, VISION_HEADS)
    _linear(runtime, ws.y, patches, VISION_HIDDEN, stem + "proj", stem + "proj_bias", ws.normed, VISION_HIDDEN)
    runtime.engine.add(ws.hidden, ws.normed, VISION_HIDDEN, patches)
    _layernorm(runtime, ws.hidden, stem + "norm2", stem + "norm2_bias", ws.normed, patches)
    _linear(
        runtime,
        ws.normed,
        patches,
        VISION_HIDDEN,
        stem + "fc1",
        stem + "fc1_bias",
        ws.packed,
        VISION_INTERMEDIATE,
    )
    runtime.engine.vision_gelu(ws.packed, patches * VISION_INTERMEDIATE, erf=False)
    _linear(
        runtime,
        ws.packed,
        patches,
        VISION_INTERMEDIATE,
        stem + "fc2",
        stem + "fc2_bias",
        ws.normed,
        VISION_HIDDEN,
    )
    runtime.engine.add(ws.hidden, ws.normed, VISION_HIDDEN, patches)


def _linear(
    runtime: Any,
    source: int,
    rows: int,
    inner: int,
    weight: str,
    bias: str,
    dest: int,
    cols: int,
) -> None:
    runtime.engine.gemm(
        runtime._weight(weight),
        runtime._scale(weight),
        source,
        dest,
        cols,
        inner,
        rows,
    )
    runtime.engine.vision_bias(dest, runtime._weight(bias), cols, rows)


def _layernorm(runtime: Any, source: int, weight: str, bias: str, dest: int, rows: int) -> None:
    runtime.engine.vision_layernorm(
        source,
        runtime._weight(weight),
        runtime._weight(bias),
        dest,
        VISION_HIDDEN,
        EPS,
        rows,
    )


def _position_table(runtime: Any, ws: _Workspace) -> np.ndarray:
    runtime.engine.vision_dequant(
        runtime._weight("visual.pos"),
        runtime._scale("visual.pos"),
        ws.table,
        VISION_HIDDEN,
        VISION_POSITIONS,
    )
    runtime.engine.sync()
    table = np.empty((VISION_POSITIONS, VISION_HIDDEN), dtype=np.float32)
    runtime.engine.d2h(table, ws.table)
    return table


def _install_rope(runtime: Any, positions: np.ndarray) -> None:
    previous = getattr(runtime, "_rope_device", None)
    packed = np.ascontiguousarray(positions, dtype=np.uint32)
    pointer = _alloc(runtime, packed.nbytes)
    runtime.engine.h2d(pointer, packed)
    runtime._rope_device = pointer
    runtime._rope_length = int(packed.shape[1])
    _free(runtime, previous)


def _install_rows(runtime: Any, length: int, spans: list[int], rows: list[np.ndarray]) -> None:
    blob = bytearray()
    index = np.full(length, _EMPTY, dtype=np.uint32)
    slot = 0
    for start, embedded in zip(spans, rows, strict=True):
        flat = np.ascontiguousarray(embedded, dtype=np.float32)
        blob += flat.tobytes()
        index[start : start + int(flat.shape[0])] = np.arange(slot, slot + int(flat.shape[0]), dtype=np.uint32)
        slot += int(flat.shape[0])
    _free_rows(runtime)
    raw = np.frombuffer(bytes(blob), dtype=np.float32)
    runtime._vision_rows = _alloc(runtime, raw.nbytes)
    runtime.engine.h2d(runtime._vision_rows, raw)
    runtime._vision_index = _alloc(runtime, index.nbytes)
    runtime.engine.h2d(runtime._vision_index, index)


def _free_rows(runtime: Any) -> None:
    _free(runtime, getattr(runtime, "_vision_rows", None))
    _free(runtime, getattr(runtime, "_vision_index", None))
    runtime._vision_rows = None
    runtime._vision_index = None


def _alloc(runtime: Any, nbytes: int) -> int:
    pointer = runtime.engine.alloc(nbytes)
    owned = getattr(runtime, "_owned", None)
    if isinstance(owned, list):
        owned.append(pointer)
    return pointer


def _free(runtime: Any, pointer: int | None) -> None:
    if not pointer:
        return
    engine = getattr(runtime, "engine", None)
    if engine is None:
        return
    engine.free(pointer)
    owned = getattr(runtime, "_owned", None)
    if isinstance(owned, list):
        try:
            owned.remove(pointer)
        except ValueError:
            pass


def _check(runtime: Any) -> None:
    check = getattr(runtime, "_check", None)
    if callable(check):
        check()


def _note(runtime: Any, text: str) -> None:
    console = getattr(runtime, "console", None)
    log = getattr(console, "log", None)
    if callable(log):
        log(text)


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
    if os.name != "nt":
        message = "image decode is a Windows bitmap"
        raise RuntimeError(message)
    if not blob:
        message = "image format is unsupported"
        raise RuntimeError(message)
    _gdiplus()
    gdi = ctypes.WinDLL("gdiplus")
    ole = ctypes.WinDLL("ole32")
    kernel = ctypes.WinDLL("kernel32")
    kernel.GlobalAlloc.restype = ctypes.c_void_p
    kernel.GlobalLock.restype = ctypes.c_void_p
    kernel.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel.GlobalFree.argtypes = [ctypes.c_void_p]
    ole.CreateStreamOnHGlobal.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    ole.CreateStreamOnHGlobal.restype = ctypes.c_long
    gdi.GdipCreateBitmapFromStream.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    gdi.GdipCreateBitmapFromStream.restype = ctypes.c_int
    gdi.GdipGetImageWidth.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    gdi.GdipGetImageHeight.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    gdi.GdipDisposeImage.argtypes = [ctypes.c_void_p]
    gdi.GdipDisposeImage.restype = ctypes.c_int
    moveable = 0x0002
    handle = kernel.GlobalAlloc(moveable, len(blob))
    if not handle:
        message = "image format is unsupported"
        raise RuntimeError(message)
    locked = kernel.GlobalLock(handle)
    ctypes.memmove(locked, blob, len(blob))
    kernel.GlobalUnlock(handle)
    stream = ctypes.c_void_p()
    if ole.CreateStreamOnHGlobal(handle, True, ctypes.byref(stream)) != 0 or not stream:
        kernel.GlobalFree(handle)
        message = "image format is unsupported"
        raise RuntimeError(message)
    bitmap = ctypes.c_void_p()
    try:
        if gdi.GdipCreateBitmapFromStream(stream, ctypes.byref(bitmap)) != 0 or not bitmap:
            message = "image format is unsupported"
            raise RuntimeError(message)
        width = ctypes.c_uint()
        height = ctypes.c_uint()
        gdi.GdipGetImageWidth(bitmap, ctypes.byref(width))
        gdi.GdipGetImageHeight(bitmap, ctypes.byref(height))
        wide = int(width.value)
        high = int(height.value)
        if wide < 1 or high < 1:
            message = "image format is unsupported"
            raise RuntimeError(message)
        stride = ((wide * 3 + 3) // 4) * 4
        raw = (ctypes.c_ubyte * (stride * high))()

        class _Rect(ctypes.Structure):
            _fields_ = (
                ("X", ctypes.c_int),
                ("Y", ctypes.c_int),
                ("Width", ctypes.c_int),
                ("Height", ctypes.c_int),
            )

        class _Data(ctypes.Structure):
            _fields_ = (
                ("Width", ctypes.c_uint),
                ("Height", ctypes.c_uint),
                ("Stride", ctypes.c_int),
                ("PixelFormat", ctypes.c_int),
                ("Scan0", ctypes.c_void_p),
                ("Reserved", ctypes.c_size_t),
            )

        data = _Data()
        rect = _Rect(0, 0, wide, high)
        pixel = 0x00021808
        gdi.GdipBitmapLockBits.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Rect),
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(_Data),
        ]
        gdi.GdipBitmapLockBits.restype = ctypes.c_int
        gdi.GdipBitmapUnlockBits.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Data)]
        gdi.GdipBitmapUnlockBits.restype = ctypes.c_int
        if gdi.GdipBitmapLockBits(bitmap, ctypes.byref(rect), 1, pixel, ctypes.byref(data)) != 0:
            message = "image format is unsupported"
            raise RuntimeError(message)
        try:
            ctypes.memmove(raw, data.Scan0, abs(int(data.Stride)) * high)
            step = int(data.Stride)
        finally:
            gdi.GdipBitmapUnlockBits(bitmap, ctypes.byref(data))
        view = np.frombuffer(raw, dtype=np.uint8).reshape(high, abs(step))
        if step < 0:
            view = view[::-1]
        rgb = np.ascontiguousarray(view[:, : wide * 3].reshape(high, wide, 3)[:, :, ::-1])
        return rgb
    finally:
        if bitmap:
            gdi.GdipDisposeImage(bitmap)
        _release(stream)


def _gdiplus() -> None:
    global _GDIPLUS
    if _GDIPLUS is not None:
        return
    token = ctypes.c_ulong()

    class _Input(ctypes.Structure):
        _fields_ = (
            ("GdiplusVersion", ctypes.c_uint32),
            ("DebugEventCallback", ctypes.c_void_p),
            ("SuppressBackgroundThread", ctypes.c_int),
            ("SuppressExternalCodecs", ctypes.c_int),
        )

    status = ctypes.WinDLL("gdiplus").GdiplusStartup(ctypes.byref(token), ctypes.byref(_Input(1, None, 0, 0)), None)
    if status != 0:
        message = "image format is unsupported"
        raise RuntimeError(message)
    _GDIPLUS = int(token.value)


def _release(unknown: ctypes.c_void_p) -> None:
    table = ctypes.cast(unknown, ctypes.POINTER(ctypes.c_void_p))
    function = ctypes.cast(
        ctypes.cast(table[0], ctypes.POINTER(ctypes.c_void_p))[2],
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p),
    )
    function(unknown)


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


def _vision_row(
    runtime: Any,
    blob: bytes,
    *,
    min_pixels: int,
    max_pixels: int,
) -> tuple[np.ndarray, int, int]:
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
    embedded, grid_h, grid_w = embed_image(runtime, blob, min_pixels=min_pixels, max_pixels=max_pixels)
    _store_vision_disk(key, embedded, grid_h, grid_w)
    return embedded, grid_h, grid_w


def _vision_cache_path(key: tuple[Any, ...]) -> Path:
    from favlo.vision_cache import cache_path

    root, digest, low, high = key
    return cache_path(_PIPELINE, str(root), digest, int(low), int(high))


def _load_vision_disk(key: tuple[Any, ...]) -> tuple[np.ndarray, int, int] | None:
    from favlo.vision_cache import load_cache

    body = load_cache(_vision_cache_path(key))
    if body is None or "embedded" not in body or "grid_h" not in body or "grid_w" not in body:
        return None
    return np.asarray(body["embedded"], dtype=np.float32), int(body["grid_h"]), int(body["grid_w"])


def _store_vision_disk(key: tuple[Any, ...], rows: np.ndarray, grid_h: int, grid_w: int) -> None:
    from favlo.vision_cache import store_cache

    store_cache(
        _vision_cache_path(key),
        {
            "embedded": np.ascontiguousarray(rows, dtype=np.float32),
            "grid_h": np.int32(grid_h),
            "grid_w": np.int32(grid_w),
        },
    )
