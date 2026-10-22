"""Vision tower for one still image.

The patch embed is the flattened Conv3d. Position rows are a bilinear sample of
the 48 by 48 table, then reordered into 2 by 2 merge blocks. Each block is
LayerNorm, full attention, and a tanh GELU MLP. The merger LayerNorms each
patch, concatenates the block, and applies erf GELU. The language prefill
overwrites the expanded image-pad rows with that output.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

import numpy as np

PATCH = 16
TEMPORAL = 2
MERGE = 2
FACTOR = PATCH * MERGE
HIDDEN = 1152
HEADS = 16
HEAD = 72
INTER = 4304
OUT = 2048
DEPTH = 27
GRID = 48
QKV = HIDDEN * 3
CHANNELS = 3
PATCH_DIM = CHANNELS * TEMPORAL * PATCH * PATCH
MERGE_DIM = HIDDEN * MERGE * MERGE
WIDE = 4608
MAX_PATCHES = 32 * 32
MAX_PIXELS = 512 * 512
MIN_PIXELS = FACTOR * FACTOR
EPS = np.float32(1e-6)
THETA = 10000.0

_EXPECTED = {
    "depth": DEPTH,
    "hidden_size": HIDDEN,
    "num_heads": HEADS,
    "intermediate_size": INTER,
    "out_hidden_size": OUT,
    "patch_size": PATCH,
    "temporal_patch_size": TEMPORAL,
    "spatial_merge_size": MERGE,
    "in_channels": CHANNELS,
    "num_position_embeddings": GRID * GRID,
}


def smart_size(height: int, width: int) -> tuple[int, int]:
    """Nearest factor-aligned size inside the tower's pixel bounds."""
    if height < 1 or width < 1:
        message = f"image size {width}x{height} is empty"
        raise RuntimeError(message)
    h_bar = max(FACTOR, round(height / FACTOR) * FACTOR)
    w_bar = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h_bar * w_bar > MAX_PIXELS:
        beta = ((height * width) / MAX_PIXELS) ** 0.5
        h_bar = max(FACTOR, int(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, int(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = (MIN_PIXELS / (height * width)) ** 0.5
        h_bar = int(np.ceil(height * beta / FACTOR) * FACTOR)
        w_bar = int(np.ceil(width * beta / FACTOR) * FACTOR)
    return int(h_bar), int(w_bar)


def decode_png(blob: bytes) -> np.ndarray:
    """RGB uint8 ``[height, width, 3]``. 8-bit, non-interlaced."""
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        message = "image is not a png"
        raise RuntimeError(message)
    offset = 8
    width = height = 0
    color = 0
    idat = bytearray()
    while offset < len(blob):
        kind, payload, offset = _chunk(blob, offset)
        if kind == b"IHDR":
            width, height, depth, color, _compression, _png_filter, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if depth != 8 or color not in {2, 6} or interlace != 0:
                message = f"png {width}x{height} depth {depth} color {color} is unsupported"
                raise RuntimeError(message)
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width < 1 or height < 1 or not idat:
        message = "png has no image"
        raise RuntimeError(message)
    raw = zlib.decompress(bytes(idat))
    channels = 3 if color == 2 else 4
    stride = width * channels
    rows = np.empty((height, width, channels), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for y in range(height):
        if cursor + 1 + stride > len(raw):
            message = "png rows are short"
            raise RuntimeError(message)
        filt = raw[cursor]
        row = np.frombuffer(raw[cursor + 1 : cursor + 1 + stride], dtype=np.uint8).copy()
        cursor += 1 + stride
        _unfilter(filt, row, prev, channels)
        rows[y] = row.reshape(width, channels)
        prev = row
    if channels == 4:
        rows = rows[:, :, :3]
    return np.ascontiguousarray(rows)


def pack_image(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Normalized patches ``[count, 1536]`` and the patch grid.

    The flatten order is merge block, then channel, time, and the patch square.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        message = f"image is {rgb.shape} {rgb.dtype}"
        raise RuntimeError(message)
    height, width = smart_size(int(rgb.shape[0]), int(rgb.shape[1]))
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = _resize(rgb, height, width)
    values = rgb.astype(np.float32) / np.float32(255.0)
    values = (values - np.float32(0.5)) / np.float32(0.5)
    grid_h = height // PATCH
    grid_w = width // PATCH
    frames = np.stack([values] * TEMPORAL, axis=0)
    blocks_h = grid_h // MERGE
    blocks_w = grid_w // MERGE
    frames = frames.reshape(
        1,
        TEMPORAL,
        blocks_h,
        MERGE,
        PATCH,
        blocks_w,
        MERGE,
        PATCH,
        CHANNELS,
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
    if table.shape != (side * side, HIDDEN):
        message = f"position table is {table.shape}"
        raise RuntimeError(message)
    h_idx = np.linspace(0, side - 1, grid_h, dtype=np.float64)
    w_idx = np.linspace(0, side - 1, grid_w, dtype=np.float64)
    h_floor = np.floor(h_idx).astype(np.int64)
    w_floor = np.floor(w_idx).astype(np.int64)
    h_ceil = np.minimum(h_floor + 1, side - 1)
    w_ceil = np.minimum(w_floor + 1, side - 1)
    dh = h_idx - h_floor
    dw = w_idx - w_floor
    dh_grid = dh[:, None]
    dw_grid = dw[None, :]
    hf = h_floor[:, None]
    wf = w_floor[None, :]
    hc = h_ceil[:, None]
    wc = w_ceil[None, :]
    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1.0 - dh_grid - w01
    i00 = hf * side + wf
    i01 = hf * side + wc
    i10 = hc * side + wf
    i11 = hc * side + wc
    acc = (
        table[i00] * w00[..., None]
        + table[i01] * w01[..., None]
        + table[i10] * w10[..., None]
        + table[i11] * w11[..., None]
    )
    acc = acc.reshape(grid_h // MERGE, MERGE, grid_w // MERGE, MERGE, HIDDEN)
    acc = acc.transpose(0, 2, 1, 3, 4).reshape(-1, HIDDEN)
    return np.ascontiguousarray(acc, dtype=np.float32)


def vision_rotary(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine for the duplicated 36-wide vision frequencies."""
    h_ids, w_ids = _merge_ids(grid_h, grid_w)
    inv = 1.0 / (THETA ** (np.arange(0, HEAD // 2, 2, dtype=np.float64) / (HEAD // 2)))
    freq = np.concatenate(
        (h_ids.astype(np.float64)[:, None] * inv, w_ids.astype(np.float64)[:, None] * inv),
        axis=1,
    )
    emb = np.concatenate((freq, freq), axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


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
    """Run the tower for each image and remember the rows prefill must overwrite."""
    blobs = _image_blobs(messages)
    if not blobs:
        runtime._vision_at = None
        return ids
    vision, image_id = _vision_config(runtime.root)
    _check_config(vision)
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for blob in blobs:
        embedded = embed_image(runtime, blob)
        rows.append(embedded)
        lengths.append(int(embedded.shape[0]))
    expanded, spans = expand_image_tokens(ids, lengths, image_id)
    table: dict[int, np.ndarray] = {}
    for start, embedded in zip(spans, rows, strict=True):
        for offset in range(embedded.shape[0]):
            table[start + offset] = embedded[offset]
    if expanded and (len(expanded) - 1) in table:
        message = "the last prompt token is an image pad"
        raise RuntimeError(message)
    runtime._vision_at = table
    return expanded


def embed_image(runtime: Any, blob: bytes) -> np.ndarray:
    """Merged rows ``[tokens, 2048]`` for one PNG."""
    vision, _image_id = _vision_config(runtime.root)
    _check_config(vision)
    pixels, grid_h, grid_w = pack_image(decode_png(blob))
    patches = int(pixels.shape[0])
    _note(runtime, f"vision patches {patches}")
    bufs = _Buffers(runtime.session, patches)
    _write(bufs.pixels, pixels)
    _dequant_positions(runtime, bufs)
    count = GRID * GRID * HIDDEN
    table = runtime.session.download_f32(bufs.table, count).reshape(GRID * GRID, HIDDEN)
    pos = interpolate_positions(table, grid_h, grid_w)
    cos, sin = vision_rotary(grid_h, grid_w)
    _write(bufs.pos, pos)
    _write(bufs.cos, cos)
    _write(bufs.sin, sin)
    session = runtime.session
    session.begin()
    _linear(
        runtime,
        bufs,
        bufs.pixels,
        patches,
        PATCH_DIM,
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        bufs.hidden,
        HIDDEN,
    )
    _residual(runtime, bufs.hidden, bufs.pos, patches * HIDDEN)
    for layer in range(DEPTH):
        _block(runtime, bufs, layer, patches)
    _layernorm(runtime, bufs.hidden, "model.visual.merger.norm", bufs.normed, patches)
    merged = patches // (MERGE * MERGE)
    _linear(
        runtime,
        bufs,
        bufs.normed,
        merged,
        MERGE_DIM,
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc1.bias",
        bufs.packed,
        MERGE_DIM,
    )
    _gelu(runtime, bufs.packed, merged * MERGE_DIM, erf=True)
    _linear(
        runtime,
        bufs,
        bufs.packed,
        merged,
        MERGE_DIM,
        "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.linear_fc2.bias",
        bufs.out,
        OUT,
    )
    session.end()
    embedded = runtime.session.download_f32(bufs.out, merged * OUT).reshape(merged, OUT)
    if not np.isfinite(embedded).all():
        message = "vision embedding is not finite"
        raise RuntimeError(message)
    _note(runtime, f"vision merged {merged}")
    return embedded


class _Buffers:
    def __init__(self, session: Any, patches: int) -> None:
        pad = (patches + 63) // 64 * 64
        self.patches = patches
        self.pad = pad
        self.hidden = session.zeros(patches * HIDDEN * 4)
        self.normed = session.zeros(patches * HIDDEN * 4)
        self.wide = session.zeros(pad * WIDE * 4)
        self.packed = session.zeros(patches * WIDE * 4)
        self.half = session.zeros(pad * WIDE * 2)
        self.q = session.zeros(patches * HIDDEN * 4)
        self.k = session.zeros(patches * HIDDEN * 4)
        self.v = session.zeros(patches * HIDDEN * 4)
        self.y = session.zeros(patches * HIDDEN * 4)
        self.cos = session.zeros(patches * HEAD * 4)
        self.sin = session.zeros(patches * HEAD * 4)
        self.table = session.zeros(GRID * GRID * HIDDEN * 4)
        self.pos = session.zeros(patches * HIDDEN * 4)
        self.pixels = session.zeros(patches * PATCH_DIM * 4)
        self.out = session.zeros((patches // (MERGE * MERGE)) * OUT * 4)
        self.tail = session.zeros(patches * 32 * 4)


def _block(runtime: Any, bufs: _Buffers, layer: int, patches: int) -> None:
    base = f"model.visual.blocks.{layer}."
    _layernorm(runtime, bufs.hidden, base + "norm1", bufs.normed, patches)
    _linear(
        runtime,
        bufs,
        bufs.normed,
        patches,
        HIDDEN,
        base + "attn.qkv.weight",
        base + "attn.qkv.bias",
        bufs.packed,
        QKV,
    )
    runtime.session.encode(
        "split_qkv72",
        [(bufs.packed, 0), (bufs.q, 0), (bufs.k, 0), (bufs.v, 0)],
        [_u32(patches)],
        ((QKV // 3 + 31) // 32, patches, 1),
        (32, 1, 1),
    )
    _rope(runtime, bufs, bufs.q, patches)
    _rope(runtime, bufs, bufs.k, patches)
    runtime.session.encode(
        "vision_attend72",
        [(bufs.q, 0), (bufs.k, 0), (bufs.v, 0), (bufs.y, 0)],
        [_u32(patches), _u32(HEADS)],
        (HEADS, patches, 1),
        (96, 1, 1),
    )
    _linear(
        runtime,
        bufs,
        bufs.y,
        patches,
        HIDDEN,
        base + "attn.proj.weight",
        base + "attn.proj.bias",
        bufs.normed,
        HIDDEN,
    )
    _residual(runtime, bufs.hidden, bufs.normed, patches * HIDDEN)
    _layernorm(runtime, bufs.hidden, base + "norm2", bufs.normed, patches)
    _linear(
        runtime,
        bufs,
        bufs.normed,
        patches,
        HIDDEN,
        base + "mlp.linear_fc1.weight",
        base + "mlp.linear_fc1.bias",
        bufs.packed,
        INTER,
    )
    _gelu(runtime, bufs.packed, patches * INTER, erf=False)
    _linear(
        runtime,
        bufs,
        bufs.packed,
        patches,
        INTER,
        base + "mlp.linear_fc2.weight",
        base + "mlp.linear_fc2.bias",
        bufs.normed,
        HIDDEN,
    )
    _residual(runtime, bufs.hidden, bufs.normed, patches * HIDDEN)


def _linear(
    runtime: Any,
    bufs: _Buffers,
    source: Any,
    rows: int,
    logical_k: int,
    weight: str,
    bias: str,
    dest: Any,
    logical_cols: int,
) -> None:
    file, offset, entry = runtime._loc(weight)
    stored_cols, stored_k = (int(value) for value in entry["stored_shape"])
    aligned = stored_cols // 32 * 32
    tail_n = stored_cols - aligned
    if (
        stored_cols != logical_cols
        or stored_cols > WIDE
        or stored_k > WIDE
        or stored_k % 32
        or aligned == 0
    ):
        message = f"{weight} stored {(stored_cols, stored_k)} does not fit the vision tile"
        raise RuntimeError(message)
    pad = (rows + 63) // 64 * 64
    if pad > bufs.pad:
        message = f"vision rows {rows} exceed the patch buffer"
        raise RuntimeError(message)
    bias_file, bias_at, _bias_entry = runtime._loc(bias)
    runtime.session.encode(
        "cast_pad_f16",
        [(source, 0), (bufs.half, 0)],
        [_u32(rows), _u32(pad), _u32(logical_k), _u32(stored_k)],
        ((stored_k + 31) // 32, (pad + 7) // 8, 1),
        (32, 8, 1),
    )
    fp8 = entry.get("dtype") == "fp8"
    wide = aligned % 64 == 0
    tile = 64 if wide else 32
    kind = "fp8" if fp8 else "fp4"
    width = int(runtime.session.pipes["prefill_fp4_e8m0"].threadExecutionWidth())
    scale_at = runtime._scale(entry, offset)
    runtime.session.encode(
        f"prefill_{kind}_n64" if wide else f"prefill_{kind}_e8m0",
        [(bufs.half, 0), (file, offset), (file, scale_at), (bufs.wide, 0)],
        [_u32(pad), _u32(aligned), _u32(stored_k)],
        (aligned // tile, pad // 64, 1),
        (width * 4, 1, 1),
    )
    if tail_n:
        stride = stored_k if fp8 else stored_k // 2
        runtime.session.encode(
            "gemv_fp8_tail" if fp8 else "gemv_fp4_tail",
            [
                (source, 0),
                (file, offset + aligned * stride),
                (file, scale_at + aligned * (stored_k // 32)),
                (bufs.tail, 0),
            ],
            [_u32(rows), _u32(tail_n), _u32(logical_k), _u32(stored_k)],
            (tail_n, 1, 1),
            (32, 1, 1),
        )
    runtime.session.encode(
        "pack_bias_f32",
        [(bufs.wide, 0), (bias_file, bias_at), (dest, 0), (bufs.tail, 0)],
        [_u32(rows), _u32(aligned), _u32(logical_cols), _u32(tail_n)],
        ((logical_cols + 31) // 32, (rows + 7) // 8, 1),
        (32, 8, 1),
    )


def _layernorm(runtime: Any, source: Any, name: str, dest: Any, rows: int) -> None:
    weight, weight_at, _entry = runtime._loc(name + ".weight")
    bias, bias_at, _bias = runtime._loc(name + ".bias")
    runtime.session.encode(
        "layernorm_rows_f32",
        [(source, 0), (weight, weight_at), (bias, bias_at), (dest, 0)],
        [_u32(HIDDEN), _f32(float(EPS)), _u32(rows)],
        (rows, 1, 1),
        (256, 1, 1),
    )


def _rope(runtime: Any, bufs: _Buffers, values: Any, patches: int) -> None:
    runtime.session.encode(
        "vision_rope72",
        [(values, 0), (bufs.cos, 0), (bufs.sin, 0)],
        [_u32(patches), _u32(HEADS)],
        (patches, 1, 1),
        (256, 1, 1),
    )


def _gelu(runtime: Any, values: Any, count: int, *, erf: bool) -> None:
    runtime.session.encode(
        "gelu_erf_n" if erf else "gelu_tanh_n",
        [(values, 0), (values, 0)],
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


def _dequant_positions(runtime: Any, bufs: _Buffers) -> None:
    file, offset, entry = runtime._loc("model.visual.pos_embed.weight")
    stored_rows, stored_k = (int(value) for value in entry["stored_shape"])
    if stored_rows < GRID * GRID or stored_k % 32 or HIDDEN % 32:
        message = f"position table stored {(stored_rows, stored_k)}"
        raise RuntimeError(message)
    session = runtime.session
    kernel = "dequant_fp8_rows" if entry.get("dtype") == "fp8" else "dequant_fp4_rows"
    session.begin()
    session.encode(
        kernel,
        [(file, offset), (file, runtime._scale(entry, offset)), (bufs.table, 0)],
        [_u32(HIDDEN), _u32(stored_k), _u32(GRID * GRID)],
        (GRID * GRID, 1, 1),
        (32, 1, 1),
    )
    session.end()


def _vision_config(root: Any) -> tuple[dict[str, Any], int]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    vision = config.get("vision_config")
    image_id = config.get("image_token_id")
    if not isinstance(vision, dict) or not isinstance(image_id, int):
        message = "the checkpoint has no vision config"
        raise RuntimeError(message)
    return vision, image_id


def _check_config(vision: dict[str, Any]) -> None:
    for key, value in _EXPECTED.items():
        if int(vision.get(key, -1)) != value:
            message = f"vision {key} is {vision.get(key)}"
            raise RuntimeError(message)
    if vision.get("hidden_act") != "gelu_pytorch_tanh":
        message = f"vision activation is {vision.get('hidden_act')}"
        raise RuntimeError(message)
    indexes = vision.get("deepstack_visual_indexes") or []
    if indexes:
        message = "vision deepstack indexes are set"
        raise RuntimeError(message)


def _image_blobs(messages: list[dict[str, Any]]) -> list[bytes]:
    blobs: list[bytes] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and _is_image(item):
                blobs.append(_image_bytes(item))
    return blobs


def _is_image(item: dict[str, Any]) -> bool:
    return "image" in item or "image_url" in item or item.get("type") == "image"


def _image_bytes(item: dict[str, Any]) -> bytes:
    raw = item.get("image_url", item.get("image"))
    if isinstance(raw, dict):
        raw = raw.get("url")
    if not isinstance(raw, str) or not raw:
        message = "image part has no data"
        raise RuntimeError(message)
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        if ";base64" not in header:
            message = "image data url is not base64"
            raise RuntimeError(message)
        return base64.b64decode(payload)
    return base64.b64decode(raw)


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


def _chunk(blob: bytes, offset: int) -> tuple[bytes, bytes, int]:
    if offset + 8 > len(blob):
        message = "png chunk is short"
        raise RuntimeError(message)
    length = struct.unpack(">I", blob[offset : offset + 4])[0]
    kind = blob[offset + 4 : offset + 8]
    end = offset + 8 + length
    if end + 4 > len(blob):
        message = "png chunk is short"
        raise RuntimeError(message)
    payload = blob[offset + 8 : end]
    crc = struct.unpack(">I", blob[end : end + 4])[0]
    expect = zlib.crc32(kind + payload) & 0xFFFFFFFF
    if crc != expect:
        message = "png chunk is corrupt"
        raise RuntimeError(message)
    return kind, payload, end + 4


def _unfilter(kind: int, row: np.ndarray, prev: np.ndarray, channels: int) -> None:
    for index, value in enumerate(row.tolist()):
        left = int(row[index - channels]) if index >= channels else 0
        up = int(prev[index])
        upper_left = int(prev[index - channels]) if index >= channels else 0
        if kind == 0:
            mixed = value
        elif kind == 1:
            mixed = value + left
        elif kind == 2:
            mixed = value + up
        elif kind == 3:
            mixed = value + ((left + up) // 2)
        elif kind == 4:
            mixed = value + _paeth(left, up, upper_left)
        else:
            message = f"png filter {kind} is unsupported"
            raise RuntimeError(message)
        row[index] = mixed & 255


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_upper = abs(estimate - upper_left)
    if distance_left <= distance_up and distance_left <= distance_upper:
        return left
    if distance_up <= distance_upper:
        return up
    return upper_left


def _note(runtime: Any, text: str) -> None:
    note = getattr(runtime, "_status", None)
    if callable(note):
        note(text)


def _write(buffer: Any, values: np.ndarray) -> None:
    blob = np.ascontiguousarray(values).tobytes()
    memoryview(buffer.contents().as_buffer(len(blob)))[: len(blob)] = blob


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)
