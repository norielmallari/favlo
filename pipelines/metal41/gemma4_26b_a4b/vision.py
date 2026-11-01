"""Gemma 4 vision tower. Loaded on the first image, from the packed q4_0 rows.

The resize is Algorithm 1 of the Gemma 4 technical report. The encoder is the
550M ViT in that report: 27 layers, width 1152, 16 heads of dimension 72,
non-causal attention, axial 2D-RoPE, and 2D absolute positions. A 3×3 average
and the embedding projection replace ``<|image|>``. The pixel resize is the
only host step. The tower itself is Metal.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from favlo.memory import memory
from gemma4_26b_a4b.geometry import (
    HIDDEN,
    IMAGE_TOKEN,
    PATCH,
    POOL,
    RMS_EPS,
    VISION_HEAD,
    VISION_HEADS,
    VISION_LAYERS,
    VISION_MLP,
    VISION_ROPE,
    VISION_SOFT_MAX,
    VISION_WIDTH,
    pooled_size,
)
from gemma4_26b_a4b.planes import Q40, align_k

_MAX_PATCHES = VISION_SOFT_MAX * POOL * POOL
_MAX_SIDE = VISION_SOFT_MAX * POOL
_POS_ROWS = 10240
_MLP_PAD = align_k(VISION_MLP, Q40)
_ATTN_SCALE = 1.0 / math.sqrt(VISION_HEAD)


@dataclass
class _Matrix:
    data: object
    rows: int
    k: int
    k_pad: int


@dataclass
class _Layer:
    input_norm: object
    post_attn: object
    pre: object
    post: object
    q_norm: object
    k_norm: object
    q: _Matrix
    k: _Matrix
    v: _Matrix
    o: _Matrix
    gate: _Matrix
    up: _Matrix
    down: _Matrix


def collect_images(messages: list[dict[str, Any]]) -> list[bytes]:
    """Pictures in the order the chat template emits ``<|image|>``."""
    items = [item for item in messages if isinstance(item, dict)]
    if items and items[0].get("role") in {"system", "developer"}:
        items = items[1:]
    found: list[bytes] = []
    index = 0
    while index < len(items):
        message = items[index]
        if message.get("role") == "tool":
            index += 1
            continue
        if message.get("tool_calls"):
            follow = index + 1
            while follow < len(items) and items[follow].get("role") == "tool":
                _walk(items[follow].get("content"), found)
                follow += 1
        _walk(message.get("content"), found)
        index += 1
    return found


def image_soft_tokens(blob: bytes) -> int:
    """Soft tokens for this PNG. The count is Algorithm 1, before the tower runs."""
    if len(blob) < 24 or not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        message = "image is not a PNG"
        raise RuntimeError(message)
    width, height = struct.unpack(">II", blob[16:24])
    _target_h, _target_w, soft = pooled_size(int(height), int(width))
    return soft


def decode_png(blob: bytes) -> np.ndarray:
    """RGB float32 in ``[0, 1]``, shape ``(height, width, 3)``."""
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        message = "image is not a PNG"
        raise RuntimeError(message)
    width = height = 0
    bit = color = interlace = 0
    chunks: list[bytes] = []
    cursor = 8
    while cursor + 8 <= len(blob):
        length = struct.unpack(">I", blob[cursor : cursor + 4])[0]
        kind = blob[cursor + 4 : cursor + 8]
        start = cursor + 8
        end = start + length
        if end + 4 > len(blob):
            message = "image PNG is truncated"
            raise RuntimeError(message)
        data = blob[start:end]
        if kind == b"IHDR":
            width, height, bit, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            chunks.append(data)
        elif kind == b"IEND":
            break
        cursor = end + 4
    if bit != 8 or color not in {2, 6} or interlace != 0 or width < 1 or height < 1:
        message = "image PNG must be 8-bit RGB"
        raise RuntimeError(message)
    channels = 3 if color == 2 else 4
    raw = zlib.decompress(b"".join(chunks))
    row_bytes = width * channels
    plane = _unfilter(raw, height, row_bytes)
    rgb = plane.reshape(height, width, channels)[:, :, :3]
    return rgb.astype(np.float32) * np.float32(1.0 / 255.0)


def resize_bicubic(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Catmull-Rom resize. Output pixels sit on the half-pixel grid."""
    src_h, src_w, _channels = image.shape
    if src_h == out_h and src_w == out_w:
        return np.ascontiguousarray(image, dtype=np.float32)
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    out = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for dy in range(4):
        sample_y = y0 - 1 + dy
        weight_y = _cubic(ys - sample_y).astype(np.float32)
        row = np.clip(sample_y, 0, src_h - 1)
        for dx in range(4):
            sample_x = x0 - 1 + dx
            weight_x = _cubic(xs - sample_x).astype(np.float32)
            col = np.clip(sample_x, 0, src_w - 1)
            picked = image[row][:, col]
            out += weight_y[:, None, None] * weight_x[None, :, None] * picked
    return np.clip(out, 0.0, 1.0)


def patch_tokens(image: np.ndarray) -> np.ndarray:
    """``16×16×3`` patches in row order, one row per patch."""
    height, width, _channels = image.shape
    grid_h = height // PATCH
    grid_w = width // PATCH
    view = image.reshape(grid_h, PATCH, grid_w, PATCH, 3)
    ordered = view.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(ordered.reshape(grid_h * grid_w, PATCH * PATCH * 3))


class VisionTower:
    """The packed tower. ``encode`` loads it the first time an image arrives."""

    def __init__(self, session: Any, root: Path, owner: str) -> None:
        self.session = session
        self.root = root
        self.owner = owner
        self._ready = False
        self._pos_offset = 0
        self.layers: list[_Layer] = []

    def encode(self, png: bytes) -> tuple[object, int]:
        """Soft tokens ``(count, hidden)`` already in the text embedding space."""
        self._open()
        image = decode_png(png)
        target_h, target_w, soft = pooled_size(int(image.shape[0]), int(image.shape[1]))
        resized = resize_bicubic(image, target_h, target_w)
        patches = patch_tokens(resized)
        count = int(patches.shape[0])
        grid_h = target_h // PATCH
        grid_w = target_w // PATCH
        if count > _MAX_PATCHES or soft > VISION_SOFT_MAX:
            message = f"image uses {count} patches"
            raise RuntimeError(message)
        memory.store(self.patch, 0, np.ascontiguousarray(patches, dtype=np.float32).tobytes())
        self._gemm(self.patch, self.input_proj, count, self.hidden)
        self._standardize(count)
        self._positions(grid_h, grid_w)
        self.session.flush()
        for layer in self.layers:
            self._block(layer, count, grid_w)
            self.session.flush()
        self._pool(grid_w, soft)
        self._gemm(self.pooled, self.project, soft, self.soft)
        self.session.flush()
        return self.soft, soft

    def _open(self) -> None:
        if self._ready:
            return
        import json

        index = json.loads((self.root / "resident.json").read_text(encoding="utf-8"))
        wanted = [row for row in index if _vision_name(str(row["name"]))]
        found: dict[str, _Matrix | object] = {}
        blob = self.root / "resident.bin"
        raw = open(blob, "rb", buffering=0)
        try:
            for row in wanted:
                name = str(row["name"])
                shape = tuple(int(item) for item in row["shape"])
                plane = str(row["plane"])
                nbytes = int(row["nbytes"])
                offset = int(row["offset"])
                if name.endswith("position_embedding_table"):
                    self._pos_offset = offset
                    continue
                raw.seek(offset)
                packed = raw.read(nbytes)
                if len(packed) != nbytes:
                    message = f"{name} is short"
                    raise RuntimeError(message)
                if plane == "fp16" and len(shape) == 1:
                    found[name] = self._expand(packed)
                elif plane == Q40 and len(shape) == 2:
                    handle = self.session.zeros(nbytes)
                    memory.store(handle, 0, packed)
                    rows, k = int(shape[0]), int(shape[1])
                    found[name] = _Matrix(handle, rows, k, align_k(k, Q40))
                else:
                    message = f"{name} is not a vision matrix"
                    raise RuntimeError(message)
        finally:
            raw.close()
        self.input_proj = _need(found, "model.vision_tower.patch_embedder.input_proj.weight")
        self.project = _need(found, "model.embed_vision.embedding_projection.weight")
        self.std_scale = _need(found, "model.vision_tower.std_scale")
        self.std_bias = _need(found, "model.vision_tower.std_bias")
        self.layers = []
        for layer in range(VISION_LAYERS):
            base = f"model.vision_tower.encoder.layers.{layer}"
            self.layers.append(
                _Layer(
                    input_norm=_need(found, f"{base}.input_layernorm.weight"),
                    post_attn=_need(found, f"{base}.post_attention_layernorm.weight"),
                    pre=_need(found, f"{base}.pre_feedforward_layernorm.weight"),
                    post=_need(found, f"{base}.post_feedforward_layernorm.weight"),
                    q_norm=_need(found, f"{base}.self_attn.q_norm.weight"),
                    k_norm=_need(found, f"{base}.self_attn.k_norm.weight"),
                    q=_need(found, f"{base}.self_attn.q_proj.linear.weight"),
                    k=_need(found, f"{base}.self_attn.k_proj.linear.weight"),
                    v=_need(found, f"{base}.self_attn.v_proj.linear.weight"),
                    o=_need(found, f"{base}.self_attn.o_proj.linear.weight"),
                    gate=_need(found, f"{base}.mlp.gate_proj.linear.weight"),
                    up=_need(found, f"{base}.mlp.up_proj.linear.weight"),
                    down=_need(found, f"{base}.mlp.down_proj.linear.weight"),
                )
            )
        session = self.session
        self.patch = session.zeros(_MAX_PATCHES * PATCH * PATCH * 3 * 4)
        self.hidden = session.zeros(_MAX_PATCHES * VISION_WIDTH * 4)
        self.norm = session.zeros(_MAX_PATCHES * VISION_WIDTH * 4)
        self.query = session.zeros(_MAX_PATCHES * VISION_WIDTH * 4)
        self.key = session.zeros(_MAX_PATCHES * VISION_WIDTH * 4)
        self.value = session.zeros(_MAX_PATCHES * VISION_WIDTH * 4)
        self.mix = session.zeros(_MAX_PATCHES * VISION_WIDTH * 4)
        self.gate = session.zeros(_MAX_PATCHES * VISION_MLP * 4)
        self.up = session.zeros(_MAX_PATCHES * VISION_MLP * 4)
        self.mid = session.zeros(_MAX_PATCHES * VISION_MLP * 4)
        self.wide = session.zeros(_MAX_PATCHES * _MLP_PAD * 4)
        self.half_w = session.zeros(_MLP_PAD * VISION_WIDTH * 2)
        self.half_a = session.zeros(_MAX_PATCHES * _MLP_PAD * 2)
        self.pooled = session.zeros(VISION_SOFT_MAX * VISION_WIDTH * 4)
        self.soft = session.zeros(VISION_SOFT_MAX * HIDDEN * 4)
        self.pos_row = session.zeros(_MAX_SIDE * VISION_WIDTH * 4)
        self.pos_col = session.zeros(_MAX_SIDE * VISION_WIDTH * 4)
        self._ready = True

    def _expand(self, packed: bytes) -> object:
        count = len(packed) // 2
        source = self.session.zeros(len(packed))
        memory.store(source, 0, packed)
        dest = self.session.zeros(count * 4)
        self.session.dispatch(
            "expand_fp16",
            [source, dest],
            [self.session._u32(count)],
            *_blocks(count),
        )
        memory.release(source)
        return dest

    def _positions(self, grid_h: int, grid_w: int) -> None:
        if grid_h > _MAX_SIDE or grid_w > _MAX_SIDE:
            message = f"patch grid {grid_h}x{grid_w} is past the position table"
            raise RuntimeError(message)
        row_bytes = VISION_WIDTH * 2
        raw = open(self.root / "resident.bin", "rb", buffering=0)
        try:
            raw.seek(self._pos_offset)
            rows = raw.read(grid_h * row_bytes)
            raw.seek(self._pos_offset + _POS_ROWS * row_bytes)
            cols = raw.read(grid_w * row_bytes)
        finally:
            raw.close()
        if len(rows) != grid_h * row_bytes or len(cols) != grid_w * row_bytes:
            message = "position table is short"
            raise RuntimeError(message)
        self._expand_into(rows, self.pos_row)
        self._expand_into(cols, self.pos_col)
        count = grid_h * grid_w
        self.session.encode(
            "vision_positions",
            [self.hidden, self.pos_row, self.pos_col],
            [
                self.session._u32(grid_w),
                self.session._u32(VISION_WIDTH),
                self.session._u32(count),
            ],
            *_grid2(VISION_WIDTH, count),
        )

    def _expand_into(self, packed: bytes, dest: object) -> None:
        count = len(packed) // 2
        source = self.session.zeros(len(packed))
        memory.store(source, 0, packed)
        self.session.dispatch(
            "expand_fp16",
            [source, dest],
            [self.session._u32(count)],
            *_blocks(count),
        )
        memory.release(source)

    def _standardize(self, count: int) -> None:
        self.session.encode(
            "vision_standardize",
            [self.hidden, self.std_scale, self.std_bias],
            [
                self.session._u32(VISION_WIDTH),
                self.session._f32(RMS_EPS),
                self.session._u32(count),
            ],
            (count, 1, 1),
            (256, 1, 1),
        )

    def _block(self, layer: _Layer, count: int, grid_w: int) -> None:
        self._norm(self.hidden, layer.input_norm, self.norm, count)
        self._gemm(self.norm, layer.q, count, self.query)
        self._gemm(self.norm, layer.k, count, self.key)
        self._gemm(self.norm, layer.v, count, self.value)
        self._heads(self.query, layer.q_norm, count)
        self._heads(self.key, layer.k_norm, count)
        self.session.encode(
            "rope_axial",
            [self.query, self.key],
            [
                self.session._u32(VISION_HEADS),
                self.session._u32(grid_w),
                self.session._f32(VISION_ROPE),
                self.session._u32(count),
            ],
            (count, 1, 1),
            (256, 1, 1),
        )
        self.session.encode(
            "attend_vision",
            [self.query, self.key, self.value, self.mix],
            [
                self.session._u32(VISION_HEADS),
                self.session._u32(count),
                self.session._f32(_ATTN_SCALE),
            ],
            (count, VISION_HEADS, 1),
            (32, 1, 1),
        )
        self._gemm(self.mix, layer.o, count, self.query)
        self._norm(self.query, layer.post_attn, self.norm, count)
        self._axpy(self.hidden, self.norm, count * VISION_WIDTH)
        self._norm(self.hidden, layer.pre, self.norm, count)
        self._gemm(self.norm, layer.gate, count, self.gate)
        self._gemm(self.norm, layer.up, count, self.up)
        self.session.encode(
            "geglu_pair",
            [self.gate, self.up, self.mid],
            [self.session._u32(count * VISION_MLP)],
            *_blocks(count * VISION_MLP),
        )
        self._gemm(self.mid, layer.down, count, self.query)
        self._norm(self.query, layer.post, self.norm, count)
        self._axpy(self.hidden, self.norm, count * VISION_WIDTH)

    def _pool(self, grid_w: int, soft: int) -> None:
        self.session.encode(
            "pool_patches",
            [self.hidden, self.pooled],
            [
                self.session._u32(grid_w),
                self.session._u32(VISION_WIDTH),
                self.session._u32(soft),
            ],
            *_grid2(VISION_WIDTH, soft),
        )

    def _gemm(self, source: object, matrix: _Matrix, count: int, dest: object) -> None:
        padded = _pad_tokens(count)
        pad_rows = _pad_rows(matrix.rows)
        self.session.encode(
            "write_rows_tight",
            [source, self.half_a],
            [
                self.session._u32(matrix.k),
                self.session._u32(matrix.k_pad),
                self.session._u32(count),
            ],
            ((matrix.k_pad + 255) // 256, padded, 1),
            (256, 1, 1),
        )
        if pad_rows > matrix.rows:
            tail = (pad_rows - matrix.rows) * matrix.k_pad * 2
            self.session.encode(
                "zero_buffer",
                [self.half_w],
                [self.session._u32(tail)],
                *_blocks(tail),
                origins=[matrix.rows * matrix.k_pad * 2],
            )
        blocks = matrix.rows * (matrix.k_pad // 32)
        self.session.encode(
            "dequant_q4_0",
            [matrix.data, self.half_w],
            [self.session._u32(matrix.k_pad), self.session._u32(matrix.rows)],
            *_blocks(blocks),
        )
        target = dest if pad_rows == matrix.rows else self.wide
        self.session.matmul(
            "matmul_f16",
            self.half_a,
            self.half_w,
            None,
            target,
            padded,
            pad_rows,
            matrix.k_pad,
            commit=False,
        )
        if pad_rows == matrix.rows:
            return
        self.session.encode(
            "compact_rows",
            [self.wide, dest],
            [
                self.session._u32(pad_rows),
                self.session._u32(matrix.rows),
                self.session._u32(count),
            ],
            *_grid2(matrix.rows, count),
        )

    def _norm(self, source: object, weight: object, dest: object, count: int) -> None:
        self.session.encode(
            "rmsnorm_rows",
            [source, weight, dest],
            [
                self.session._u32(VISION_WIDTH),
                self.session._f32(RMS_EPS),
                self.session._u32(count),
            ],
            (count, 1, 1),
            (256, 1, 1),
        )

    def _heads(self, values: object, weight: object, count: int) -> None:
        self.session.encode(
            "rmsnorm_heads",
            [values, weight],
            [self.session._u32(VISION_HEAD), self.session._f32(RMS_EPS)],
            (count * VISION_HEADS, 1, 1),
            (96, 1, 1),
        )

    def _axpy(self, hidden: object, update: object, count: int) -> None:
        self.session.encode(
            "residual_scale",
            [hidden, update],
            [self.session._f32(1.0), self.session._u32(count)],
            *_blocks(count),
        )


def _vision_name(name: str) -> bool:
    return "vision_tower" in name or "embed_vision" in name


def _need(found: dict[str, Any], name: str) -> Any:
    if name not in found:
        message = f"{name} is not in the packed record"
        raise RuntimeError(message)
    return found[name]


def _pad_tokens(count: int) -> int:
    return (count + 7) & ~7


def _pad_rows(rows: int) -> int:
    return (rows + 31) & ~31


def _blocks(count: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((count + 255) // 256, 1, 1), (256, 1, 1)


def _grid2(width: int, rows: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((width + 255) // 256, rows, 1), (256, 1, 1)


def _walk(content: Any, found: list[bytes]) -> None:
    if isinstance(content, str) or content is None or not isinstance(content, list):
        return
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in {"video", "audio", "input_audio"} or "video" in item or "video_url" in item:
            message = "video and audio input are not supported"
            raise RuntimeError(message)
        if kind in {"image", "image_url"} or "image" in item or "image_url" in item:
            found.append(_image_bytes(item))


def _image_bytes(item: dict[str, Any]) -> bytes:
    raw = item.get("image_url", item.get("image", item.get("url")))
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
        message = "image url must be a data url"
        raise RuntimeError(message)
    try:
        return base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        message = "image url must be a data url"
        raise RuntimeError(message) from exc


def _unfilter(raw: bytes, height: int, row_bytes: int) -> np.ndarray:
    stride = row_bytes + 1
    if len(raw) < height * stride:
        message = "image PNG is truncated"
        raise RuntimeError(message)
    rows = np.frombuffer(raw, dtype=np.uint8, count=height * stride).reshape(height, stride)
    out = np.zeros((height, row_bytes), dtype=np.uint8)
    previous = np.zeros(row_bytes, dtype=np.int32)
    for y in range(height):
        kind = int(rows[y, 0])
        current = rows[y, 1:].astype(np.int32)
        if kind == 0:
            rebuilt = current
        elif kind == 1:
            rebuilt = np.cumsum(current) & 255
        elif kind == 2:
            rebuilt = (current + previous) & 255
        elif kind == 3:
            rebuilt = _average(current, previous)
        elif kind == 4:
            rebuilt = _paeth(current, previous)
        else:
            message = f"image PNG filter {kind} is unsupported"
            raise RuntimeError(message)
        out[y] = rebuilt.astype(np.uint8)
        previous = rebuilt
    return out


def _average(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    rebuilt = np.empty_like(current)
    left = 0
    for index, value in enumerate(current):
        rebuilt[index] = (int(value) + ((left + int(previous[index])) // 2)) & 255
        left = int(rebuilt[index])
    return rebuilt


def _paeth(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    rebuilt = np.empty_like(current)
    left = 0
    upper_left = 0
    for index, value in enumerate(current):
        up = int(previous[index])
        base = left + up - upper_left
        nearest = left
        if abs(base - up) < abs(base - nearest):
            nearest = up
        if abs(base - upper_left) < abs(base - nearest):
            nearest = upper_left
        rebuilt[index] = (int(value) + nearest) & 255
        upper_left = up
        left = int(rebuilt[index])
    return rebuilt


def _cubic(distance: np.ndarray) -> np.ndarray:
    span = np.abs(distance)
    near = span < 1.0
    far = (span >= 1.0) & (span < 2.0)
    weight = np.zeros_like(span, dtype=np.float32)
    close = span[near]
    weight[near] = ((1.5 * close - 2.5) * close) * close + 1.0
    edge = span[far]
    weight[far] = ((-0.5 * edge + 2.5) * edge - 4.0) * edge + 2.0
    return weight
