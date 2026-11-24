"""Inkling vision tower.

A still is padded to a 40-pixel grid. Each language token is four 20-pixel
patches. The tower is a 23-layer pre-norm encoder with axial RoPE, then a
two-layer projector. Rank-2 weights are E4M3 with one float scale per 32
along K. The open reads ``vision.bin`` in 8 MiB pieces and releases it after
the rows are written. A later request with the same bytes reads
``~/.favlo/vision-cache/inkling_2_mini``. An image already in the prefix is
not encoded again.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from favlo.memory import memory
from favlo.vision_cache import cache_path, load_cache, store_cache
from inkling_2_mini.metal41.codec import e4m3_to_float
from inkling_2_mini.metal41.gptq import config_digest, write_pack
from inkling_2_mini.metal41.resident import pread_handle
from inkling_2_mini.metal41.session import f32, u32
from inkling_2_mini.metal41.weights import Tensors

PIPELINE = "inkling_2_mini"
PACK_NAME = "vision-fp8.safetensors"
PACK_FORMAT = "favlo-vision-fp8-1"
IMAGE_TOKEN = 200054

LAYERS = 23
HIDDEN = 1024
HEADS = 16
HEAD_DIM = 64
MID = 4096
TEXT = 2304
PATCH = 20
MERGE = 2
TOKEN = 40
GRID = 32
ROPE_THETA = 10000.0
LN_EPS = 1e-5
ATTN_SCALE = 0.125
FP8_MAX = 448.0

_PNG = b"\x89PNG\r\n\x1a\n"
_E4M3 = np.array([e4m3_to_float(i) for i in range(256)], np.float32)
_POS = np.array([i for i in range(128) if np.isfinite(_E4M3[i])], np.int32)
_POS = _POS[np.argsort(_E4M3[_POS])]
_VAL = _E4M3[_POS]
_MIDS = (_VAL[:-1] + _VAL[1:]) * 0.5


def token_grid(height: int, width: int) -> tuple[int, int]:
    """Language-token rows and columns. One token is 40 by 40 pixels."""
    if height < 1 or width < 1:
        message = "the image is empty"
        raise ValueError(message)
    return (height + TOKEN - 1) // TOKEN, (width + TOKEN - 1) // TOKEN


def quantize_fp8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 codes and a float scale per 32 along K. K is padded to a multiple of 64."""
    values = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, cols = values.shape
    pad = (cols + 63) // 64 * 64
    if pad != cols:
        wide = np.zeros((rows, pad), np.float32)
        wide[:, :cols] = values
        values = wide
    groups = pad // 32
    blocks = values.reshape(rows, groups, 32)
    peak = np.max(np.abs(blocks), axis=-1)
    scale = np.where(peak == 0, np.float32(1.0), peak / np.float32(FP8_MAX)).astype(np.float32)
    scaled = blocks / scale[..., None]
    mag = np.abs(scaled)
    index = np.clip(np.searchsorted(_MIDS, mag, side="right"), 0, len(_POS) - 1)
    codes = _POS[index].astype(np.uint8)
    codes = np.where(scaled < 0, codes | np.uint8(0x80), codes)
    return np.ascontiguousarray(codes.reshape(rows, pad)), np.ascontiguousarray(scale)


def dequant_fp8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Rebuild a matrix from ``quantize_fp8``."""
    rows, inner = codes.shape
    values = _E4M3[codes]
    group = np.repeat(scales.reshape(rows, inner // 32), 32, axis=1)
    return values * group


def patch_hw(height: int, width: int, merge: int = MERGE) -> np.ndarray:
    """Native patch coordinates in merge-block order. ``height`` and ``width`` are patches."""
    hpos, wpos = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    shape = (height // merge, merge, width // merge, merge)
    hpos = hpos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    wpos = wpos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    return np.stack([hpos, wpos], axis=1).astype(np.int32)


def axis_taps(index: np.ndarray, size: float, side: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear taps into a ``side`` table. ``align_corners`` is false."""
    src = (index.astype(np.float32) + 0.5) * float(side) / float(size) - 0.5
    floor = np.floor(src)
    taps = np.stack([floor, floor + 1.0], axis=-1)
    taps = np.clip(taps, 0, side - 1).astype(np.int32)
    distance = np.abs(src[:, None] - floor[:, None] - np.array([0.0, 1.0], np.float32))
    weights = np.clip(1.0 - distance, 0.0, None).astype(np.float32)
    return taps, weights


def interpolate(hw: np.ndarray, height: int, width: int, table: np.ndarray) -> np.ndarray:
    """Sample the learned 32 by 32 position table at native patch coordinates."""
    row = hw[:, 0].astype(np.float32)
    col = hw[:, 1].astype(np.float32)
    h_taps, h_weights = axis_taps(row, height)
    w_taps, w_weights = axis_taps(col, width)
    out = np.zeros((len(hw), table.shape[1]), np.float32)
    for i in range(2):
        for j in range(2):
            index = h_taps[:, i] * GRID + w_taps[:, j]
            weight = (h_weights[:, i] * w_weights[:, j])[:, None]
            out += table[index] * weight
    return out


def rope_tables(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cos and sin for axial RoPE. ``coords`` is ``(seq, 2)`` of height, width."""
    spatial = HEAD_DIM // 2
    exp = np.arange(0, spatial, 2, dtype=np.float32) / np.float32(spatial)
    inv = np.float32(1.0) / (np.float32(ROPE_THETA) ** exp)
    freq = coords[:, :, None] * inv[None, None, :]
    cos = np.cos(freq)
    sin = np.sin(freq)
    cos_full = np.repeat(np.concatenate([cos[:, 1], cos[:, 0]], axis=-1), 2, axis=-1)
    sin_full = np.repeat(np.concatenate([sin[:, 1], sin[:, 0]], axis=-1), 2, axis=-1)
    return cos_full.astype(np.float32), sin_full.astype(np.float32)


def png_size(blob: bytes) -> tuple[int, int]:
    """Height and width of an 8-bit RGB or RGBA PNG."""
    _check_png(blob)
    return _ihdr(blob)[:2]


def decode_png(blob: bytes) -> np.ndarray:
    """RGB uint8, height by width by 3. An alpha plane is composited on the checkerboard."""
    _check_png(blob)
    height, width, depth, color, interlace = _ihdr(blob)
    if depth != 8 or interlace != 0 or color not in (2, 6):
        message = "the image is not an 8-bit PNG"
        raise ValueError(message)
    channels = 3 if color == 2 else 4
    raw = _inflate(blob)
    stride = width * channels
    if len(raw) != height * (stride + 1):
        message = "the PNG rows do not match its header"
        raise ValueError(message)
    plane = _unfilter(raw, height, width, channels)
    rgb = plane.reshape(height, width, channels)
    if channels == 3:
        return np.ascontiguousarray(rgb)
    return _composite(rgb)


def expand_prompt(
    messages: list[dict[str, Any]], ids: list[int]
) -> tuple[list[int], list[tuple[int, int, bytes]]]:
    """Replace each image token with one id per 40 by 40 tile."""
    blobs = _image_blobs(messages)
    spots = [index for index, token in enumerate(ids) if token == IMAGE_TOKEN]
    if not blobs and not spots:
        return list(ids), []
    if len(blobs) != len(spots):
        message = "the image tokens do not match the images"
        raise ValueError(message)
    out: list[int] = []
    plan: list[tuple[int, int, bytes]] = []
    cursor = 0
    for spot, blob in zip(spots, blobs, strict=True):
        out.extend(ids[cursor:spot])
        height, width = png_size(blob)
        rows, cols = token_grid(height, width)
        count = rows * cols
        start = len(out)
        out.extend([IMAGE_TOKEN] * count)
        plan.append((start, count, blob))
        cursor = spot + 1
    out.extend(ids[cursor:])
    return out, plan


def ensure_pack(root: Path, dest: Path | None = None, log: Any = None) -> None:
    """Write the fp8 vision pack when it is missing or does not match ``config.json``."""
    base = Path(root)
    pack = Path(dest or root)
    pack.mkdir(parents=True, exist_ok=True)
    path = pack / PACK_NAME
    if _pack_ready(path, base):
        return
    if log is not None:
        log("baking the vision fp8 planes")
    _write_pack(base, path, log)


def bind_vision(runtime: Any, tokens: list[int], hit: int, dropped: int) -> None:
    """Encode images past the prefix hit. Fully covered images are not loaded."""
    plan = getattr(runtime, "_vision_plan", None) or []
    runtime._vision_bank = None
    runtime._vision_slots = None
    if not plan:
        return
    pending = _shift(plan, dropped)
    if not pending:
        return
    rows: list[np.ndarray] = []
    slots = np.full(len(tokens), -1, np.int32)
    tower: Tower | None = None
    try:
        for start, count, blob in pending:
            if start + count <= hit:
                _vision_line(runtime, "cached", count)
                continue
            found = _cached_rows(runtime.root, blob, count)
            if found is None:
                _vision_line(runtime, "patches", count)
                if tower is None:
                    tower = Tower(runtime.root, runtime.session, getattr(runtime, "packs", None))
                found = tower.encode_blob(blob)
                if found.shape != (count, TEXT):
                    message = "vision rows do not match the image"
                    raise RuntimeError(message)
                store_cache(_cache_file(runtime.root, blob), {"rows": found})
                _vision_line(runtime, "merged", count)
            else:
                _vision_line(runtime, "cache", count)
            base = sum(int(item.shape[0]) for item in rows)
            rows.append(found)
            for offset in range(count):
                if start + offset >= hit:
                    slots[start + offset] = base + offset
    finally:
        if tower is not None:
            tower.close()
    if not rows:
        return
    bank = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float16)
    owner = f"{runtime._owner}:vision"
    runtime._vision_owner = owner
    runtime._vision_bank = memory.upload(bank.tobytes(), owner=owner)
    runtime._vision_slots = memory.upload(np.ascontiguousarray(slots).tobytes(), owner=owner)


def release_vision(runtime: Any) -> None:
    """Drop the image rows. The tower weights are already released."""
    for name in ("_vision_bank", "_vision_slots"):
        handle = getattr(runtime, name, None)
        if handle is not None:
            memory.release(handle)
        setattr(runtime, name, None)
    runtime._vision_plan = None


class Tower:
    """One vision pack on the device. ``close`` releases it."""

    def __init__(self, root: Path, session: Any, pack_root: Path | None = None) -> None:
        from inkling_2_mini.metal41.record import ready

        self.root = Path(root)
        self.pack_root = Path(pack_root or root)
        if not ready(self.pack_root):
            ensure_pack(self.root, self.pack_root)
        self.session = session
        self.owner = f"inkling-vision:{id(self)}"
        self.pack = None
        self.spans: dict[str, tuple[int, tuple[int, ...]]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.position = np.zeros((GRID * GRID, HIDDEN), np.float32)
        self._scratch: list[Any] = []
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._drop_scratch()
        if self.pack is not None:
            memory.release(self.pack)
            self.pack = None

    def encode_blob(self, blob: bytes) -> np.ndarray:
        """Merged language rows, float16 ``[tokens, 2304]``."""
        rgb = decode_png(blob)
        rows, cols = token_grid(rgb.shape[0], rgb.shape[1])
        return self._encode(rgb, rows, cols)

    def _load(self) -> None:
        from inkling_2_mini.metal41.record import open_vision

        store = open_vision(self.pack_root)
        if store is not None:
            path = store.path
            tensors: Any = store
            data_off = 0
        else:
            path = self.pack_root / PACK_NAME
            tensors = Tensors(path)
            data_off = tensors._data
        try:
            if store is None:
                if tensors.metadata.get("format") != PACK_FORMAT:
                    message = "vision pack format does not match"
                    raise RuntimeError(message)
                if tensors.metadata.get("config") != config_digest(self.root):
                    message = "vision pack does not match config.json"
                    raise RuntimeError(message)
            for name, info in tensors.header.items():
                start, _end = info["data_offsets"]
                self.spans[name] = (int(start), tuple(int(v) for v in info["shape"]))
            for name in _vector_names():
                raw, _info = tensors.raw(name)
                self.vectors[name] = np.frombuffer(raw, np.float16).astype(np.float32).copy()
            codes, code_info = tensors.raw("pos.code")
            scales, scale_info = tensors.raw("pos.scale")
            code = np.frombuffer(codes, np.uint8).reshape(code_info["shape"]).copy()
            scale = _scale_array(scales, tuple(scale_info["shape"]))
            self.position = dequant_fp8(code, scale)
            nbytes = path.stat().st_size - data_off
            self.pack = memory.allocate(nbytes, owner=self.owner)
            if store is None:
                fd = os.open(path, os.O_RDONLY)
                try:
                    pread_handle(fd, self.pack, data_off, nbytes)
                finally:
                    os.close(fd)
            else:
                pread_handle(tensors.fileno(), self.pack, data_off, nbytes)
        finally:
            tensors.close()

    def _encode(self, rgb: np.ndarray, rows: int, cols: int) -> np.ndarray:
        tokens = rows * cols
        native_h = rows * MERGE
        native_w = cols * MERGE
        patches = _patch_matrix(_pixels(rgb, rows, cols))
        n_patches = patches.shape[0]
        seq = 1 + n_patches
        self._open_scratch(seq, tokens)
        try:
            self._assemble(patches, rows, cols, native_h, native_w)
            self._layernorm(self.hidden, "ln_pre.w", "ln_pre.b", self.branch, seq)
            self._copy(self.branch, self.hidden, seq, HIDDEN)
            for index in range(LAYERS):
                self._layer(index, seq)
            self.session.launch(
                "vision_merge",
                [self.hidden, self.mid],
                [u32(tokens), u32(HIDDEN)],
                tokens * MID,
            )
            self._linear(self.mid, tokens, MID, MID, "proj_in", None, self.mid)
            self.session.launch(
                "vision_gelu",
                [self.mid, self.mid],
                [u32(tokens * MID)],
                tokens * MID,
            )
            self._linear(self.mid, tokens, TEXT, MID, "proj_out", None, self.branch)
            raw = self.session.download(self.branch, tokens * TEXT * 2)
            return np.frombuffer(raw, np.float16).reshape(tokens, TEXT).copy()
        finally:
            self._drop_scratch()

    def _assemble(
        self, patches: np.ndarray, rows: int, cols: int, native_h: int, native_w: int
    ) -> None:
        n_patches = patches.shape[0]
        self._store(self.pixels, patches)
        self._linear(self.pixels, n_patches, HIDDEN, PATCH * PATCH * 3, "patch", None, self.q)
        hw = patch_hw(native_h, native_w)
        pos = interpolate(hw, native_h, native_w, self.position).astype(np.float16)
        self._store(self.pos, pos)
        cls = (self.vectors["class"] + self.vectors["cls_pos"]).astype(np.float16)
        memory.store(self.hidden, 0, np.ascontiguousarray(cls).tobytes())
        self.session.launch(
            "vision_add",
            [self.q, self.pos, (self.hidden, HIDDEN * 2)],
            [u32(n_patches * HIDDEN)],
            n_patches * HIDDEN,
        )
        seq = 1 + n_patches
        coords = np.zeros((seq, 2), np.float32)
        coords[1:] = hw.astype(np.float32) + 1.0
        cos, sin = rope_tables(coords)
        self._store(self.cos, cos)
        self._store(self.sin, sin)

    def _layer(self, index: int, seq: int) -> None:
        prefix = f"l{index}"
        self._copy(self.hidden, self.residual, seq, HIDDEN)
        self._layernorm(self.hidden, f"{prefix}.ln1.w", f"{prefix}.ln1.b", self.branch, seq)
        self._linear(self.branch, seq, HIDDEN, HIDDEN, f"{prefix}.q", f"{prefix}.q.b", self.q)
        self._linear(self.branch, seq, HIDDEN, HIDDEN, f"{prefix}.k", f"{prefix}.k.b", self.k)
        self._linear(self.branch, seq, HIDDEN, HIDDEN, f"{prefix}.v", f"{prefix}.v.b", self.v)
        self._rope(self.q, seq)
        self._rope(self.k, seq)
        self.session.dispatch(
            "vision_attend",
            [self.q, self.k, self.v, self.branch],
            [u32(seq), u32(HEADS), u32(HEAD_DIM), f32(ATTN_SCALE)],
            (seq, HEADS, 1),
            (32, 1, 1),
        )
        self._linear(self.branch, seq, HIDDEN, HIDDEN, f"{prefix}.o", f"{prefix}.o.b", self.q)
        self._residual(self.residual, self.q, f"{prefix}.ls1", self.hidden, seq)
        self._copy(self.hidden, self.residual, seq, HIDDEN)
        self._layernorm(self.hidden, f"{prefix}.ln2.w", f"{prefix}.ln2.b", self.branch, seq)
        self._linear(self.branch, seq, MID, HIDDEN, f"{prefix}.fc", f"{prefix}.fc.b", self.mid)
        self.session.launch("vision_gelu", [self.mid, self.mid], [u32(seq * MID)], seq * MID)
        self._linear(self.mid, seq, HIDDEN, MID, f"{prefix}.proj", f"{prefix}.proj.b", self.branch)
        self._residual(self.residual, self.branch, f"{prefix}.ls2", self.hidden, seq)

    def _linear(
        self,
        act: Any,
        rows: int,
        cols: int,
        inner: int,
        name: str,
        bias: str | None,
        dest: Any,
    ) -> None:
        code_off, code_shape = self.spans[f"{name}.code"]
        scale_off, scale_shape = self.spans[f"{name}.scale"]
        out, padded = code_shape
        if out != cols or padded < inner or padded % 64 or cols % 64:
            message = f"{name} is not a vision matrix"
            raise RuntimeError(message)
        if scale_shape[0] != out or scale_shape[1] != (padded // 32) * 4:
            message = f"{name} scales do not match"
            raise RuntimeError(message)
        self.session.launch(
            "vision_dequant",
            [(self.pack, code_off), (self.pack, scale_off), self.weight],
            [u32(out), u32(padded)],
            out * padded,
        )
        pad_rows = _align(rows)
        self.session.launch(
            "vision_pad",
            [act, self.act],
            [u32(rows), u32(inner), u32(pad_rows), u32(padded)],
            pad_rows * padded,
        )
        self.session.dispatch(
            "matmul_f16_m64",
            [self.act, self.weight, self.gemm],
            [u32(pad_rows), u32(cols), u32(padded)],
            (cols // 64, pad_rows // 64, 1),
        )
        use = 0
        if bias is not None:
            values = np.ascontiguousarray(self.vectors[bias], dtype=np.float32)
            memory.store(self.bias, 0, values.tobytes())
            use = 1
        self.session.launch(
            "vision_bias_cast",
            [self.gemm, self.bias, dest],
            [u32(rows), u32(cols), u32(cols), u32(use)],
            rows * cols,
        )

    def _layernorm(self, source: Any, weight: str, bias: str, dest: Any, rows: int) -> None:
        weight_f = np.ascontiguousarray(self.vectors[weight], np.float32)
        bias_f = np.ascontiguousarray(self.vectors[bias], np.float32)
        memory.store(self.vec_a, 0, weight_f.tobytes())
        memory.store(self.vec_b, 0, bias_f.tobytes())
        self.session.dispatch(
            "vision_ln",
            [source, self.vec_a, self.vec_b, dest],
            [u32(rows), u32(HIDDEN), f32(LN_EPS)],
            (rows, 1, 1),
            (256, 1, 1),
        )

    def _residual(self, residual: Any, branch: Any, gamma: str, dest: Any, rows: int) -> None:
        memory.store(self.vec_a, 0, np.ascontiguousarray(self.vectors[gamma], np.float32).tobytes())
        self.session.launch(
            "vision_residual",
            [residual, branch, self.vec_a, dest],
            [u32(rows), u32(HIDDEN)],
            rows * HIDDEN,
        )

    def _rope(self, states: Any, seq: int) -> None:
        self.session.launch(
            "vision_rope",
            [states, self.cos, self.sin],
            [u32(seq), u32(HEADS), u32(HEAD_DIM)],
            seq * HEADS * (HEAD_DIM // 2),
        )

    def _copy(self, source: Any, dest: Any, rows: int, cols: int) -> None:
        self.session.launch(
            "vision_pad",
            [source, dest],
            [u32(rows), u32(cols), u32(rows), u32(cols)],
            rows * cols,
        )

    def _open_scratch(self, seq: int, tokens: int) -> None:
        pad_rows = max(_align(seq), _align(tokens))
        n_patches = seq - 1
        self.weight = self._alloc(MID * MID * 2)
        self.act = self._alloc(pad_rows * MID * 2)
        self.gemm = self._alloc(pad_rows * MID * 4)
        self.bias = self._alloc(MID * 4)
        self.vec_a = self._alloc(MID * 4)
        self.vec_b = self._alloc(MID * 4)
        self.hidden = self._alloc(seq * HIDDEN * 2)
        self.branch = self._alloc(max(seq, tokens) * MID * 2)
        self.residual = self._alloc(seq * HIDDEN * 2)
        self.q = self._alloc(seq * HIDDEN * 2)
        self.k = self._alloc(seq * HIDDEN * 2)
        self.v = self._alloc(seq * HIDDEN * 2)
        self.mid = self._alloc(max(seq, tokens) * MID * 2)
        self.pixels = self._alloc(n_patches * PATCH * PATCH * 3 * 2)
        self.pos = self._alloc(n_patches * HIDDEN * 2)
        self.cos = self._alloc(seq * HEAD_DIM * 4)
        self.sin = self._alloc(seq * HEAD_DIM * 4)

    def _alloc(self, nbytes: int) -> Any:
        handle = memory.allocate(max(int(nbytes), 16), owner=self.owner)
        self._scratch.append(handle)
        return handle

    def _store(self, handle: Any, values: np.ndarray) -> None:
        memory.store(handle, 0, np.ascontiguousarray(values).tobytes())

    def _drop_scratch(self) -> None:
        for handle in self._scratch:
            memory.release(handle)
        self._scratch.clear()


def _vision_line(runtime: Any, kind: str, count: int) -> None:
    """Queue one vision line behind the prompt note. A direct write lands too early."""
    note = getattr(runtime, "_note", None)
    if callable(note):
        note("vision", kind=kind, count=count)
        return
    _status(runtime, f"vision {kind} {count}")


def _status(runtime: Any, text: str) -> None:
    note = getattr(runtime, "_status", None)
    if callable(note):
        note(text)
        return
    print(text, flush=True)


def _shift(plan: list[tuple[int, int, bytes]], dropped: int) -> list[tuple[int, int, bytes]]:
    shifted: list[tuple[int, int, bytes]] = []
    for start, count, blob in plan:
        if start + count <= dropped:
            continue
        if start < dropped:
            message = "an image does not fit the context window"
            raise ValueError(message)
        shifted.append((start - dropped, count, blob))
    return shifted


def _cached_rows(root: Path, blob: bytes, count: int) -> np.ndarray | None:
    loaded = load_cache(_cache_file(root, blob))
    if loaded is None:
        return None
    rows = loaded.get("rows")
    if rows is None or rows.shape != (count, TEXT) or rows.dtype != np.float16:
        return None
    return np.ascontiguousarray(rows)


def _cache_file(root: Path, blob: bytes) -> Path:
    digest = hashlib.sha256(blob)
    digest.update(config_digest(root).encode())
    return cache_path(PIPELINE, str(root), digest.digest(), 0, 0)


def _image_blobs(messages: list[dict[str, Any]]) -> list[bytes]:
    blobs: list[bytes] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("image", "input_image", "image_url"):
                continue
            blobs.append(_blob(part))
    return blobs


def _blob(part: dict[str, Any]) -> bytes:
    url = part.get("image_url")
    if url is None:
        url = part.get("url") if part.get("url") is not None else part.get("image")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        message = "an image part has no data"
        raise ValueError(message)
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            message = "an image data URL is not base64"
            raise ValueError(message)
        try:
            return base64.b64decode(payload, validate=True)
        except ValueError as exc:
            message = "an image data URL is not base64"
            raise ValueError(message) from exc
    if url.startswith("file://"):
        try:
            return Path(url[7:]).read_bytes()
        except OSError as exc:
            message = "the image file could not be read"
            raise ValueError(message) from exc
    message = "an image URL is not a data URL"
    raise ValueError(message)


def _pixels(rgb: np.ndarray, rows: int, cols: int) -> np.ndarray:
    height, width = rows * TOKEN, cols * TOKEN
    canvas = np.full((height, width, 3), -1.0, np.float32)
    canvas[: rgb.shape[0], : rgb.shape[1]] = rgb
    norm = (canvas / 255.0 - 0.5) / 0.5
    tiles = norm.reshape(rows, TOKEN, cols, TOKEN, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(tiles.reshape(rows * cols, TOKEN, TOKEN, 3))


def _patch_matrix(tokens: np.ndarray) -> np.ndarray:
    count = tokens.shape[0]
    shaped = tokens.reshape(count, MERGE, PATCH, MERGE, PATCH, 3)
    shaped = shaped.transpose(0, 1, 3, 2, 4, 5)
    flat = shaped.reshape(count * MERGE * MERGE, PATCH * PATCH * 3)
    return np.ascontiguousarray(flat, dtype=np.float16)


def _check_png(blob: bytes) -> None:
    if not blob.startswith(_PNG):
        message = "the image is not a PNG"
        raise ValueError(message)


def _ihdr(blob: bytes) -> tuple[int, int, int, int, int]:
    if len(blob) < 26 or blob[12:16] != b"IHDR":
        message = "the image is not a PNG"
        raise ValueError(message)
    width, height = struct.unpack(">II", blob[16:24])
    depth, color = blob[24], blob[25]
    interlace = blob[28] if len(blob) > 28 else 1
    return int(height), int(width), int(depth), int(color), int(interlace)


def _inflate(blob: bytes) -> bytes:
    parts: list[bytes] = []
    offset = 8
    while offset + 8 <= len(blob):
        length = struct.unpack(">I", blob[offset : offset + 4])[0]
        kind = blob[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(blob):
            break
        if kind == b"IDAT":
            parts.append(blob[start:end])
        elif kind == b"IEND":
            break
        offset = end + 4
    try:
        return zlib.decompress(b"".join(parts))
    except zlib.error as exc:
        message = "the image is not a PNG"
        raise ValueError(message) from exc


def _unfilter(raw: bytes, height: int, width: int, channels: int) -> np.ndarray:
    stride = width * channels
    src = np.frombuffer(raw, np.uint8)
    out = np.zeros((height, stride), np.int16)
    offset = 0
    for y in range(height):
        kind = int(src[offset])
        offset += 1
        for x in range(stride):
            value = int(src[offset + x])
            left = int(out[y, x - channels]) if x >= channels else 0
            up = int(out[y - 1, x]) if y else 0
            upper_left = int(out[y - 1, x - channels]) if y and x >= channels else 0
            if kind == 1:
                value += left
            elif kind == 2:
                value += up
            elif kind == 3:
                value += (left + up) // 2
            elif kind == 4:
                value += _paeth(left, up, upper_left)
            elif kind != 0:
                message = "the PNG filter is not supported"
                raise ValueError(message)
            out[y, x] = value & 255
        offset += stride
    return out.astype(np.uint8)


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


def _composite(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    color = rgba[:, :, :3].astype(np.int32)
    alpha = rgba[:, :, 3:4].astype(np.int32)
    rows = np.arange(height)[:, None] // 8
    columns = np.arange(width)[None, :] // 8
    background = np.where((rows + columns) % 2 == 0, 204, 153).astype(np.int32)
    mixed = (color * alpha + background[:, :, None] * (255 - alpha) + 127) // 255
    return np.ascontiguousarray(mixed.astype(np.uint8))


def _align(count: int) -> int:
    return (int(count) + 63) // 64 * 64


def _scale_array(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    rows, width = shape
    return np.frombuffer(raw, np.float32).reshape(rows, width // 4).copy()


def _vector_names() -> list[str]:
    names = ["class", "cls_pos", "ln_pre.w", "ln_pre.b"]
    for index in range(LAYERS):
        prefix = f"l{index}"
        names.extend(
            [
                f"{prefix}.ln1.w",
                f"{prefix}.ln1.b",
                f"{prefix}.ln2.w",
                f"{prefix}.ln2.b",
                f"{prefix}.ls1",
                f"{prefix}.ls2",
                f"{prefix}.q.b",
                f"{prefix}.k.b",
                f"{prefix}.v.b",
                f"{prefix}.o.b",
                f"{prefix}.fc.b",
                f"{prefix}.proj.b",
            ]
        )
    return names


def _pack_ready(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    tensors = Tensors(path)
    try:
        return tensors.metadata.get("format") == PACK_FORMAT and tensors.metadata.get(
            "config"
        ) == config_digest(root)
    finally:
        tensors.close()


def _write_pack(root: Path, path: Path, log: Any = None) -> None:
    source = root / "vision.safetensors"
    if not source.is_file():
        source = root / "model-00001-of-00001.safetensors"
    tensors = Tensors(source)
    packed: dict[str, np.ndarray] = {}
    try:
        for name, checkpoint, shape in _vector_catalog():
            values = tensors.vector(checkpoint, shape)
            blob = np.ascontiguousarray(values, np.float16).tobytes()
            packed[name] = np.frombuffer(blob, np.uint8).copy()
        for name, checkpoint, shape in _matrix_catalog():
            if log is not None and name.startswith("l") and name.endswith(".q"):
                log(name.removesuffix(".q"))
            codes, scales = quantize_fp8(tensors.vector(checkpoint, shape))
            packed[f"{name}.code"] = codes
            raw = np.ascontiguousarray(scales, np.float32).tobytes()
            packed[f"{name}.scale"] = (
                np.frombuffer(raw, np.uint8).reshape(scales.shape[0], scales.shape[1] * 4).copy()
            )
    finally:
        tensors.close()
    write_pack(
        path,
        packed,
        {"format": PACK_FORMAT, "config": config_digest(root)},
    )


def _vector_catalog() -> list[tuple[str, str, tuple[int, ...]]]:
    back = "model.visual.backbone"
    rows: list[tuple[str, str, tuple[int, ...]]] = [
        ("class", f"{back}.embeddings.class_embedding", (HIDDEN,)),
        ("cls_pos", f"{back}.embeddings.cls_position_embedding", (HIDDEN,)),
        ("ln_pre.w", f"{back}.ln_pre.weight", (HIDDEN,)),
        ("ln_pre.b", f"{back}.ln_pre.bias", (HIDDEN,)),
    ]
    for index in range(LAYERS):
        layer = f"{back}.layers.{index}"
        prefix = f"l{index}"
        rows.extend(
            [
                (f"{prefix}.ln1.w", f"{layer}.ln_1.weight", (HIDDEN,)),
                (f"{prefix}.ln1.b", f"{layer}.ln_1.bias", (HIDDEN,)),
                (f"{prefix}.ln2.w", f"{layer}.ln_2.weight", (HIDDEN,)),
                (f"{prefix}.ln2.b", f"{layer}.ln_2.bias", (HIDDEN,)),
                (f"{prefix}.ls1", f"{layer}.ls_1.gamma", (HIDDEN,)),
                (f"{prefix}.ls2", f"{layer}.ls_2.gamma", (HIDDEN,)),
                (f"{prefix}.q.b", f"{layer}.self_attn.q_proj.bias", (HIDDEN,)),
                (f"{prefix}.k.b", f"{layer}.self_attn.k_proj.bias", (HIDDEN,)),
                (f"{prefix}.v.b", f"{layer}.self_attn.v_proj.bias", (HIDDEN,)),
                (f"{prefix}.o.b", f"{layer}.self_attn.out_proj.bias", (HIDDEN,)),
                (f"{prefix}.fc.b", f"{layer}.mlp.c_fc.bias", (MID,)),
                (f"{prefix}.proj.b", f"{layer}.mlp.c_proj.bias", (HIDDEN,)),
            ]
        )
    return rows


def _matrix_catalog() -> list[tuple[str, str, tuple[int, int]]]:
    back = "model.visual.backbone"
    rows: list[tuple[str, str, tuple[int, int]]] = [
        ("patch", f"{back}.embeddings.patch_embedding.weight", (HIDDEN, PATCH * PATCH * 3)),
        ("pos", f"{back}.embeddings.position_embedding", (GRID * GRID, HIDDEN)),
        ("proj_in", "model.visual.proj_in.weight", (MID, MID)),
        ("proj_out", "model.visual.proj_out.weight", (TEXT, MID)),
    ]
    for index in range(LAYERS):
        layer = f"{back}.layers.{index}"
        prefix = f"l{index}"
        rows.extend(
            [
                (f"{prefix}.q", f"{layer}.self_attn.q_proj.weight", (HIDDEN, HIDDEN)),
                (f"{prefix}.k", f"{layer}.self_attn.k_proj.weight", (HIDDEN, HIDDEN)),
                (f"{prefix}.v", f"{layer}.self_attn.v_proj.weight", (HIDDEN, HIDDEN)),
                (f"{prefix}.o", f"{layer}.self_attn.out_proj.weight", (HIDDEN, HIDDEN)),
                (f"{prefix}.fc", f"{layer}.mlp.c_fc.weight", (MID, HIDDEN)),
                (f"{prefix}.proj", f"{layer}.mlp.c_proj.weight", (HIDDEN, MID)),
            ]
        )
    return rows
