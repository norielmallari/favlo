"""Resident record and Metal generation for Qwen3.5-4B."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from favlo.memory import memory
from qwen35_4b import progress
from qwen35_4b.closed import RequestClosed
from qwen35_4b.layout import (
    ATTN_HEADS,
    ATTN_SCALE,
    BLOCK,
    CONTEXT,
    CONV_KERNEL,
    FULL,
    GDN_STATE_BYTES,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HEADS,
    LAYERS,
    LINEAR_HEAD_DIM,
    LINEAR_KEY_HEADS,
    LINEAR_VALUE_HEADS,
    RMS_EPS,
    ROPE_DIM,
    STEM,
    THETA,
    VOCAB,
    cache_bytes,
    is_full,
    kv_mode,
)
from qwen35_4b.session import GEMV_ROWS, PREFILL_TILE, Session, f32, u32

_PIECE = 8 << 20
_CONV_CH = LINEAR_KEY_HEADS * LINEAR_HEAD_DIM * 2 + LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
_GDN_Z = LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
_MIN_CTX = 2048
_GROUP = ATTN_HEADS // KV_HEADS
_TILE = PREFILL_TILE
# Dense prefill planes that run int8 x int8. Attention q/k/v/o and GDN a/b stay on
# the half x fp8 matmul. Decode never reads these paths.
_I8_PLANES = frozenset({"gate", "up", "down", "qkv", "z", "out"})
# (cols, inner) of each int8 plane. The FFN planes exist on every layer; qkv, z, and out
# only on the GDN layers.
_I8_SHAPES = {
    "gate": (INTERMEDIATE, HIDDEN),
    "up": (INTERMEDIATE, HIDDEN),
    "down": (HIDDEN, INTERMEDIATE),
    "qkv": (_CONV_CH, HIDDEN),
    "z": (_GDN_Z, HIDDEN),
    "out": (HIDDEN, _GDN_Z),
}
_I8_FFN = frozenset({"gate", "up", "down"})
_I8_WEIGHT_BYTES = max(INTERMEDIATE * HIDDEN, _CONV_CH * HIDDEN, _GDN_Z * HIDDEN)
_I8_INNER = max(HIDDEN, INTERMEDIATE, _GDN_Z)
_I8_COLS = max(HIDDEN, INTERMEDIATE, _CONV_CH, _GDN_Z)


@dataclass(frozen=True)
class Plane:
    data: int
    scale: int


class Qwen35Runtime:
    """Packed Qwen3.5-4B record. Prefill and decode run on Metal."""

    def __init__(
        self,
        root: Path,
        *,
        kv_cache: str = "half",
        console: Any = None,
        owner: str | None = None,
        mtp: bool = False,
    ) -> None:
        del mtp
        try:
            chosen = kv_mode(kv_cache)
        except ValueError as exc:
            message = f"qwen35-4b cache is half or fp8_e4m3; {kv_cache} is not built"
            raise RuntimeError(message) from exc
        self.root = Path(root)
        self.console = console
        self.kv_cache = chosen
        self._kv_elem = 2 if chosen == "half" else 1
        self._owner = owner or f"{STEM}:{id(self)}"
        self._stop: Callable[[], bool] | None = None
        self._halted = False
        self._request_id: int | None = None
        self._load_progress: Any = None
        self._prefill_progress: Any = None
        self._cached_tokens = 0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._output_ids: list[int] = []
        self.max_memory = memory.budget()
        self.max_pos = CONTEXT
        self.pos = 0
        self._rope_delta = 0
        self._rope_device = None
        self._rope_length = 0
        self._vision_rows = None
        self._vision_index = None
        self._presence = 0.0
        self._frequency = 0.0
        manifest = self.root / "manifest.json"
        weights = self.root / "weights.bin"
        if not manifest.is_file() or not weights.is_file():
            message = f"missing packed record at {self.root}"
            raise FileNotFoundError(message)
        body = json.loads(manifest.read_text(encoding="utf-8"))
        if body.get("complete") is not True:
            message = f"incomplete packed record at {self.root}"
            raise RuntimeError(message)
        self._tensors = body["tensors"]
        size = weights.stat().st_size
        detail = getattr(console, "load_detail", "") or self.root.name
        self._load_progress = progress.begin_load(console, size, detail)
        summary = ""
        try:
            self.session = Session(self._owner)
            self._open(weights, size)
            summary = (
                f"loaded {self.root.name} kv {self.kv_cache} "
                f"context {self.max_pos} memory {self.max_memory}"
            )
        except Exception:
            if getattr(self, "session", None) is not None:
                self.session.close()
            raise
        finally:
            row = self._load_progress
            self._load_progress = None
            progress.end_phase(console, row, summary)

    def refresh_budget(self) -> None:
        """Re-read the serve ledger. A larger answer is kept for the next size."""
        ceiling = memory.budget()
        if ceiling > self.max_memory:
            self.max_memory = ceiling

    def _open(self, weights: Path, size: int) -> None:
        self._planes = {
            name: Plane(data=int(spec["data"]), scale=int(spec["scale"]))
            for name, spec in self._tensors.items()
        }
        gdn = GDN_STATE_BYTES
        conv = (LAYERS - len(FULL)) * _CONV_CH * (CONV_KERNEL - 1) * 4
        self.refresh_budget()
        budget = self.max_memory
        capacity = CONTEXT
        while capacity >= _MIN_CTX:
            need = (
                size
                + _workspace_bytes(capacity)
                + gdn
                + conv
                + cache_bytes(capacity, self.kv_cache)
            )
            if need <= budget:
                break
            capacity = max(_MIN_CTX, capacity // 2)
        need = (
            size + _workspace_bytes(capacity) + gdn + conv + cache_bytes(capacity, self.kv_cache)
        )
        if need > budget:
            message = (
                f"qwen35-4b needs {need} bytes for a {_MIN_CTX}-token cache; "
                f"budget is {budget}"
            )
            raise RuntimeError(message)
        self.max_pos = capacity
        # Restored int8 codes and channel scales stay resident when they fit beside the
        # rest of the opening. Otherwise every int8 matmul restores into the stage.
        resident = _i8_resident_bytes()
        self._i8_fits = need + resident <= budget

        self.weights = self.session.allocate(size)
        view = memory.map(self.weights, 0, size)
        with weights.open("rb") as src:
            offset = 0
            while offset < size:
                piece = min(_PIECE, size - offset)
                read = src.readinto(view[offset : offset + piece])
                if not read:
                    message = f"{weights.name} stopped at {offset} of {size}"
                    raise OSError(message)
                offset += read
                progress.note_load(self._load_progress, read)

        s = self.session
        self.hidden = s.allocate(HIDDEN * 4, zero=True)
        self.normed = s.allocate(HIDDEN * 4, zero=True)
        self.delta = s.allocate(HIDDEN * 4, zero=True)
        self.gate = s.allocate(INTERMEDIATE * 4, zero=True)
        self.up = s.allocate(INTERMEDIATE * 4, zero=True)
        self.ff = s.allocate(INTERMEDIATE * 4, zero=True)
        self.mixed = s.allocate(_CONV_CH * 4, zero=True)
        self.z_buf = s.allocate(LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * 4, zero=True)
        self.a_buf = s.allocate(LINEAR_VALUE_HEADS * 4, zero=True)
        self.b_buf = s.allocate(LINEAR_VALUE_HEADS * 4, zero=True)
        self.gdn_out = s.allocate(LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * 4, zero=True)
        self.q_fuse = s.allocate(ATTN_HEADS * HEAD_DIM * 2 * 4, zero=True)
        self.k_buf = s.allocate(KV_HEADS * HEAD_DIM * 4, zero=True)
        self.v_buf = s.allocate(KV_HEADS * HEAD_DIM * 4, zero=True)
        self.attn = s.allocate(ATTN_HEADS * HEAD_DIM * 4, zero=True)
        self.scores = s.allocate(ATTN_HEADS * capacity * 4, zero=True)
        self.logits = s.allocate(VOCAB * 4, zero=True)
        groups = (VOCAB + 255) // 256
        self.argmax_v = s.allocate(groups * 4, zero=True)
        self.argmax_i = s.allocate(groups * 4, zero=True)
        self.token_buf = s.allocate(4, zero=True)
        self.token_out = s.allocate(4, zero=True)
        self.token_counts = s.allocate(VOCAB * 4, zero=True)
        self.gdn_state = s.allocate(gdn, zero=True)
        self.conv_hist = s.allocate(conv, zero=True)
        plane = len(FULL) * capacity * KV_HEADS * HEAD_DIM * self._kv_elem
        self.k_data = s.allocate(plane, zero=True)
        self.v_data = s.allocate(plane, zero=True)
        if self.kv_cache == "fp8_e4m3":
            scales = len(FULL) * capacity * KV_HEADS * (HEAD_DIM // BLOCK)
            self.k_scale = s.allocate(scales, zero=True)
            self.v_scale = s.allocate(scales, zero=True)
        else:
            self.k_scale = 0
            self.v_scale = 0
        tile = _TILE
        widest = max(
            HIDDEN,
            INTERMEDIATE,
            _CONV_CH,
            _GDN_Z,
            ATTN_HEADS * HEAD_DIM * 2,
            ATTN_HEADS * HEAD_DIM,
            KV_HEADS * HEAD_DIM,
            LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM,
        )
        self.pf_tokens = s.allocate(tile * 4, zero=True)
        self.pf_hidden = s.allocate(tile * HIDDEN * 4, zero=True)
        self.pf_normed = s.allocate(tile * HIDDEN * 4, zero=True)
        self.pf_half = s.allocate(tile * widest * 2, zero=True)
        self.pf_delta = s.allocate(tile * HIDDEN * 4, zero=True)
        self.pf_gate = s.allocate(tile * INTERMEDIATE * 4, zero=True)
        self.pf_up = s.allocate(tile * INTERMEDIATE * 4, zero=True)
        self.pf_ff = s.allocate(tile * INTERMEDIATE * 4, zero=True)
        self.pf_mixed = s.allocate(tile * _CONV_CH * 4, zero=True)
        self.pf_z = s.allocate(tile * _GDN_Z * 4, zero=True)
        self.pf_a = s.allocate(tile * LINEAR_VALUE_HEADS * 4, zero=True)
        self.pf_b = s.allocate(tile * LINEAR_VALUE_HEADS * 4, zero=True)
        self.pf_gdn = s.allocate(tile * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * 4, zero=True)
        self.pf_q = s.allocate(tile * ATTN_HEADS * HEAD_DIM * 2 * 4, zero=True)
        self.pf_k = s.allocate(tile * KV_HEADS * HEAD_DIM * 4, zero=True)
        self.pf_v = s.allocate(tile * KV_HEADS * HEAD_DIM * 4, zero=True)
        self.pf_attn = s.allocate(tile * ATTN_HEADS * HEAD_DIM * 4, zero=True)
        # Score scratch for batched prefill attention. Long contexts fall back
        # to the single-query kernel when the live span exceeds this span.
        self._score_span = min(capacity, 8192)
        self.pf_scores = s.allocate(tile * ATTN_HEADS * self._score_span * 4, zero=True)
        # int8 prefill stages: weight codes, weight channel scales, activation codes,
        # activation row scales.
        self.i8_weight = s.allocate(_I8_WEIGHT_BYTES)
        self.i8_weight_scale = s.allocate(_I8_COLS * 4, zero=True)
        self.i8_act = s.allocate(tile * _I8_INNER, zero=True)
        self.i8_act_scale = s.allocate(tile * 4, zero=True)
        self._i8_cache: dict[str, tuple[int, int, int, int]] = {}
        self._i8_ready: set[str] = set()
        self.i8_resident: Any = None
        self.i8_resident_scale: Any = None
        if self._i8_fits:
            data_cursor = 0
            scale_cursor = 0
            for layer in range(LAYERS):
                for name, (cols, inner) in _I8_SHAPES.items():
                    if name not in _I8_FFN and is_full(layer):
                        continue
                    self._i8_cache[f"layer.{layer}.{name}"] = (
                        data_cursor,
                        scale_cursor,
                        cols,
                        inner,
                    )
                    data_cursor += cols * inner
                    scale_cursor += cols * 4
            self.i8_resident = s.allocate(data_cursor)
            self.i8_resident_scale = s.allocate(scale_cursor, zero=True)
        self._full_index = {layer: i for i, layer in enumerate(FULL)}
        self._gdn_index: dict[int, int] = {}
        cursor = 0
        for layer in range(LAYERS):
            if not is_full(layer):
                self._gdn_index[layer] = cursor
                cursor += 1

    def close(self) -> None:
        try:
            if getattr(self, "session", None) is not None:
                self.session.close()
            memory.sweep(self._owner)
        except Exception:
            return

    def halt(self) -> None:
        self._halted = True

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        stop: set[int],
        *,
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
        on_token: Callable[[int], None] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence: float = 0.0,
        frequency: float = 0.0,
        think_close: list[int] | None = None,
        think_end: int | None = None,
        thinking_budget: int | None = None,
        mtp: bool = False,
        mtp_depth: int | None = None,
    ) -> list[int]:
        del (
            forced,
            stop_after_forced,
            top_p,
            top_k,
            think_close,
            think_end,
            thinking_budget,
            mtp,
            mtp_depth,
            temperature,
        )
        if self._halted or (self._stop is not None and self._stop()):
            raise RequestClosed()
        if not prompt:
            message = "prompt is empty"
            raise ValueError(message)
        if len(prompt) > self.max_pos:
            message = f"prompt length {len(prompt)} exceeds context {self.max_pos}"
            raise ValueError(message)

        self._presence = float(presence)
        self._frequency = float(frequency)
        self.refresh_budget()
        self._reset_state()
        self._output_ids = []
        self._cached_tokens = 0
        console = self.console
        progress.note(console, "prompt", tokens=len(prompt))
        progress.note(console, "cache", hit=0)
        uncached = len(prompt)
        self._prefill_progress = progress.begin_phase(
            console, "prefill", uncached, request=self._request_id
        )
        prefill_start = time.perf_counter()
        try:
            done = 0
            while done < uncached:
                self._check()
                piece = prompt[done : done + _TILE]
                done += len(piece)
                self._prefill_chunk(
                    [int(token) for token in piece],
                    logits=done >= uncached,
                )
                progress.note_prefill(self._prefill_progress, done)
            self.session.wait()
            next_id = self._sample()
        finally:
            self._prefill_seconds = time.perf_counter() - prefill_start
            row = self._prefill_progress
            self._prefill_progress = None
            progress.end_phase(
                console, row, progress.rate("prefill", uncached, self._prefill_seconds)
            )

        decode_row = progress.begin_phase(
            console, "decode", 0, request=self._request_id, live=True
        )
        decode_start = time.perf_counter()
        produced = 0
        try:
            while produced < max_tokens:
                self._check()
                if next_id in stop:
                    break
                self._output_ids.append(next_id)
                produced += 1
                progress.note_decode(decode_row, produced)
                if on_token is not None:
                    on_token(next_id)
                if self.pos >= self.max_pos:
                    break
                self._step(next_id)
                next_id = self._sample()
        finally:
            self._decode_seconds = time.perf_counter() - decode_start
            progress.end_phase(
                console, decode_row, progress.rate("decode", produced, self._decode_seconds)
            )
        return list(self._output_ids)

    def _check(self) -> None:
        if self._halted or (self._stop is not None and self._stop()):
            raise RequestClosed()

    def _pad(self, count: int) -> int:
        # Matmul2d needs a multiple of 64; the chunk may be smaller than PREFILL_TILE.
        return max(64, ((count + 63) // 64) * 64)

    def _cast_half(self, source: Any, width: int, rows: int, *, valid: int | None = None) -> None:
        kept = rows if valid is None else valid
        if kept < rows:
            words = (rows * width * 2) // 4
            self.session.launch(
                "zero_u32",
                [self.pf_half],
                [u32(words)],
                ((words + 255) // 256, 1, 1),
                (256, 1, 1),
            )
        self.session.launch(
            "cast_f32_half",
            [source, self.pf_half],
            [u32(kept * width)],
            ((kept * width + 255) // 256, 1, 1),
            (256, 1, 1),
        )

    def _matmul(self, plane: Plane, destination: Any, rows: int, cols: int, inner: int) -> None:
        """``pf_half`` holds the left activation [rows, inner] as float16."""
        row_tile = 64
        if rows % row_tile != 0:
            message = f"matmul tile requires rows multiple of {row_tile}; got {rows}x{cols}"
            raise ValueError(message)
        if cols % row_tile == 0:
            self.session.launch(
                "matmul_fp8",
                [self.pf_half, self.weights, self.weights, destination],
                [u32(rows), u32(cols), u32(inner)],
                (cols // row_tile, rows // row_tile, 1),
                (128, 1, 1),
                offsets=[0, plane.data, plane.scale, 0],
            )
            return
        if cols % 32 != 0:
            message = f"matmul tile requires cols multiple of 32 or {row_tile}; got {rows}x{cols}"
            raise ValueError(message)
        self.session.launch(
            "matmul_fp8_n32",
            [self.pf_half, self.weights, self.weights, destination],
            [u32(rows), u32(cols), u32(inner)],
            (cols // 32, rows // row_tile, 1),
            (128, 1, 1),
            offsets=[0, plane.data, plane.scale, 0],
        )

    def _dense(
        self,
        layer: int,
        name: str,
        destination: Any,
        rows: int,
        cols: int,
        inner: int,
        *,
        fresh: bool = True,
    ) -> None:
        """Prefill projection ``layer.<n>.<name>``: int8 for the planes in ``_I8_PLANES``.

        ``fresh`` is False when ``pf_half`` was already quantized for the previous call
        (up after gate, z after qkv).
        """
        plane = self._planes[f"layer.{layer}.{name}"]
        if name in _I8_PLANES:
            self._matmul_i8(
                plane,
                f"layer.{layer}.{name}",
                destination,
                rows,
                cols,
                inner,
                fresh=fresh,
                wide=name == "down",
            )
            return
        self._matmul(plane, destination, rows, cols, inner)

    def _matmul_i8(
        self,
        plane: Plane,
        key: str,
        destination: Any,
        rows: int,
        cols: int,
        inner: int,
        *,
        fresh: bool = True,
        wide: bool = False,
    ) -> None:
        """``pf_half`` [rows, inner] times the fp8 plane through int8 matmul2d.

        The fp8 plane is restored to int8 codes with one channel scale per output
        column, the activation is quantized per row, and ``scale_i8_f32`` rewrites the
        int32 accumulators in ``destination`` as floats. When the resident atlas fitted,
        the restore runs once per ``key`` into its offsets and later calls only bind them.
        """
        if rows % 64 != 0 or cols % 64 != 0 or inner % 128 != 0:
            message = f"int8 matmul needs rows, cols multiples of 64 and inner of 128; got {rows}x{cols}x{inner}"
            raise ValueError(message)
        if (
            cols * inner > _I8_WEIGHT_BYTES
            or inner > _I8_INNER
            or cols > _I8_COLS
            or rows > _TILE
        ):
            message = f"int8 matmul {rows}x{cols}x{inner} exceeds its stage"
            raise ValueError(message)
        session = self.session
        entry = self._i8_cache.get(key) if self.i8_resident is not None else None
        if entry is not None:
            data_off, scale_off, entry_cols, entry_inner = entry
            if entry_cols != cols or entry_inner != inner:
                message = f"int8 plane {key} is {entry_cols}x{entry_inner}, not {cols}x{inner}"
                raise ValueError(message)
            weight, weight_scale = self.i8_resident, self.i8_resident_scale
            if key not in self._i8_ready:
                session.launch(
                    "restore_fp8_i8",
                    [self.weights, self.weights, weight, weight_scale],
                    [u32(inner)],
                    (cols, 1, 1),
                    (256, 1, 1),
                    offsets=[plane.data, plane.scale, data_off, scale_off],
                )
                self._i8_ready.add(key)
        else:
            data_off = scale_off = 0
            weight, weight_scale = self.i8_weight, self.i8_weight_scale
            session.launch(
                "restore_fp8_i8",
                [self.weights, self.weights, weight, weight_scale],
                [u32(inner)],
                (cols, 1, 1),
                (256, 1, 1),
                offsets=[plane.data, plane.scale, 0, 0],
            )
        if fresh:
            session.launch(
                "quantize_half_i8",
                [self.pf_half, self.i8_act, self.i8_act_scale],
                [u32(inner)],
                (rows, 1, 1),
                (256, 1, 1),
            )
        if wide and cols % 128 == 0 and rows % 32 == 0:
            session.launch(
                "prefill_i8_m32",
                [self.i8_act, weight, destination],
                [u32(rows), u32(cols), u32(inner)],
                (cols // 128, rows // 32, 1),
                (128, 1, 1),
                offsets=[0, data_off, 0],
            )
        else:
            session.launch(
                "prefill_i8_n64",
                [self.i8_act, weight, destination],
                [u32(rows), u32(cols), u32(inner)],
                (cols // 64, rows // 64, 1),
                (128, 1, 1),
                offsets=[0, data_off, 0],
            )
        quads = rows * (cols // 4)
        session.launch(
            "scale_i8_f32",
            [destination, self.i8_act_scale, weight_scale],
            [u32(rows), u32(cols)],
            ((quads + 255) // 256, 1, 1),
            (256, 1, 1),
            offsets=[0, 0, scale_off],
        )

    def _prefill_chunk(self, tokens: list[int], *, logits: bool) -> None:
        count = len(tokens)
        if count == 0:
            return
        if count > _TILE:
            message = f"prefill chunk {count} exceeds tile {_TILE}"
            raise ValueError(message)
        position0 = self.pos
        row_pad = self._pad(count)
        packed = b"".join(struct.pack("<I", token) for token in tokens)
        if len(tokens) < _TILE:
            packed += b"\x00" * (4 * (_TILE - len(tokens)))
        session = self.session
        # Prior chunk may still be reading pf_tokens / workspace.
        session.wait()
        memory.store(self.pf_tokens, 0, packed)
        session.begin()
        if count < row_pad:
            # Clear padded activation rows so matmul tiles stay zero.
            session.launch(
                "zero_f32",
                [self.pf_hidden],
                [u32(row_pad * HIDDEN)],
                ((row_pad * HIDDEN + 255) // 256, 1, 1),
                (256, 1, 1),
            )
        session.launch(
            "embed_rows",
            [self.weights, self.weights, self.pf_tokens, self.pf_hidden],
            [u32(count), u32(HIDDEN)],
            (HIDDEN, count, 1),
            (1, 1, 1),
            offsets=[self._planes["embedding"].data, self._planes["embedding"].scale, 0, 0],
        )
        self._splice_vision(count, position0)
        for layer in range(LAYERS):
            self._prefill_layer(layer, count, row_pad, position0)
        if logits:
            last = count - 1
            session.launch(
                "copy_row_f32",
                [self.pf_hidden, self.hidden],
                [u32(HIDDEN), u32(last)],
                ((HIDDEN + 255) // 256, 1, 1),
                (256, 1, 1),
            )
            self._rmsnorm(self.hidden, self._planes["final_norm"], self.normed)
            self._gemv(self.normed, self._planes["embedding"], self.logits, VOCAB, HIDDEN)
        session.end(wait=False)
        self.pos = position0 + count

    def _splice_vision(self, count: int, position0: int) -> None:
        rows = self._vision_rows
        index = self._vision_index
        if rows is None or index is None or count < 1:
            return
        self.session.launch(
            "splice_rows",
            [rows, index, self.pf_hidden],
            [u32(position0), u32(count), u32(HIDDEN)],
            ((HIDDEN + 255) // 256, count, 1),
            (256, 1, 1),
        )

    def _rope_rows(self, data: Any, heads: int, stride: int, count: int, position0: int) -> None:
        table = self._rope_device
        length = int(self._rope_length)
        if table is not None and length > 0:
            self.session.launch(
                "rope_mrope_rows",
                [data, table],
                [
                    u32(heads),
                    u32(stride),
                    u32(ROPE_DIM),
                    u32(position0),
                    u32(count),
                    u32(length),
                    f32(THETA),
                ],
                (heads * (ROPE_DIM // 2), count, 1),
                (1, 1, 1),
            )
            return
        self.session.launch(
            "rope_heads_rows",
            [data],
            [u32(heads), u32(stride), u32(ROPE_DIM), u32(position0), u32(count), f32(THETA)],
            (heads * (ROPE_DIM // 2), count, 1),
            (1, 1, 1),
        )

    def _prefill_layer(self, layer: int, count: int, row_pad: int, position0: int) -> None:
        session = self.session
        session.launch(
            "rmsnorm_rows",
            [self.pf_hidden, self.weights, self.pf_normed],
            [u32(HIDDEN), u32(count), f32(RMS_EPS), f32(1.0)],
            (count, 1, 1),
            (256, 1, 1),
            offsets=[0, self._planes[f"layer.{layer}.in_norm"].data, 0],
        )
        if is_full(layer):
            self._prefill_full(layer, count, row_pad, position0)
        else:
            self._prefill_gdn(layer, count, row_pad)
        session.launch(
            "residual_add_rows",
            [self.pf_hidden, self.pf_delta],
            [u32(HIDDEN), u32(count)],
            (HIDDEN, count, 1),
            (1, 1, 1),
        )
        session.launch(
            "rmsnorm_rows",
            [self.pf_hidden, self.weights, self.pf_normed],
            [u32(HIDDEN), u32(count), f32(RMS_EPS), f32(1.0)],
            (count, 1, 1),
            (256, 1, 1),
            offsets=[0, self._planes[f"layer.{layer}.post_norm"].data, 0],
        )
        self._cast_half(self.pf_normed, HIDDEN, row_pad, valid=count)
        self._dense(layer, "gate", self.pf_gate, row_pad, INTERMEDIATE, HIDDEN)
        self._dense(layer, "up", self.pf_up, row_pad, INTERMEDIATE, HIDDEN, fresh=False)
        session.launch(
            "silu_mul_rows",
            [self.pf_gate, self.pf_up, self.pf_ff],
            [u32(INTERMEDIATE), u32(count)],
            (INTERMEDIATE, count, 1),
            (1, 1, 1),
        )
        self._cast_half(self.pf_ff, INTERMEDIATE, row_pad, valid=count)
        self._dense(layer, "down", self.pf_delta, row_pad, HIDDEN, INTERMEDIATE)
        session.launch(
            "residual_add_rows",
            [self.pf_hidden, self.pf_delta],
            [u32(HIDDEN), u32(count)],
            (HIDDEN, count, 1),
            (1, 1, 1),
        )

    def _prefill_gdn(self, layer: int, count: int, row_pad: int) -> None:
        session = self.session
        gdn_i = self._gdn_index[layer]
        state_off = gdn_i * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * 4
        hist_off = gdn_i * _CONV_CH * (CONV_KERNEL - 1) * 4
        self._cast_half(self.pf_normed, HIDDEN, row_pad, valid=count)
        self._dense(layer, "qkv", self.pf_mixed, row_pad, _CONV_CH, HIDDEN)
        self._dense(layer, "z", self.pf_z, row_pad, _GDN_Z, HIDDEN, fresh=False)
        # a/b are 32 cols — matmul2d N=32 tile over the padded chunk.
        self._matmul(
            self._planes[f"layer.{layer}.a"],
            self.pf_a,
            row_pad,
            LINEAR_VALUE_HEADS,
            HIDDEN,
        )
        self._matmul(
            self._planes[f"layer.{layer}.b"],
            self.pf_b,
            row_pad,
            LINEAR_VALUE_HEADS,
            HIDDEN,
        )
        session.launch(
            "conv_silu_chunk",
            [self.weights, self.pf_mixed, self.conv_hist],
            [u32(_CONV_CH), u32(count)],
            (_CONV_CH, 1, 1),
            (1, 1, 1),
            offsets=[self._planes[f"layer.{layer}.conv"].data, 0, hist_off],
        )
        session.launch(
            "gdn_chunk",
            [
                self.gdn_state,
                self.pf_mixed,
                self.pf_a,
                self.pf_b,
                self.weights,
                self.weights,
                self.pf_gdn,
            ],
            [u32(LINEAR_KEY_HEADS), u32(LINEAR_VALUE_HEADS), u32(count)],
            (LINEAR_VALUE_HEADS, 1, 1),
            (LINEAR_HEAD_DIM, 1, 1),
            offsets=[
                state_off,
                0,
                0,
                0,
                self._planes[f"layer.{layer}.A_log"].data,
                self._planes[f"layer.{layer}.dt_bias"].data,
                0,
            ],
        )
        session.launch(
            "gated_rmsnorm_rows",
            [self.pf_gdn, self.pf_z, self.weights],
            [u32(count), u32(LINEAR_VALUE_HEADS), f32(RMS_EPS)],
            (LINEAR_VALUE_HEADS, count, 1),
            (LINEAR_HEAD_DIM, 1, 1),
            offsets=[0, 0, self._planes[f"layer.{layer}.gdn_norm"].data],
        )
        self._cast_half(
            self.pf_gdn, LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM, row_pad, valid=count
        )
        self._dense(
            layer,
            "out",
            self.pf_delta,
            row_pad,
            HIDDEN,
            LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM,
        )

    def _kv_offsets(self, layer: int) -> tuple[int, int]:
        slot = self._full_index[layer]
        data = slot * self.max_pos * KV_HEADS * HEAD_DIM * self._kv_elem
        scale = slot * self.max_pos * KV_HEADS * (HEAD_DIM // BLOCK)
        return data, scale

    def _store_kv(
        self,
        key: int,
        value: int,
        *,
        rows: bool,
        position: int,
        count: int,
        plane: int,
        data_off: int,
        scale_off: int,
    ) -> None:
        session = self.session
        if self.kv_cache == "half":
            if not rows:
                session.launch(
                    "store_kv_half",
                    [key, value, self.k_data, self.v_data],
                    [u32(position), u32(KV_HEADS), u32(plane)],
                    (KV_HEADS, 1, 1),
                    (HEAD_DIM, 1, 1),
                    offsets=[0, 0, data_off, data_off],
                )
                return
            session.launch(
                "store_kv_rows_half",
                [key, value, self.k_data, self.v_data],
                [u32(position), u32(KV_HEADS), u32(count), u32(plane)],
                (KV_HEADS, count, 1),
                (HEAD_DIM, 1, 1),
                offsets=[0, 0, data_off, data_off],
            )
            return
        if not rows:
            session.launch(
                "store_kv",
                [key, value, self.k_data, self.k_scale, self.v_data, self.v_scale],
                [u32(position), u32(KV_HEADS), u32(plane)],
                (KV_HEADS, 1, 1),
                (HEAD_DIM, 1, 1),
                offsets=[0, 0, data_off, scale_off, data_off, scale_off],
            )
            return
        session.launch(
            "store_kv_rows",
            [key, value, self.k_data, self.k_scale, self.v_data, self.v_scale],
            [u32(position), u32(KV_HEADS), u32(count), u32(plane)],
            (KV_HEADS, count, 1),
            (HEAD_DIM, 1, 1),
            offsets=[0, 0, data_off, scale_off, data_off, scale_off],
        )

    def _attend_decode(
        self,
        query: int,
        output: int,
        *,
        length: int,
        data_off: int,
        scale_off: int,
        query_off: int = 0,
        output_off: int = 0,
    ) -> None:
        session = self.session
        consts = [
            u32(length),
            u32(self.max_pos),
            u32(KV_HEADS),
            u32(_GROUP),
            f32(ATTN_SCALE),
        ]
        grid = (ATTN_HEADS, 1, 1)
        threads = (HEAD_DIM, 1, 1)
        if self.kv_cache == "half":
            session.launch(
                "attend_decode_half",
                [query, self.k_data, self.v_data, self.scores, output],
                consts,
                grid,
                threads,
                offsets=[query_off, data_off, data_off, 0, output_off],
            )
            return
        session.launch(
            "attend_decode",
            [query, self.k_data, self.k_scale, self.v_data, self.v_scale, self.scores, output],
            consts,
            grid,
            threads,
            offsets=[query_off, data_off, scale_off, data_off, scale_off, 0, output_off],
        )

    def _attend_prefill(
        self,
        *,
        position0: int,
        count: int,
        data_off: int,
        scale_off: int,
    ) -> None:
        session = self.session
        consts = [
            u32(position0),
            u32(count),
            u32(self._score_span),
            u32(ATTN_HEADS),
            u32(KV_HEADS),
            u32(_GROUP),
            f32(ATTN_SCALE),
        ]
        grid = (ATTN_HEADS, count, 1)
        threads = (HEAD_DIM, 1, 1)
        if self.kv_cache == "half":
            session.launch(
                "attend_prefill_half",
                [self.pf_q, self.k_data, self.v_data, self.pf_scores, self.pf_attn],
                consts,
                grid,
                threads,
                offsets=[0, data_off, data_off, 0, 0],
            )
            return
        session.launch(
            "attend_prefill",
            [
                self.pf_q,
                self.k_data,
                self.k_scale,
                self.v_data,
                self.v_scale,
                self.pf_scores,
                self.pf_attn,
            ],
            consts,
            grid,
            threads,
            offsets=[0, data_off, scale_off, data_off, scale_off, 0, 0],
        )

    def _prefill_full(self, layer: int, count: int, row_pad: int, position0: int) -> None:
        session = self.session
        k_data_off, k_scale_off = self._kv_offsets(layer)
        qn = self._planes[f"layer.{layer}.q_norm"]
        kn = self._planes[f"layer.{layer}.k_norm"]
        self._cast_half(self.pf_normed, HIDDEN, row_pad, valid=count)
        self._matmul(
            self._planes[f"layer.{layer}.q"],
            self.pf_q,
            row_pad,
            ATTN_HEADS * HEAD_DIM * 2,
            HIDDEN,
        )
        self._matmul(
            self._planes[f"layer.{layer}.k"], self.pf_k, row_pad, KV_HEADS * HEAD_DIM, HIDDEN
        )
        self._matmul(
            self._planes[f"layer.{layer}.v"], self.pf_v, row_pad, KV_HEADS * HEAD_DIM, HIDDEN
        )
        session.launch(
            "head_rmsnorm_rows",
            [self.pf_q, self.weights],
            [u32(HEAD_DIM * 2), u32(ATTN_HEADS), u32(count), f32(RMS_EPS), f32(1.0)],
            (ATTN_HEADS, count, 1),
            (HEAD_DIM, 1, 1),
            offsets=[0, qn.data],
        )
        session.launch(
            "head_rmsnorm_rows",
            [self.pf_k, self.weights],
            [u32(HEAD_DIM), u32(KV_HEADS), u32(count), f32(RMS_EPS), f32(1.0)],
            (KV_HEADS, count, 1),
            (HEAD_DIM, 1, 1),
            offsets=[0, kn.data],
        )
        self._rope_rows(self.pf_q, ATTN_HEADS, HEAD_DIM * 2, count, position0)
        self._rope_rows(self.pf_k, KV_HEADS, HEAD_DIM, count, position0)
        for plane in (0, 1):
            self._store_kv(
                self.pf_k,
                self.pf_v,
                rows=True,
                position=position0,
                count=count,
                plane=plane,
                data_off=k_data_off,
                scale_off=k_scale_off,
            )
        span = position0 + count
        if span <= self._score_span:
            self._attend_prefill(
                position0=position0,
                count=count,
                data_off=k_data_off,
                scale_off=k_scale_off,
            )
        else:
            for index in range(count):
                self._attend_decode(
                    self.pf_q,
                    self.pf_attn,
                    length=position0 + index + 1,
                    data_off=k_data_off,
                    scale_off=k_scale_off,
                    query_off=index * ATTN_HEADS * HEAD_DIM * 2 * 4,
                    output_off=index * ATTN_HEADS * HEAD_DIM * 4,
                )
        self._cast_half(self.pf_attn, ATTN_HEADS * HEAD_DIM, row_pad, valid=count)
        self._matmul(
            self._planes[f"layer.{layer}.o"],
            self.pf_delta,
            row_pad,
            HIDDEN,
            ATTN_HEADS * HEAD_DIM,
        )

    def _reset_state(self) -> None:
        session = self.session
        session.flush()
        session.launch(
            "zero_f32",
            [self.gdn_state],
            [u32(GDN_STATE_BYTES // 4)],
            ((GDN_STATE_BYTES // 4 + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        hist = (LAYERS - len(FULL)) * _CONV_CH * (CONV_KERNEL - 1)
        session.launch(
            "zero_f32",
            [self.conv_hist],
            [u32(hist)],
            ((hist + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        session.launch(
            "zero_u32",
            [self.token_counts],
            [u32(VOCAB)],
            ((VOCAB + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        session.flush()
        self.pos = 0

    def _gemv(
        self,
        activation: Any,
        plane: Plane,
        destination: Any,
        rows: int,
        inner: int,
        *,
        act_off: int = 0,
        dest_off: int = 0,
    ) -> None:
        # Four-row GEMV shares the activation stream. It wins for mid-width
        # FFN/GDN outs; the eight-row GEMV stays faster for the head and for
        # tall-K downs.
        if rows % 4 == 0 and 256 <= rows <= 16384 and inner <= 4096:
            self.session.launch(
                "gemv_fp8_x4",
                [activation, self.weights, self.weights, destination],
                [u32(rows), u32(inner)],
                (rows // 4, 1, 1),
                (32, 1, 1),
                offsets=[act_off, plane.data, plane.scale, dest_off],
            )
            return
        self.session.launch(
            "gemv_fp8",
            [activation, self.weights, self.weights, destination],
            [u32(rows), u32(inner)],
            ((rows + GEMV_ROWS - 1) // GEMV_ROWS, 1, 1),
            (256, 1, 1),
            offsets=[act_off, plane.data, plane.scale, dest_off],
        )

    def _rmsnorm(self, source: Any, gain: Plane, destination: Any) -> None:
        self.session.launch(
            "rmsnorm",
            [source, self.weights, destination],
            [u32(HIDDEN), f32(RMS_EPS), f32(1.0)],
            (1, 1, 1),
            (256, 1, 1),
            offsets=[0, gain.data, 0],
        )

    def _step(self, token: int) -> None:
        session = self.session
        emb = self._planes["embedding"]
        memory.store(self.token_buf, 0, struct.pack("<I", token))
        session.begin()
        session.launch(
            "embed_row",
            [self.weights, self.weights, self.token_buf, self.hidden],
            [u32(0), u32(HIDDEN)],
            (HIDDEN, 1, 1),
            (1, 1, 1),
            offsets=[emb.data, emb.scale, 0, 0],
        )
        for layer in range(LAYERS):
            self._layer(layer)
        self._rmsnorm(self.hidden, self._planes["final_norm"], self.normed)
        self._gemv(self.normed, emb, self.logits, VOCAB, HIDDEN)
        session.end()
        self.pos += 1

    def _layer(self, layer: int) -> None:
        session = self.session
        self._rmsnorm(self.hidden, self._planes[f"layer.{layer}.in_norm"], self.normed)
        if is_full(layer):
            self._full_attn(layer)
        else:
            self._gdn(layer)
        session.launch(
            "residual_add",
            [self.hidden, self.delta],
            [u32(HIDDEN)],
            ((HIDDEN + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        self._rmsnorm(self.hidden, self._planes[f"layer.{layer}.post_norm"], self.normed)
        self._mlp(layer)
        session.launch(
            "residual_add",
            [self.hidden, self.delta],
            [u32(HIDDEN)],
            ((HIDDEN + 255) // 256, 1, 1),
            (256, 1, 1),
        )

    def _mlp(self, layer: int) -> None:
        session = self.session
        self._gemv(self.normed, self._planes[f"layer.{layer}.gate"], self.gate, INTERMEDIATE, HIDDEN)
        self._gemv(self.normed, self._planes[f"layer.{layer}.up"], self.up, INTERMEDIATE, HIDDEN)
        session.launch(
            "silu_mul",
            [self.gate, self.up, self.ff],
            [u32(INTERMEDIATE)],
            ((INTERMEDIATE + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        self._gemv(self.ff, self._planes[f"layer.{layer}.down"], self.delta, HIDDEN, INTERMEDIATE)

    def _gdn(self, layer: int) -> None:
        session = self.session
        gdn_i = self._gdn_index[layer]
        state_off = gdn_i * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * 4
        hist_off = gdn_i * _CONV_CH * (CONV_KERNEL - 1) * 4
        self._gemv(self.normed, self._planes[f"layer.{layer}.qkv"], self.mixed, _CONV_CH, HIDDEN)
        session.launch(
            "conv_silu",
            [self.weights, self.mixed, self.conv_hist],
            [u32(_CONV_CH)],
            ((_CONV_CH + 255) // 256, 1, 1),
            (256, 1, 1),
            offsets=[self._planes[f"layer.{layer}.conv"].data, 0, hist_off],
        )
        self._gemv(
            self.normed,
            self._planes[f"layer.{layer}.z"],
            self.z_buf,
            LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM,
            HIDDEN,
        )
        self._gemv(
            self.normed, self._planes[f"layer.{layer}.a"], self.a_buf, LINEAR_VALUE_HEADS, HIDDEN
        )
        self._gemv(
            self.normed, self._planes[f"layer.{layer}.b"], self.b_buf, LINEAR_VALUE_HEADS, HIDDEN
        )
        a_log = self._planes[f"layer.{layer}.A_log"]
        dt = self._planes[f"layer.{layer}.dt_bias"]
        session.launch(
            "gdn_step",
            [
                self.gdn_state,
                self.mixed,
                self.a_buf,
                self.b_buf,
                self.weights,
                self.weights,
                self.gdn_out,
            ],
            [u32(LINEAR_KEY_HEADS), u32(LINEAR_VALUE_HEADS)],
            (LINEAR_VALUE_HEADS, 1, 1),
            (128, 1, 1),
            offsets=[state_off, 0, 0, 0, a_log.data, dt.data, 0],
        )
        session.launch(
            "gated_rmsnorm",
            [self.gdn_out, self.z_buf, self.weights],
            [f32(RMS_EPS)],
            (LINEAR_VALUE_HEADS, 1, 1),
            (LINEAR_HEAD_DIM, 1, 1),
            offsets=[0, 0, self._planes[f"layer.{layer}.gdn_norm"].data],
        )
        self._gemv(
            self.gdn_out,
            self._planes[f"layer.{layer}.out"],
            self.delta,
            HIDDEN,
            LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM,
        )

    def _full_attn(self, layer: int) -> None:
        session = self.session
        length = self.pos + 1
        k_data_off, k_scale_off = self._kv_offsets(layer)
        qn = self._planes[f"layer.{layer}.q_norm"]
        kn = self._planes[f"layer.{layer}.k_norm"]

        self._gemv(
            self.normed,
            self._planes[f"layer.{layer}.q"],
            self.q_fuse,
            ATTN_HEADS * HEAD_DIM * 2,
            HIDDEN,
        )
        self._gemv(
            self.normed, self._planes[f"layer.{layer}.k"], self.k_buf, KV_HEADS * HEAD_DIM, HIDDEN
        )
        self._gemv(
            self.normed, self._planes[f"layer.{layer}.v"], self.v_buf, KV_HEADS * HEAD_DIM, HIDDEN
        )
        # q half lives in first HEAD_DIM of each 2*HEAD_DIM fused row; head_rmsnorm
        # with stride 2*HEAD_DIM norms the q half in place.
        session.launch(
            "head_rmsnorm",
            [self.q_fuse, self.weights],
            [u32(HEAD_DIM * 2), f32(RMS_EPS), f32(1.0)],
            (ATTN_HEADS, 1, 1),
            (HEAD_DIM, 1, 1),
            offsets=[0, qn.data],
        )
        session.launch(
            "head_rmsnorm",
            [self.k_buf, self.weights],
            [u32(HEAD_DIM), f32(RMS_EPS), f32(1.0)],
            (KV_HEADS, 1, 1),
            (HEAD_DIM, 1, 1),
            offsets=[0, kn.data],
        )
        # rope only the query half: rope_heads with stride 2*HEAD_DIM
        rope_pos = max(0, self.pos + int(self._rope_delta))
        session.launch(
            "rope_heads",
            [self.q_fuse],
            [u32(ATTN_HEADS), u32(HEAD_DIM * 2), u32(ROPE_DIM), u32(rope_pos), f32(THETA)],
            ((ATTN_HEADS * (ROPE_DIM // 2) + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        session.launch(
            "rope_heads",
            [self.k_buf],
            [u32(KV_HEADS), u32(HEAD_DIM), u32(ROPE_DIM), u32(rope_pos), f32(THETA)],
            ((KV_HEADS * (ROPE_DIM // 2) + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        for plane in (0, 1):
            self._store_kv(
                self.k_buf,
                self.v_buf,
                rows=False,
                position=self.pos,
                count=1,
                plane=plane,
                data_off=k_data_off,
                scale_off=k_scale_off,
            )
        self._attend_decode(
            self.q_fuse,
            self.attn,
            length=length,
            data_off=k_data_off,
            scale_off=k_scale_off,
        )
        self._gemv(
            self.attn, self._planes[f"layer.{layer}.o"], self.delta, HIDDEN, ATTN_HEADS * HEAD_DIM
        )

    def _sample(self) -> int:
        session = self.session
        if self._presence or self._frequency:
            session.launch(
                "penalize",
                [self.logits, self.token_counts],
                [u32(VOCAB), f32(self._presence), f32(self._frequency)],
                ((VOCAB + 255) // 256, 1, 1),
                (256, 1, 1),
            )
        groups = (VOCAB + 255) // 256
        session.launch(
            "argmax_partial",
            [self.logits, self.argmax_v, self.argmax_i],
            [u32(VOCAB)],
            (groups, 1, 1),
            (32, 1, 1),
        )
        session.launch(
            "argmax_final",
            [self.argmax_v, self.argmax_i, self.token_out],
            [u32(groups), u32(0)],
            (1, 1, 1),
            (32, 1, 1),
        )
        if self._presence or self._frequency:
            session.launch(
                "bump_count",
                [self.token_counts, self.token_out],
                [u32(0)],
                (1, 1, 1),
                (1, 1, 1),
            )
        session.flush()
        return int.from_bytes(memory.load(self.token_out, 0, 4), "little")


def _i8_resident_bytes() -> int:
    """Bytes of restored int8 codes plus channel scales for every int8 plane."""
    total = 0
    for layer in range(LAYERS):
        for name, (cols, inner) in _I8_SHAPES.items():
            if name in _I8_FFN or not is_full(layer):
                total += cols * inner + cols * 4
    return total


def _workspace_bytes(capacity: int) -> int:
    tile = _TILE
    widest = max(
        HIDDEN,
        INTERMEDIATE,
        _CONV_CH,
        _GDN_Z,
        ATTN_HEADS * HEAD_DIM * 2,
        ATTN_HEADS * HEAD_DIM,
        KV_HEADS * HEAD_DIM,
        LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM,
    )
    floats = (
        HIDDEN * 4
        + INTERMEDIATE * 3
        + _CONV_CH
        + LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM * 2
        + LINEAR_VALUE_HEADS * 2
        + ATTN_HEADS * HEAD_DIM * 3
        + KV_HEADS * HEAD_DIM * 2
        + ATTN_HEADS * capacity
        + VOCAB
        + (VOCAB + 255) // 256 * 2
        + 2
        + tile
        + tile * HIDDEN * 3
        + tile * INTERMEDIATE * 3
        + tile * _CONV_CH
        + tile * _GDN_Z
        + tile * LINEAR_VALUE_HEADS * 2
        + tile * LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
        + tile * ATTN_HEADS * HEAD_DIM * 3
        + tile * KV_HEADS * HEAD_DIM * 2
        + tile * ATTN_HEADS * min(capacity, 8192)
    )
    i8 = _I8_WEIGHT_BYTES + _I8_COLS * 4 + tile * _I8_INNER + tile * 4
    return floats * 4 + tile * widest * 2 + i8

