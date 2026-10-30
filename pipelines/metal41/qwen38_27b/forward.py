"""Text forward. Weights are fp4, with promoted feed-forward layers at fp8. The key/value cache is fp8.

A feed-forward layer that does not fit stays in the record and is copied into one
stage before that layer runs. The MTP head leaves the device when the prompt
needs its bytes. Decode does not run that block.

Temperature 0 is an argmax. A positive temperature samples with top-k and
top-p. Presence, frequency, and logit bias follow the OpenAI logit update.
A repeated span of 16 or more tokens ends the completion. An image request
runs the vision tower and writes those tokens into the prompt.
"""

from __future__ import annotations

import gc
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import objc

from favlo.memory import memory
from favlo.pipeline import ClientClosed
from qwen38_27b.attn import bind_plane, value_block_bytes
from qwen38_27b.budget import (
    choose_wire,
    group_tensors,
    same_weight_image,
    startup_bytes,
)
from qwen38_27b.cache import (
    DEFAULT_CONTEXT_LENGTH,
    kv_mode,
    layer_cache_bytes,
    max_position,
    parse_memory,
    plane_bytes,
    sequence_cache_bytes,
)
from qwen38_27b.layout import (
    AB,
    ATTN_OUT,
    CHUNK,
    EPS,
    FULL_LAYERS,
    HIDDEN,
    INTERMEDIATE,
    KV_PROJ,
    LAYERS,
    LINEAR_LAYERS,
    LINEAR_V,
    MIX,
    MTP_SLOT,
    Q_HEADS,
    Q_PROJ,
    THETA,
    VOCAB,
    Z_WIDTH,
)
from qwen38_27b.prefill import _Workspace
from qwen38_27b.prefix import open_session, pack_plane, unpack_plane
from qwen38_27b.resident import DeviceImage
from qwen38_27b.session import Session, release_buffers


class RequestClosed(ClientClosed):
    """The client went away or the process was signaled."""


_ARGMAX_GROUPS = (VOCAB + 255) // 256


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


_STATE_BYTES = LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4
_HISTORY_BYTES = LINEAR_LAYERS * 3 * MIX * 2
_DEVICE_FIELDS = (
    "resident",
    "history",
    "state",
    "hidden",
    "normed",
    "hold",
    "wide",
    "z",
    "y",
    "query",
    "gate",
    "kv",
    "vv",
    "ab",
    "ab2",
    "logits",
    "best_v",
    "best_i",
    "plane_q",
    "plane_scores",
    "plane_probs",
    "plane_prob_half",
    "plane_row_scale",
    "plane_parts",
    "plane_partial",
    "plane_acc",
    "plane_max",
    "plane_sum",
    "plane_tail",
    "plane_chunk_max",
    "plane_chunk_sum",
)


def _device_buffers(runtime: Any) -> list[Any]:
    """Metal buffers this runtime still holds. The caller releases them after the drop."""
    found: list[Any] = []
    weights = getattr(runtime, "weights", None)
    buffer = getattr(weights, "buffer", None)
    if buffer is not None:
        found.append(buffer)
    for name in ("k_cache", "v_cache", "_held"):
        held = getattr(runtime, name, None)
        if isinstance(held, list):
            found.extend(item for item in held if item is not None)
    for name in _DEVICE_FIELDS:
        item = getattr(runtime, name, None)
        if item is not None:
            found.append(item)
    blob = getattr(runtime, "_vision_blob", None)
    if blob is not None:
        found.append(blob)
    return found


_SCRATCH_FIELDS = (
    "plane_q",
    "plane_scores",
    "plane_probs",
    "plane_prob_half",
    "plane_row_scale",
    "plane_parts",
    "plane_partial",
    "plane_acc",
    "plane_max",
    "plane_sum",
    "plane_tail",
    "plane_chunk_max",
    "plane_chunk_sum",
)


class Qwen38Runtime:
    """One resident record. ``close`` drops its buffers."""

    def __init__(
        self,
        path: Path,
        *,
        kv_cache: str = "e4m3",
        console: Any = None,
        resident_percent: float | None = None,
        max_memory: str | int | None = None,
        owner: str | None = None,
    ) -> None:
        del resident_percent
        self._owner = owner or f"qwen38-27b:{id(self)}"
        self.root = Path(path)
        self.console = console
        self.kv_cache = kv_mode(kv_cache)
        self._request_id = 0
        self._hit_limit = False
        self._halted = False
        self._stop = None
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            message = f"{self.root.name} has no manifest.json"
            raise FileNotFoundError(message)
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete"):
            message = f"{self.root.name} is not a finished record"
            raise RuntimeError(message)
        self.tensors: dict[str, dict] = self.manifest["tensors"]
        self._record_mtp = "mtp.fc.weight" in self.tensors
        self._group = group_tensors(self.tensors)
        self.layer_types = tuple(self.manifest.get("layer_types") or _default_layers())
        self.linear_slot = _slots(self.layer_types, "linear_attention")
        self.full_slot = _slots(self.layer_types, "full_attention")
        card_limit = max_position(self.root) if (self.root / "config.json").is_file() else 262144
        self._limit = min(card_limit, DEFAULT_CONTEXT_LENGTH)
        self._memory_cap = parse_memory(max_memory)
        self.filled = 0
        self.drafts = False
        self.kv_slots = FULL_LAYERS
        self.max_pos = 0
        self.wire = None
        self.weights = None
        self._record = None
        self._prefill = None
        self._prefill_progress = None
        self._load_progress = None
        self._on_prefill = None
        self._prefill_noted = 0
        self._output_ids = []
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._cached_tokens = 0
        self.session = None
        self.resident = None
        self.k_cache = []
        self.v_cache = []
        plan = choose_wire(
            memory.budget() if self._memory_cap is None else self._memory_cap,
            2048,
            self._group,
            self._limit,
            mtp_available=self._record_mtp,
        )
        summary = ""
        self._load_progress = self._begin_load(
            startup_bytes(plan.weights, plan.positions, slots=plan.slots, drafts=plan.mtp)
        )
        try:
            self.session = Session(self._owner)
            self.session._load_note = self._note_load
            self._record = (self.root / "resident.bin").open("rb")
            zeros = self.session.zeros
            self.history = zeros(_HISTORY_BYTES)
            self.state = zeros(_STATE_BYTES)
            self.hidden = zeros(HIDDEN * 4)
            self.normed = zeros(HIDDEN * 4)
            self.hold = None
            self.wide = zeros(max(MIX, Q_PROJ, INTERMEDIATE) * 4)
            self.z = zeros(max(Z_WIDTH, INTERMEDIATE) * 4)
            self.y = zeros(max(Z_WIDTH, ATTN_OUT, INTERMEDIATE) * 4)
            self.query = zeros(ATTN_OUT * 4)
            self.gate = zeros(ATTN_OUT * 4)
            self.kv = zeros(KV_PROJ * 4)
            self.vv = zeros(KV_PROJ * 4)
            self.ab = zeros(AB * 4)
            self.ab2 = zeros(AB * 4)
            self.logits = zeros(VOCAB * 4)
            self.best_v = zeros(_ARGMAX_GROUPS * 4)
            self.best_i = zeros(_ARGMAX_GROUPS * 4)
            self._apply(plan, 0, [])
            self._prefill = _Workspace(self.session, drafts=plan.mtp)
            self.prefix = open_session(self)
            summary = self._loaded_summary()
        finally:
            if self.session is not None:
                self.session._load_note = None
            progress = self._load_progress
            self._load_progress = None
            if progress is not None and self.console is not None:
                self.console.end(progress, summary)
            elif summary:
                self._status(summary)
            if not summary:
                self.close()

    def close(self) -> None:
        """Write the live prefix, mark every device buffer reusable, then return it."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            prefix = getattr(self, "prefix", None)
            session = getattr(self, "session", None)
            if prefix is not None and session is not None:
                try:
                    prefix.flush()
                except (OSError, RuntimeError):
                    pass
            adapter = getattr(prefix, "adapter", None) if prefix is not None else None
            if adapter is not None:
                adapter.runtime = None
            self.prefix = None
            buffers = _device_buffers(self)
            self._end_open_progress()
            self._prefill = None
            weights = getattr(self, "weights", None)
            if weights is not None:
                weights.close()
            self.weights = None
            record = getattr(self, "_record", None)
            self._record = None
            if record is not None:
                record.close()
            for name in ("k_cache", "v_cache", "_held"):
                held = getattr(self, name, None)
                if isinstance(held, list):
                    held.clear()
            for name in _DEVICE_FIELDS:
                if hasattr(self, name):
                    setattr(self, name, None)
            if session is not None:
                finish = getattr(session, "finish", None)
                if callable(finish):
                    try:
                        finish()
                    except (OSError, RuntimeError):
                        pass
            release_buffers(buffers)
            if session is not None:
                try:
                    session.close()
                except (OSError, RuntimeError):
                    pass
            self.session = None
            weights = None
            record = None
            prefix = None
            adapter = None
            session = None
            buffers.clear()
            with objc.autorelease_pool():
                gc.collect()
        finally:
            self._sweep_owner()

    def _sweep_owner(self) -> None:
        """Return this opening's spans. ``session.close`` sweeps the same string."""
        session = getattr(self, "session", None)
        if session is not None:
            try:
                session.close()
            except (OSError, RuntimeError):
                pass
            self.session = None
        owner = getattr(self, "_owner", None)
        if not isinstance(owner, str) or not owner:
            return
        memory.sweep(owner)

    def ensure(self, needed: int) -> bool:
        """Grow the cache so ``needed`` positions fit. Leave the device alone when it cannot."""
        if needed <= self.max_pos:
            return True
        budget = self._growth_budget()
        previous = self.wire
        count = 0
        saved: list[tuple[int, str, bytes]] = []
        captured = False
        extra = 0
        intact = True
        tried: set[tuple[int, bool, tuple[int, ...]]] = set()
        while True:
            try:
                plan = choose_wire(
                    budget,
                    needed,
                    self._group,
                    self._limit,
                    extra_unwire=extra,
                    mtp_available=self._record_mtp,
                )
            except RuntimeError as exc:
                if "does not fit" not in str(exc):
                    raise
                break
            key = (plan.positions, plan.mtp, plan.wired)
            if plan.positions < needed or key in tried:
                break
            tried.add(key)
            if not captured:
                count, saved = self._capture_planes()
                captured = True
            try:
                if same_weight_image(previous, plan):
                    self._extend_cache(plan, count, saved)
                else:
                    self._apply(plan, count, saved)
            except RuntimeError as exc:
                if "device refused" not in str(exc):
                    raise
                if self.weights is None or not self.k_cache:
                    intact = False
                extra += 1
                continue
            self._note_growth(previous, plan)
            return True
        if not intact and previous is not None:
            try:
                self._apply(previous, count, saved)
            except RuntimeError as exc:
                if "device refused" not in str(exc):
                    raise
        return self.max_pos >= needed

    def prepare_ffn(self, layer: int | None) -> None:
        """Copy an unwired feed-forward layer into the stage before its kernels run."""
        weights = self.weights
        if layer is None or weights is None or layer in weights.wired:
            return
        if weights.staged == layer:
            if getattr(self.session, "_pass_encoder", None) is None:
                self.session.resume()
            weights.stage_busy = True
            return
        if weights.stage_busy:
            self.session.finish()
            weights.stage_busy = False
        weights.load_layer(layer)
        if getattr(self.session, "_pass_encoder", None) is None:
            self.session.resume()
        weights.stage_busy = True

    def _budget(self) -> int:
        if self._memory_cap is not None:
            return self._memory_cap
        return memory.budget()

    def _growth_budget(self) -> int:
        """Bytes the next image may use while the host copy of the live cache stays."""
        if self._memory_cap is not None:
            return self._memory_cap
        count = min(int(self.filled), int(self.max_pos))
        packed = sequence_cache_bytes(count, self.kv_slots) if count > 0 and self.k_cache else 0
        return max(0, memory.budget() - packed)

    def _held_bytes(self) -> int:
        if self.weights is None or not self.max_pos:
            return 0
        return startup_bytes(
            self.weights.nbytes, self.max_pos, slots=self.kv_slots, drafts=self.drafts
        )

    def _apply(self, plan: Any, count: int, saved: list[tuple[int, str, bytes]]) -> None:
        self._release_elastic()
        assert self.session is not None
        assert self._record is not None
        image = DeviceImage(
            self.session,
            self._record,
            self._group,
            plan,
            self.session._load_note,
        )
        self.weights = image
        self.resident = image.buffer
        self.drafts = plan.mtp
        self.kv_slots = plan.slots
        self.max_pos = plan.positions
        self.wire = plan
        zeros = self.session.zeros
        one = plane_bytes(plan.positions)
        self.k_cache = [zeros(one) for _ in range(plan.slots)]
        self.v_cache = [zeros(one) for _ in range(plan.slots)]
        bind_plane(self)
        self.hold = zeros(HIDDEN * 4) if plan.mtp else None
        work = self._prefill
        if work is not None:
            work.mix = zeros(CHUNK * MIX * 4) if plan.mtp else None
        self._restore_planes(count, saved)
        self._held = [self.resident, self.history, self.state, *self.k_cache, *self.v_cache]

    def _extend_cache(self, plan: Any, count: int, saved: list[tuple[int, str, bytes]]) -> None:
        """Replace the key/value planes and release the planes they replace."""
        assert self.session is not None
        old = [getattr(self, "plane_parts", None), *self.k_cache, *self.v_cache]
        zeros = self.session.zeros
        made: list[Any] = []
        try:
            parts = zeros(value_block_bytes(plan.positions))
            made.append(parts)
            one = plane_bytes(plan.positions)
            new_k: list[Any] = []
            new_v: list[Any] = []
            for _slot in range(plan.slots):
                new_k.append(zeros(one))
                made.append(new_k[-1])
            for _slot in range(plan.slots):
                new_v.append(zeros(one))
                made.append(new_v[-1])
        except RuntimeError:
            release_buffers(made)
            raise
        self.plane_parts = parts
        self.k_cache = new_k
        self.v_cache = new_v
        self.max_pos = plan.positions
        self.wire = plan
        release_buffers(old)
        self._restore_planes(count, saved)
        self._held = [self.resident, self.history, self.state, *self.k_cache, *self.v_cache]

    def _release_elastic(self) -> None:
        """Drop the weight image, the cache, and the scratch, and return their spans."""
        weights = getattr(self, "weights", None)
        image = getattr(weights, "buffer", None) if weights is not None else None
        if weights is not None:
            weights.close()
        self.weights = None
        resident = getattr(self, "resident", None)
        self.resident = None
        caches = [*getattr(self, "k_cache", ()), *getattr(self, "v_cache", ())]
        self.k_cache = []
        self.v_cache = []
        hold = getattr(self, "hold", None)
        self.hold = None
        scratch: list[Any] = []
        for name in _SCRATCH_FIELDS:
            item = getattr(self, name, None)
            if item is not None:
                scratch.append(item)
                setattr(self, name, None)
        work = getattr(self, "_prefill", None)
        mix = getattr(work, "mix", None) if work is not None else None
        if work is not None:
            work.mix = None
        held = getattr(self, "_held", None)
        if isinstance(held, list):
            held.clear()
        release_buffers((image, resident, hold, mix, *caches, *scratch))

    def _capture_planes(self) -> tuple[int, list[tuple[int, str, bytes]]]:
        count = min(int(self.filled), int(self.max_pos))
        if count < 1 or not self.k_cache:
            return 0, []
        saved: list[tuple[int, str, bytes]] = []
        for slot in range(self.kv_slots):
            saved.append((slot, "key", self._pack_plane(self.k_cache[slot], count, kind="key")))
            saved.append((slot, "value", self._pack_plane(self.v_cache[slot], count, kind="value")))
        return count, saved

    def _pack_plane(self, buffer: Any, count: int, *, kind: str) -> bytes:
        view = np.frombuffer(
            memoryview(buffer.contents().as_buffer(plane_bytes(self.max_pos))),
            dtype=np.uint8,
        )
        return pack_plane(view, self.max_pos, count, kind=kind)

    def _restore_planes(self, count: int, saved: list[tuple[int, str, bytes]]) -> None:
        if count < 1:
            return
        for slot, kind, blob in saved:
            if slot >= self.kv_slots:
                continue
            buffer = self.k_cache[slot] if kind == "key" else self.v_cache[slot]
            view = np.frombuffer(
                memoryview(buffer.contents().as_buffer(plane_bytes(self.max_pos))),
                dtype=np.uint8,
            )
            unpack_plane(view, self.max_pos, count, blob, kind=kind)

    def _loaded_summary(self) -> str:
        text = (
            f"loaded {self.root.name} kv {self.kv_cache} context {self.max_pos} "
            f"weights {self.weights.nbytes} cache {sequence_cache_bytes(self.max_pos, self.kv_slots)}"
        )
        return text + self._fit_suffix()

    def _fit_suffix(self) -> str:
        suffix = ""
        if self._record_mtp and not self.drafts:
            suffix += " mtp off"
        wired = len(self.wire.wired) if self.wire is not None else LAYERS
        if wired != LAYERS:
            suffix += f" feed-forward {wired}/{LAYERS}"
        return suffix

    def _note_growth(self, previous: Any, plan: Any) -> None:
        if previous is not None and (
            previous.positions != plan.positions
            or previous.mtp != plan.mtp
            or previous.wired != plan.wired
        ):
            self._status(f"context {plan.positions}{self._fit_suffix()}")

    def _end_open_progress(self) -> None:
        console = getattr(self, "console", None)
        for name in ("_load_progress", "_prefill_progress"):
            progress = getattr(self, name, None)
            setattr(self, name, None)
            if progress is None or console is None or getattr(progress, "closed", True):
                continue
            try:
                console.end(progress, "")
            except (OSError, RuntimeError, AttributeError):
                pass

    def halt(self) -> None:
        """Stop before the next chunk or the next token."""
        self._halted = True

    def _check_stop(self) -> None:
        if self._halted:
            raise RequestClosed()
        stop = self._stop
        if stop is not None and stop():
            raise RequestClosed()

    def reset(self) -> None:
        session = self.session
        for buffer, nbytes in (
            (self.history, _HISTORY_BYTES),
            (self.state, _STATE_BYTES),
        ):
            _wipe(buffer, nbytes)
        for cache in (*self.k_cache, *self.v_cache):
            _wipe(cache, plane_bytes(self.max_pos))
        _wipe(self.plane_tail, self.kv_slots * 4 * 32 * 256 * 4)
        self.filled = 0
        del session

    def weight(self, layer: int, rest: str) -> str:
        return self._resolve(
            f"model.language_model.layers.{layer}.{rest}",
            f"model.layers.{layer}.{rest}",
        )

    def locate(self, *names: str) -> tuple[Any, int, dict]:
        name = self._resolve(*names)
        entry = self.tensors[name]
        return self.resident, self.weights.offset(name), entry

    def locate_weight(self, layer: int, rest: str) -> tuple[Any, int, dict]:
        return self.locate(self.weight(layer, rest))

    def scale_offset(self, entry: dict, offset: int) -> int:
        return offset + int(entry["data_nbytes"]) + int(entry["scale_pad"])

    def _resolve(self, *names: str) -> str:
        for name in names:
            if name in self.tensors:
                return name
        message = f"{names[0]} is not in this record"
        raise KeyError(message)

    def _gemv(self, act: Any, name: str, dest: Any) -> None:
        buf, offset, entry = self.locate(name)
        rows, inner = (int(value) for value in entry["stored_shape"])
        scale = self.scale_offset(entry, offset)
        kind = str(entry.get("dtype", "fp4"))
        if kind == "fp8":
            if rows % 4 or inner % 16:
                message = f"{name} is not an fp8 four-row tile"
                raise RuntimeError(message)
            threads = 160 if (inner >> 4) % 160 == 0 else 256
            self.session.encode(
                "gemv_fp8_x4",
                [(act, 0), (buf, offset), (buf, scale), (dest, 0)],
                [_u32(rows), _u32(inner)],
                (rows // 4, 1, 1),
                (threads, 1, 1),
            )
            return
        if rows % 4 == 0:
            self.session.encode(
                "gemv_fp4_x4",
                [(act, 0), (buf, offset), (buf, scale), (dest, 0)],
                [_u32(rows), _u32(inner)],
                (rows // 4, 1, 1),
                (32, 1, 1),
            )
            return
        self.session.encode(
            "gemv_fp4_vec",
            [(act, 0), (buf, offset), (buf, scale), (dest, 0)],
            [_u32(rows), _u32(inner)],
            (rows, 1, 1),
            (32, 1, 1),
        )

    def _rms(self, src: Any, name: str, dest: Any) -> None:
        buf, offset, _entry = self.locate(name)
        self.session.encode(
            "rmsnorm_f32",
            [(src, 0), (buf, offset), (dest, 0)],
            [_u32(HIDDEN), _f32(EPS)],
            (1, 1, 1),
            (256, 1, 1),
        )

    def _embed(self, token: int) -> None:
        buf, offset, entry = self.locate(
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
        _rows, inner = entry["stored_shape"]
        self.session.encode(
            "gather_fp4_row",
            [(buf, offset), (buf, self.scale_offset(entry, offset)), (self.hidden, 0)],
            [_u32(token), _u32(int(inner))],
            (1, 1, 1),
            (32, 1, 1),
        )

    def _gdn(self, layer: int) -> None:
        qkv_w, qkv_s = self._pair(self.weight(layer, "linear_attn.in_proj_qkv.weight"))
        z_w, z_s = self._pair(self.weight(layer, "linear_attn.in_proj_z.weight"))
        a_w, a_s = self._pair(self.weight(layer, "linear_attn.in_proj_a.weight"))
        b_w, b_s = self._pair(self.weight(layer, "linear_attn.in_proj_b.weight"))
        self.session.encode(
            "gdn_in_proj",
            [
                (self.normed, 0),
                qkv_w,
                qkv_s,
                z_w,
                z_s,
                a_w,
                a_s,
                b_w,
                b_s,
                (self.wide, 0),
                (self.z, 0),
                (self.ab, 0),
                (self.ab2, 0),
            ],
            [],
            (MIX + Z_WIDTH + AB + AB, 1, 1),
            (32, 1, 1),
        )
        slot = self.linear_slot[layer]
        weight, woff, _entry = self.locate_weight(layer, "linear_attn.conv1d.weight")
        self.session.encode(
            "gdn_conv_shift",
            [
                (self.history, slot * 3 * MIX * 2),
                (self.wide, 0),
                (weight, woff),
                (self.wide, 0),
            ],
            [_u32(MIX)],
            (MIX // 256, 1, 1),
            (256, 1, 1),
        )
        along, aoff, _entry = self.locate_weight(layer, "linear_attn.A_log")
        bias, boff, _entry = self.locate_weight(layer, "linear_attn.dt_bias")
        norm, noff, _entry = self.locate_weight(layer, "linear_attn.norm.weight")
        self.session.encode(
            "gdn_state_rows",
            [
                (self.state, slot * LINEAR_V * 128 * 128 * 4),
                (self.wide, 0),
                (self.ab, 0),
                (self.ab2, 0),
                (self.z, 0),
                (along, aoff),
                (bias, boff),
                (norm, noff),
                (self.y, 0),
            ],
            [_f32(EPS), _u32(1)],
            (LINEAR_V, 1, 1),
            (128, 1, 1),
        )
        self._gemv(self.y, self.weight(layer, "linear_attn.out_proj.weight"), self.wide)
        self.session.encode(
            "residual_rows",
            [(self.hidden, 0), (self.wide, 0), (self.hidden, 0)],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )

    def _attend(
        self, layer: int | None, pos: int, rope: tuple[int, int, int] | None = None
    ) -> None:
        from qwen38_27b.attn import attend_plane

        q_name = self._w(layer, "self_attn.q_proj.weight")
        k_name = self._w(layer, "self_attn.k_proj.weight")
        v_name = self._w(layer, "self_attn.v_proj.weight")
        kinds = {self._dtype(name) for name in (q_name, k_name, v_name)}
        if kinds == {"fp8"}:
            self._gemv(self.normed, q_name, self.wide)
            self._gemv(self.normed, k_name, self.kv)
            self._gemv(self.normed, v_name, self.vv)
        elif kinds == {"fp4"}:
            q_w, q_s = self._pair(q_name)
            k_w, k_s = self._pair(k_name)
            v_w, v_s = self._pair(v_name)
            self.session.encode(
                "attn_qkv",
                [
                    (self.normed, 0),
                    q_w,
                    q_s,
                    k_w,
                    k_s,
                    v_w,
                    v_s,
                    (self.wide, 0),
                    (self.kv, 0),
                    (self.vv, 0),
                ],
                [],
                (Q_PROJ + KV_PROJ + KV_PROJ, 1, 1),
                (32, 1, 1),
            )
        else:
            message = f"{q_name} dtype {sorted(kinds)} is not a decode projection"
            raise RuntimeError(message)
        self.session.encode(
            "split_q_gate_rows",
            [(self.wide, 0), (self.query, 0), (self.gate, 0)],
            [_u32(1)],
            (Q_HEADS, 1, 1),
            (256, 1, 1),
        )
        q_buf, q_off, _entry = self.locate(self._w(layer, "self_attn.q_norm.weight"))
        k_buf, k_off, _entry = self.locate(self._w(layer, "self_attn.k_norm.weight"))
        self.session.encode(
            "qk_norm_rope",
            [(self.query, 0), (self.kv, 0), (q_buf, q_off), (k_buf, k_off)],
            [
                _f32(THETA),
                _f32(EPS),
                _u32(pos if rope is None else rope[0]),
                _u32(pos if rope is None else rope[1]),
                _u32(pos if rope is None else rope[2]),
            ],
            (Q_HEADS + 4, 1, 1),
            (256, 1, 1),
        )
        attend_plane(
            self,
            query=self.query,
            key=self.kv,
            value=self.vv,
            gate=self.gate,
            dest=self.y,
            pos0=pos,
            count=1,
            slot=MTP_SLOT if layer is None else self.full_slot[layer],
        )
        self._gemv(self.y, self._w(layer, "self_attn.o_proj.weight"), self.wide)
        self.session.encode(
            "residual_rows",
            [(self.hidden, 0), (self.wide, 0), (self.hidden, 0)],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )

    def _ffn(self, layer: int | None) -> None:
        self.prepare_ffn(layer)
        gate_name = self._w(layer, "mlp.gate_proj.weight")
        up_name = self._w(layer, "mlp.up_proj.weight")
        _buf, _offset, gate_entry = self.locate(gate_name)
        _buf, _offset, up_entry = self.locate(up_name)
        gate_kind = str(gate_entry.get("dtype", "fp4"))
        up_kind = str(up_entry.get("dtype", "fp4"))
        if gate_kind != up_kind or gate_kind not in {"fp4", "fp8"}:
            message = f"{gate_name} and {up_name} widths differ"
            raise RuntimeError(message)
        if gate_kind == "fp8":
            self._gemv(self.normed, gate_name, self.wide)
            self._gemv(self.normed, up_name, self.z)
        else:
            gate_w, gate_s = self._pair(gate_name)
            up_w, up_s = self._pair(up_name)
            self.session.encode(
                "ffn_gate_up",
                [
                    (self.normed, 0),
                    gate_w,
                    gate_s,
                    up_w,
                    up_s,
                    (self.wide, 0),
                    (self.z, 0),
                ],
                [_u32(INTERMEDIATE), _u32(HIDDEN)],
                (INTERMEDIATE * 2, 1, 1),
                (32, 1, 1),
            )
        self.session.encode(
            "swiglu_f32",
            [(self.wide, 0), (self.z, 0), (self.y, 0)],
            [],
            (INTERMEDIATE // 256, 1, 1),
            (256, 1, 1),
        )
        self._gemv(self.y, self._w(layer, "mlp.down_proj.weight"), self.wide)
        self.session.encode(
            "residual_rows",
            [(self.hidden, 0), (self.wide, 0), (self.hidden, 0)],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )

    def _w(self, layer: int | None, rest: str) -> str:
        if layer is None:
            return f"mtp.layers.0.{rest}"
        return self.weight(layer, rest)

    def _dtype(self, name: str) -> str:
        _buf, _offset, entry = self.locate(name)
        return str(entry.get("dtype", "fp4"))

    def _copy_normed(self) -> None:
        """Copy the post-norm vector in ``normed`` back onto ``hidden``."""
        self.session.encode(
            "copy_f32",
            [(self.normed, 0), (self.hidden, 0)],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )

    def _pair(self, name: str) -> tuple[tuple[Any, int], tuple[Any, int]]:
        buf, offset, entry = self.locate(name)
        return (buf, offset), (buf, self.scale_offset(entry, offset))

    def _rope_at(self, index: int) -> tuple[int, int, int] | None:
        rope = getattr(self, "_rope_ids", None)
        if rope is None:
            return None
        row = rope[index]
        return int(row[0]), int(row[1]), int(row[2])

    def step(
        self,
        token: int,
        pos: int,
        *,
        logits: bool,
        rope: tuple[int, int, int] | None = None,
    ) -> int | None:
        if pos >= self.max_pos:
            message = f"position {pos} is past {self.max_pos}"
            raise RuntimeError(message)
        self.session.begin()
        try:
            self._embed(token)
            for layer, kind in enumerate(self.layer_types):
                if layer > 0 and layer % 4 == 0:
                    self.session.submit()
                self._rms(self.hidden, self.weight(layer, "input_layernorm.weight"), self.normed)
                if kind == "full_attention":
                    self._attend(layer, pos, rope)
                else:
                    self._gdn(layer)
                self._rms(
                    self.hidden,
                    self.weight(layer, "post_attention_layernorm.weight"),
                    self.normed,
                )
                self._ffn(layer)
            final = self._resolve(
                "model.language_model.norm.weight",
                "model.norm.weight",
            )
            self._rms(self.hidden, final, self.normed)
            if logits:
                head = self._resolve("lm_head.weight", "model.language_model.lm_head.weight")
                self._gemv(self.normed, head, self.logits)
                self.session.encode(
                    "argmax_partial",
                    [(self.logits, 0), (self.best_v, 0), (self.best_i, 0)],
                    [_u32(VOCAB)],
                    (_ARGMAX_GROUPS, 1, 1),
                    (256, 1, 1),
                )
            self.session.end()
        except Exception:
            self.session.finish()
            raise
        self.filled = pos + 1
        if not logits:
            return None
        return _best(self)

    def _argmax_hidden(self) -> int:
        """Argmax of the residual already stored in ``hidden``."""
        self.session.begin()
        try:
            final = self._resolve(
                "model.language_model.norm.weight",
                "model.norm.weight",
            )
            self._rms(self.hidden, final, self.normed)
            head = self._resolve("lm_head.weight", "model.language_model.lm_head.weight")
            self._gemv(self.normed, head, self.logits)
            self.session.encode(
                "argmax_partial",
                [(self.logits, 0), (self.best_v, 0), (self.best_i, 0)],
                [_u32(VOCAB)],
                (_ARGMAX_GROUPS, 1, 1),
                (256, 1, 1),
            )
            self.session.end()
        except Exception:
            self.session.finish()
            raise
        return _best(self)

    def _fill_logits(self) -> None:
        """Write the final-norm logits for the residual already stored in ``hidden``."""
        self.session.begin()
        try:
            final = self._resolve(
                "model.language_model.norm.weight",
                "model.norm.weight",
            )
            self._rms(self.hidden, final, self.normed)
            head = self._resolve("lm_head.weight", "model.language_model.lm_head.weight")
            self._gemv(self.normed, head, self.logits)
            self.session.end()
        except Exception:
            self.session.finish()
            raise

    def _mtp_body(self, token: int, pos: int) -> None:
        """MTP layer at ``pos``. ``hidden`` is already the post-norm state. Embeds ``token``."""
        self._rms(self.hidden, "mtp.pre_fc_norm_hidden.weight", self.hold)
        self._embed(token)
        self._rms(self.hidden, "mtp.pre_fc_norm_embedding.weight", self.normed)
        self.session.encode(
            "concat_pair",
            [(self.normed, 0), (self.hold, 0), (self.wide, 0)],
            [_u32(HIDDEN), _u32(1)],
            (1, 1, 1),
            (256, 1, 1),
        )
        self._gemv(self.wide, "mtp.fc.weight", self.hidden)
        self._rms(self.hidden, self._w(None, "input_layernorm.weight"), self.normed)
        self._attend(None, pos)
        self._rms(self.hidden, self._w(None, "post_attention_layernorm.weight"), self.normed)
        self._ffn(None)

    def _draft(self, token: int, pos: int, *, backbone: bool) -> int:
        """Draft the token after ``token``. ``backbone`` norms the language residual first.

        A later draft in the same chain leaves ``hidden`` as this step's ``mtp.norm``
        output, which is the input the next draft reads.
        """
        self.session.begin()
        if backbone:
            final = self._resolve(
                "model.language_model.norm.weight",
                "model.norm.weight",
            )
            self._rms(self.hidden, final, self.normed)
            self._copy_normed()
        self._mtp_body(token, pos)
        self._rms(self.hidden, "mtp.norm.weight", self.normed)
        self._copy_normed()
        head_name = self._resolve("lm_head.weight", "model.language_model.lm_head.weight")
        self._gemv(self.normed, head_name, self.logits)
        self.session.encode(
            "argmax_partial",
            [(self.logits, 0), (self.best_v, 0), (self.best_i, 0)],
            [_u32(VOCAB)],
            (_ARGMAX_GROUPS, 1, 1),
            (256, 1, 1),
        )
        self.session.end()
        return _best(self)

    def generate(
        self,
        prompt: list[int],
        max_tokens: int | None,
        stop: set[int],
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
        on_token: Any = None,
        on_prefill: Any = None,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        presence: float = 0.0,
        frequency: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        rng: np.random.Generator | None = None,
        mtp: bool = False,
    ) -> list[int]:
        self._hit_limit = False
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._top_k = int(top_k)
        self._presence = float(presence)
        self._frequency = float(frequency)
        self._logit_bias = dict(logit_bias or {})
        self._rng = rng or np.random.default_rng()
        self._sample_host = (
            self._temperature > 0.0
            or self._presence != 0.0
            or self._frequency != 0.0
            or bool(self._logit_bias)
        )
        self._output_ids = []
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._on_prefill = on_prefill
        self._prefill_noted = 0
        if not prompt:
            return []
        prompt = [int(token) for token in prompt]
        if prompt[-1] < 0:
            message = f"prompt length {len(prompt)} is past {self.max_pos}"
            raise RuntimeError(message)
        if not self.ensure(len(prompt)):
            reached = choose_wire(
                self._budget(),
                len(prompt),
                self._group,
                self._limit,
                mtp_available=self._record_mtp,
            ).positions
            message = f"prompt length {len(prompt)} is past {max(self.max_pos, reached)}"
            raise RuntimeError(message)
        self.ensure(len(prompt) + 1)
        self._status(f"prompt tokens {len(prompt)}")
        hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            self._status(f"prefix load {hit} tokens")
        self._status(f"cache hit {hit}" if hit else "cache miss")
        self._cached_tokens = hit
        suffix = prompt[hit:]
        prefill_row = None
        decode_row = None
        boundary = False
        out: list[int] = []
        try:
            from qwen38_27b.tower import bind_request

            bind_request(self, prompt, hit)
            prefill_row = self._begin_phase("prefill", len(suffix))
            self._prefill_progress = prefill_row
            started = time.perf_counter()
            if len(suffix) > 1:
                from qwen38_27b.prefill import prefill_ids

                prefill_ids(self, suffix[:-1], hit, follow=suffix[-1])
            nxt = self.step(suffix[-1], len(prompt) - 1, logits=True, rope=self._rope_at(len(prompt) - 1))
            assert nxt is not None
            counts: dict[int, int] = {}
            chosen = self._stable_draw([], self._pick(nxt, counts), counts)
            sampled_at = time.perf_counter()
            prefill_elapsed = max(sampled_at - started, 0.0)
            self._prefill_seconds = prefill_elapsed
            prefill_rate = 0.0 if prefill_elapsed <= 0 else len(suffix) / prefill_elapsed
            summary = (
                f"prefill {prefill_rate:.2f} tok/s "
                f"({len(suffix)} tok in {prefill_elapsed:.2f}s)"
            )
            self._prefill_progress = None
            self._end_phase(prefill_row, summary)
            print(summary, flush=True)
            prefill_row = None
            self.prefix.note(prompt)
            self._store_prefix()
            boundary = True
            self._note_prefill(len(suffix))
            pending = list(forced or [])
            close_after = bool(stop_after_forced) and bool(pending)
            room = max(0, self._limit - len(prompt))
            decode_total = room if max_tokens is None else max(0, min(max_tokens, room))
            decode_row = self._begin_phase("decode", decode_total, live=True)
            decode_started = sampled_at
            last_sample: float | None = None
            self._mtp_accepts = 0
            self._mtp_tries = 0
            use_mtp = bool(mtp) and bool(getattr(self, "drafts", False))
            emitted = 0
            rope_cursor = getattr(self, "_rope_next", None)

            def step_new(token: int, pos: int, *, logits: bool) -> int | None:
                nonlocal rope_cursor
                if rope_cursor is None:
                    return self.step(token, pos, logits=logits)
                angle = int(rope_cursor)
                rope_cursor += 1
                return self.step(token, pos, logits=logits, rope=(angle, angle, angle))
            if decode_total == 0:
                self._hit_limit = True
            while emitted < decode_total:
                self._check_stop()
                forced_token = False
                if pending:
                    chosen = int(pending.pop(0))
                    finished_forced = close_after and not pending
                    forced_token = True
                else:
                    finished_forced = False
                if chosen in stop or repeated_tail(out):
                    break
                out.append(chosen)
                counts[chosen] = counts.get(chosen, 0) + 1
                emitted += 1
                self._output_ids = list(out)
                if decode_row is not None:
                    decode_row.done = len(out)
                if on_token is not None:
                    on_token(chosen)
                pos = len(prompt) + emitted - 1
                at_cap = emitted == decode_total
                if finished_forced or at_cap:
                    step_new(chosen, pos, logits=False)
                    if at_cap:
                        self._hit_limit = True
                    break
                if pos + 1 >= self.max_pos:
                    if decode_row is not None:
                        decode_row.detail = "growing context"
                    try:
                        self.ensure(pos + 2)
                    finally:
                        if decode_row is not None:
                            decode_row.detail = ""
                    if pos + 1 >= self.max_pos:
                        self._hit_limit = True
                        break
                if repeated_tail(out):
                    step_new(chosen, pos, logits=False)
                    break
                speculated = False
                if use_mtp and not forced_token and pos > 0:
                    from qwen38_27b.speculate import DEPTH, draft_window, propose, verify

                    room = decode_total - emitted
                    need = pos + min(DEPTH, room)
                    if room > 0 and self.ensure(need) and self.drafts:
                        drafts = propose(self, chosen, pos - 1, min(DEPTH, room))
                        feeds, expected = draft_window(chosen, drafts, stop, room)
                        if expected:
                            produced = list(out)
                            sample_counts = dict(counts)

                            def draw() -> int:
                                self._fill_logits()
                                picked = sample_token(
                                    self._logits(),
                                    self._temperature,
                                    self._top_p,
                                    self._top_k,
                                    self._presence,
                                    self._frequency,
                                    sample_counts,
                                    self._logit_bias,
                                    self._rng,
                                )
                                return self._stable_draw(produced, picked, sample_counts)

                            def commit(token: int) -> None:
                                produced.append(int(token))
                                sample_counts[int(token)] = sample_counts.get(int(token), 0) + 1

                            accepted, nxt, keep = verify(
                                self,
                                pos,
                                feeds,
                                expected,
                                draw=draw if self._sample_host else None,
                                commit=commit if self._sample_host else None,
                            )
                            self._mtp_tries += len(expected)
                            self._mtp_accepts += accepted
                            if rope_cursor is not None:
                                rope_cursor += keep
                            emit = expected[: accepted - 1] if accepted == len(expected) else expected[:accepted]
                            halted = False
                            for token in emit:
                                if token in stop or repeated_tail(out):
                                    chosen = int(token)
                                    halted = True
                                    break
                                if emitted >= decode_total:
                                    halted = True
                                    break
                                out.append(int(token))
                                counts[int(token)] = counts.get(int(token), 0) + 1
                                emitted += 1
                                self._output_ids = list(out)
                                if decode_row is not None:
                                    decode_row.done = len(out)
                                if on_token is not None:
                                    on_token(int(token))
                            if not halted:
                                if self._sample_host:
                                    chosen = int(nxt)
                                else:
                                    chosen = self._stable_draw(out, self._pick(nxt, counts), counts)
                            last_sample = time.perf_counter()
                            speculated = True
                            if halted or emitted >= decode_total:
                                if emitted >= decode_total:
                                    self._hit_limit = True
                                break
                if speculated:
                    continue
                nxt_id = step_new(chosen, pos, logits=True)
                assert nxt_id is not None
                chosen = self._stable_draw(out, self._pick(nxt_id, counts), counts)
                last_sample = time.perf_counter()
            if self._mtp_tries:
                self._status(f"mtp {self._mtp_accepts}/{self._mtp_tries}")
            decode_elapsed = 0.0 if last_sample is None else max(last_sample - decode_started, 0.0)
            self._decode_seconds = decode_elapsed
            decode_rate = 0.0 if decode_elapsed <= 0 else len(out) / decode_elapsed
            decode_summary = (
                f"decode {decode_rate:.2f} tok/s ({len(out)} tok in {decode_elapsed:.2f}s)"
            )
            self._end_phase(decode_row, decode_summary)
            print(decode_summary, flush=True)
            decode_row = None
            self.prefix.note((prompt + out)[: self.filled])
            self._store_prefix()
        except RequestClosed:
            if boundary:
                self.prefix.note((prompt + out)[: self.filled])
                self._store_prefix()
            else:
                self.prefix.tokens = []
                self.prefix.dirty = False
                self.reset()
            raise
        finally:
            self._vision_images = None
            self._vision_prepared = None
            self._on_prefill = None
            self._prefill_progress = None
            self._end_phase(prefill_row, "")
            self._end_phase(decode_row, "")
        return out

    def _logits(self) -> np.ndarray:
        return self.session.download_f32(self.logits, VOCAB)

    def _pick(self, argmax_id: int, counts: dict[int, int]) -> int:
        if not self._sample_host:
            return int(argmax_id)
        return sample_token(
            self._logits(),
            self._temperature,
            self._top_p,
            self._top_k,
            self._presence,
            self._frequency,
            counts,
            self._logit_bias,
            self._rng,
        )

    def _tokenizer(self) -> Any:
        tokenizer = getattr(self, "_loaded_tokenizer", None)
        if tokenizer is None:
            from qwen38_27b.chat import load_tokenizer

            tokenizer = load_tokenizer(self.root)
            self._loaded_tokenizer = tokenizer
        return tokenizer

    def _stable_draw(self, produced: list[int], chosen: int, counts: dict[int, int]) -> int:
        """Redraw until the token is the encoding of the text it adds.

        The next prompt encodes the decoded completion. A token that would
        resegment the text already drawn is skipped, and the draw repeats.
        """
        from qwen38_27b.chat import keeps_token_ids

        tokenizer = self._tokenizer()
        if keeps_token_ids(tokenizer, produced, chosen):
            return int(chosen)
        raw = self._logits()
        first = int(chosen)
        banned: list[int] = []
        for _ in range(64):
            banned.append(int(chosen))
            scores = np.array(raw, dtype=np.float64, copy=True)
            scores[banned] = -np.inf
            if not np.isfinite(scores).any():
                return first
            chosen = sample_token(
                scores,
                self._temperature,
                self._top_p,
                self._top_k,
                self._presence,
                self._frequency,
                counts,
                self._logit_bias,
                self._rng,
            )
            if not np.isfinite(scores[int(chosen)]):
                return first
            if keeps_token_ids(tokenizer, produced, int(chosen)):
                return int(chosen)
        return first

    def _store_prefix(self) -> None:
        try:
            path = self.prefix.flush()
        except OSError as exc:
            self._status(f"prefix store failed {exc}")
            return
        if path is not None:
            self._status(f"prefix store {len(self.prefix.tokens)} tokens")

    def _status(self, line: str) -> None:
        console = self.console
        if console is None:
            print(line, flush=True)
            return
        console.log(line)

    def _begin_load(self, total: int) -> Any:
        console = self.console
        if console is None or total <= 0 or not getattr(console, "shows_bars", False):
            return None
        detail = getattr(console, "load_detail", "") or self.root.name
        return console.begin("load", total=total, unit="bytes", detail=detail)

    def _note_load(self, step: int) -> None:
        progress = self._load_progress
        if progress is None or step <= 0:
            return
        progress.done = min(progress.total, progress.done + step)

    def _begin_phase(self, phase: str, total: int, *, live: bool = False) -> Any:
        console = self.console
        if console is None:
            return None
        if live:
            if not getattr(console, "live_decode", False):
                return None
            # Unknown length. A fraction of the request budget is not a bar.
            total = 0
        elif total <= 0:
            return None
        detail = f"req #{self._request_id}" if self._request_id else ""
        return console.begin(
            phase,
            total=total,
            unit="tok",
            detail=detail,
            request=self._request_id or None,
        )

    def _end_phase(self, progress: Any, summary: str) -> None:
        if progress is None:
            return
        console = self.console
        if console is not None:
            console.end(progress, summary)

    def _note_prefill(self, done: int) -> None:
        progress = self._prefill_progress
        if progress is not None:
            capped = min(progress.total, done)
            if capped > progress.done:
                progress.done = capped
            noted = progress.done
        else:
            noted = done
        if noted <= getattr(self, "_prefill_noted", 0):
            return
        self._prefill_noted = noted
        hook = getattr(self, "_on_prefill", None)
        if hook is not None:
            hook(noted)


def repeated_tail(tokens: list[int], *, minimum: int = 16) -> bool:
    """True when the tail copies the span before it.

    The span is at least ``minimum`` tokens and at most 64. A greedy copy of
    the previous paragraph trips this and the completion stops.
    """
    count = len(tokens)
    for period in range(minimum, min(64, count // 2) + 1):
        if tokens[-period:] == tokens[-2 * period : -period]:
            return True
    return False


def sample_token(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    frequency: float,
    counts: dict[int, int],
    logit_bias: dict[int, float],
    rng: np.random.Generator,
) -> int:
    """One token from logits.

    OpenAI's penalty update is ``logit - count * frequency - presence`` when
    ``count > 0``. Temperature 0 is the argmax of those logits. A positive
    temperature divides the logits, keeps the top-k scores, then drops the
    nucleus tail (Holtzman et al.).
    """
    scores = np.asarray(logits, dtype=np.float64)
    if logit_bias or ((presence or frequency) and counts):
        scores = np.array(scores, copy=True)
        for token, bias in logit_bias.items():
            index = int(token)
            if 0 <= index < scores.size:
                scores[index] += float(bias)
        for token, count in counts.items():
            if count <= 0:
                continue
            index = int(token)
            if 0 <= index < scores.size:
                scores[index] -= float(count) * float(frequency) + float(presence)
    if temperature <= 0.0:
        return int(np.argmax(scores))
    if 0 < top_k < scores.size:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        chosen = scores[keep]
        return int(keep[_nucleus(chosen, temperature, top_p, rng)])
    return _nucleus(scores, temperature, top_p, rng)


def _nucleus(scores: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return int(np.argmax(scores))
    probs /= total
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        remove = cumulative > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        probs[order[remove]] = 0.0
        total = float(np.sum(probs))
        if total <= 0.0:
            return int(np.argmax(scores))
        probs /= total
    return int(rng.choice(probs.size, p=probs))


def _best(runtime: Qwen38Runtime) -> int:
    return _best_from(runtime, runtime.best_v, runtime.best_i)


def _best_from(runtime: Qwen38Runtime, values_buf: Any, index_buf: Any) -> int:
    values = runtime.session.download_f32(values_buf, _ARGMAX_GROUPS)
    raw = runtime.session.download(index_buf, _ARGMAX_GROUPS * 4)
    indices = np.frombuffer(raw, dtype=np.uint32).copy()
    choice = 0
    for index in range(1, _ARGMAX_GROUPS):
        better = values[index] > values[choice]
        tie = values[index] == values[choice] and int(indices[index]) < int(indices[choice])
        if better or tie:
            choice = index
    return int(indices[choice])


def _wipe(buffer: Any, nbytes: int) -> None:
    view = memoryview(buffer.contents().as_buffer(nbytes))
    chunk = b"\x00" * (1 << 20)
    offset = 0
    while offset < nbytes:
        end = min(offset + len(chunk), nbytes)
        view[offset:end] = chunk[: end - offset]
        offset = end


def _slots(layer_types: tuple[str, ...], kind: str) -> list[int]:
    slots = []
    seen = 0
    for name in layer_types:
        if name == kind:
            slots.append(seen)
            seen += 1
        else:
            slots.append(-1)
    return slots


def _default_layers() -> tuple[str, ...]:
    return tuple((["linear_attention"] * 3 + ["full_attention"]) * 16)


def layer_cache_ok() -> int:
    """One layer's planes, used by the size test."""
    return layer_cache_bytes(1)
