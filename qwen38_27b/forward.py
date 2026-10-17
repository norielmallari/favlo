"""Text forward. Weights are uniform fp4. The key/value cache is fp8.

Generation is an argmax. The sampling card is stored for the registry and is
not applied. The MTP block is packed and is not drafted. A text request does
not run the vision tower.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from qwen38_27b.attn import bind_plane, scratch_bytes
from qwen38_27b.cache import (
    available_bytes,
    cache_bytes,
    fit_positions,
    kv_mode,
    layer_cache_bytes,
    max_position,
    parse_memory,
    plane_bytes,
)
from qwen38_27b.layout import (
    AB,
    ATTN_OUT,
    EPS,
    FULL_LAYERS,
    HIDDEN,
    INTERMEDIATE,
    KV_PROJ,
    LINEAR_LAYERS,
    LINEAR_V,
    MIX,
    Q_HEADS,
    Q_PROJ,
    THETA,
    VOCAB,
    Z_WIDTH,
)
from qwen38_27b.prefill import workspace_bytes
from qwen38_27b.prefix import open_session
from qwen38_27b.session import Session


class RequestClosed(Exception):
    """The client went away or the process was signaled."""


_ARGMAX_GROUPS = (VOCAB + 255) // 256


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _file_buffer(session: Session, path: Path) -> Any:
    size = path.stat().st_size
    buffer = session.allocate(size)
    if buffer is None:
        message = f"{path.name} did not fit in a device buffer"
        raise RuntimeError(message)
    note = session._load_note
    view = memoryview(buffer.contents().as_buffer(size))
    with path.open("rb") as handle:
        offset = 0
        while offset < size:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
            if note is not None:
                note(len(chunk))
    if offset != size:
        message = f"{path.name} stopped at {offset} of {size}"
        raise OSError(message)
    return buffer


def startup_bytes(resident: int, positions: int) -> int:
    """Bytes copied or cleared before the first token: the record, the cache, and scratch."""
    one = plane_bytes(positions)
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
    state = LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4
    history = LINEAR_LAYERS * 3 * MIX * 2
    return resident + one * FULL_LAYERS * 2 + state + history + decode + scratch_bytes(positions)


def _fixed_bytes(positions: int) -> int:
    """Bytes besides the resident file and the key/value cache."""
    decode = (
        HIDDEN * 4 * 4
        + INTERMEDIATE * 4 * 4
        + Z_WIDTH * 4 * 2
        + ATTN_OUT * 4 * 3
        + KV_PROJ * 4 * 2
        + AB * 4 * 2
        + VOCAB * 4
        + _ARGMAX_GROUPS * 8
    )
    state = LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4
    history = LINEAR_LAYERS * 3 * MIX * 2
    return workspace_bytes() + scratch_bytes(positions) + decode + state + history


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
    ) -> None:
        del resident_percent
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
        self.layer_types = tuple(self.manifest.get("layer_types") or _default_layers())
        self.linear_slot = _slots(self.layer_types, "linear_attention")
        self.full_slot = _slots(self.layer_types, "full_attention")
        card_limit = max_position(self.root) if (self.root / "config.json").is_file() else 262144
        resident_path = self.root / "resident.bin"
        resident_bytes = resident_path.stat().st_size
        cap = parse_memory(max_memory)
        self.max_memory = available_bytes() if cap is None else cap
        self.max_pos = fit_positions(
            self.max_memory, resident_bytes, _fixed_bytes(card_limit)
        )
        self.max_pos = min(self.max_pos, card_limit)
        if self.max_pos < 2048:
            message = "the record does not fit a 2048-token cache"
            raise RuntimeError(message)
        self.filled = 0
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
        summary = ""
        self._load_progress = self._begin_load(startup_bytes(resident_bytes, self.max_pos))
        try:
            self.session = Session()
            self.session._load_note = self._note_load
            self.resident = _file_buffer(self.session, resident_path)
            zeros = self.session.zeros
            one = plane_bytes(self.max_pos)
            self.k_cache = [zeros(one) for _ in range(FULL_LAYERS)]
            self.v_cache = [zeros(one) for _ in range(FULL_LAYERS)]
            bind_plane(self)
            self.history = zeros(LINEAR_LAYERS * 3 * MIX * 2)
            self.state = zeros(LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4)
            self.hidden = zeros(HIDDEN * 4)
            self.normed = zeros(HIDDEN * 4)
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
            self._held = [self.resident, self.history, self.state, *self.k_cache, *self.v_cache]
            self.prefix = open_session(self)
            summary = (
                f"loaded {self.root.name} kv {self.kv_cache} context {self.max_pos} "
                f"weights {resident_bytes} cache {cache_bytes(self.max_pos)}"
            )
        finally:
            if self.session is not None:
                self.session._load_note = None
            progress = self._load_progress
            self._load_progress = None
            if progress is not None and self.console is not None:
                self.console.end(progress, summary)
            elif summary:
                self._status(summary)

    def close(self) -> None:
        prefix = getattr(self, "prefix", None)
        if prefix is not None and self.session is not None:
            try:
                prefix.flush()
            except OSError:
                pass
        self._prefill = None
        self._held.clear()
        self.resident = None
        self.session = None

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
            (self.history, LINEAR_LAYERS * 3 * MIX * 2),
            (self.state, LINEAR_LAYERS * LINEAR_V * 128 * 128 * 4),
        ):
            _wipe(buffer, nbytes)
        for cache in (*self.k_cache, *self.v_cache):
            _wipe(cache, plane_bytes(self.max_pos))
        _wipe(self.plane_tail, FULL_LAYERS * 4 * 32 * 256 * 4)
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
        return self.resident, int(entry["offset"]), entry

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

    def _attend(self, layer: int, pos: int) -> None:
        from qwen38_27b.attn import attend_plane

        q_w, q_s = self._pair(self.weight(layer, "self_attn.q_proj.weight"))
        k_w, k_s = self._pair(self.weight(layer, "self_attn.k_proj.weight"))
        v_w, v_s = self._pair(self.weight(layer, "self_attn.v_proj.weight"))
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
        self.session.encode(
            "split_q_gate_rows",
            [(self.wide, 0), (self.query, 0), (self.gate, 0)],
            [_u32(1)],
            (Q_HEADS, 1, 1),
            (256, 1, 1),
        )
        q_buf, q_off, _entry = self.locate_weight(layer, "self_attn.q_norm.weight")
        k_buf, k_off, _entry = self.locate_weight(layer, "self_attn.k_norm.weight")
        self.session.encode(
            "qk_norm_rope",
            [(self.query, 0), (self.kv, 0), (q_buf, q_off), (k_buf, k_off)],
            [_f32(THETA), _f32(EPS), _u32(pos)],
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
            slot=self.full_slot[layer],
        )
        self._gemv(self.y, self.weight(layer, "self_attn.o_proj.weight"), self.wide)
        self.session.encode(
            "residual_rows",
            [(self.hidden, 0), (self.wide, 0), (self.hidden, 0)],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )

    def _ffn(self, layer: int) -> None:
        gate_w, gate_s = self._pair(self.weight(layer, "mlp.gate_proj.weight"))
        up_w, up_s = self._pair(self.weight(layer, "mlp.up_proj.weight"))
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
        self._gemv(self.y, self.weight(layer, "mlp.down_proj.weight"), self.wide)
        self.session.encode(
            "residual_rows",
            [(self.hidden, 0), (self.wide, 0), (self.hidden, 0)],
            [_u32(HIDDEN)],
            (HIDDEN // 256, 1, 1),
            (256, 1, 1),
        )

    def _pair(self, name: str) -> tuple[tuple[Any, int], tuple[Any, int]]:
        buf, offset, entry = self.locate(name)
        return (buf, offset), (buf, self.scale_offset(entry, offset))

    def step(self, token: int, pos: int, *, logits: bool) -> int | None:
        if pos >= self.max_pos:
            message = f"position {pos} is past {self.max_pos}"
            raise RuntimeError(message)
        self.session.begin()
        self._embed(token)
        for layer, kind in enumerate(self.layer_types):
            self._rms(self.hidden, self.weight(layer, "input_layernorm.weight"), self.normed)
            if kind == "full_attention":
                self._attend(layer, pos)
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
        self.filled = pos + 1
        if not logits:
            return None
        return _best(self)

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        stop: set[int],
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
        on_token: Any = None,
        on_prefill: Any = None,
    ) -> list[int]:
        self._hit_limit = False
        self._output_ids = []
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._on_prefill = on_prefill
        self._prefill_noted = 0
        if not prompt:
            return []
        prompt = [int(token) for token in prompt]
        if prompt[-1] < 0 or len(prompt) > self.max_pos:
            message = f"prompt length {len(prompt)} is past {self.max_pos}"
            raise RuntimeError(message)
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
            prefill_row = self._begin_phase("prefill", len(suffix))
            self._prefill_progress = prefill_row
            started = time.perf_counter()
            if len(suffix) > 1:
                from qwen38_27b.prefill import prefill_ids

                prefill_ids(self, suffix[:-1], hit)
            nxt = self.step(suffix[-1], len(prompt) - 1, logits=True)
            assert nxt is not None
            self.prefix.note(prompt)
            self._store_prefix()
            boundary = True
            self._note_prefill(len(suffix))
            elapsed = max(time.perf_counter() - started, 1e-9)
            self._prefill_seconds = elapsed
            summary = (
                f"prefill {len(suffix) / elapsed:.2f} tok/s "
                f"({len(suffix)} tok in {elapsed:.2f}s)"
            )
            self._prefill_progress = None
            self._end_phase(prefill_row, summary)
            print(summary, flush=True)
            prefill_row = None
            pending = list(forced or [])
            close_after = bool(stop_after_forced) and bool(pending)
            decode_row = self._begin_phase("decode", max_tokens, live=True)
            decode_started = time.perf_counter()
            chosen = nxt
            for count in range(max_tokens):
                self._check_stop()
                if pending:
                    chosen = int(pending.pop(0))
                    finished_forced = close_after and not pending
                else:
                    finished_forced = False
                if chosen in stop:
                    break
                out.append(chosen)
                self._output_ids = list(out)
                if decode_row is not None:
                    decode_row.done = len(out)
                if on_token is not None:
                    on_token(chosen)
                if finished_forced or count + 1 == max_tokens:
                    self.step(chosen, len(prompt) + count, logits=False)
                    if count + 1 == max_tokens:
                        self._hit_limit = True
                    break
                if len(prompt) + count + 1 >= self.max_pos:
                    self._hit_limit = True
                    break
                chosen = self.step(chosen, len(prompt) + count, logits=True)
                assert chosen is not None
            decode_elapsed = max(time.perf_counter() - decode_started, 1e-9)
            self._decode_seconds = decode_elapsed
            decode_summary = (
                f"decode {len(out) / decode_elapsed:.2f} tok/s "
                f"({len(out)} tok in {decode_elapsed:.2f}s)"
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
            self._on_prefill = None
            self._prefill_progress = None
            self._end_phase(prefill_row, "")
            self._end_phase(decode_row, "")
        return out

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
        if console is None or total <= 0 or getattr(console, "keeps_lines", True):
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


def _best(runtime: Qwen38Runtime) -> int:
    values = runtime.session.download_f32(runtime.best_v, _ARGMAX_GROUPS)
    raw = runtime.session.download(runtime.best_i, _ARGMAX_GROUPS * 4)
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
