"""Text forward for one token at a time. The MTP block is packed and not drafted."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from qwen36_35b_a3b.atlas import ExpertAtlas, available_bytes
from qwen36_35b_a3b.attn import attend_plane, bind_plane, plane_bytes
from qwen36_35b_a3b.cache import (
    kv_buffer_bytes,
    kv_mode,
    kv_record_bytes,
    max_position,
    parse_memory,
)
from qwen36_35b_a3b.formats import EXPERT_IMAGE
from qwen36_35b_a3b.prefill import workspace_bytes
from qwen36_35b_a3b.session import Session, _clear

HIDDEN = 2048
VOCAB = 248320
HEAD = 256
Q_HEADS = 16
KV = 2
EPS = 1e-6
# Each head stores query then gate. The other split is the flat halves.
GATE_PER_HEAD = 1


def _reserved_bytes(resident: int, kv_one: int, plane: int) -> int:
    """Bytes allocated besides the expert atlas, including the chunk workspace."""
    scratch = (
        HIDDEN * 4
        + HIDDEN * 4
        + 8192 * 2
        + 8192 * 4
        + 4096 * 4
        + 32 * 4
        + 32 * 4
        + 8192 * 4
        + 4096 * 4
        + 4096 * 4
        + 4096 * 4
        + 512 * 4
        + 512 * 4
        + 4096 * 4
        + 2048 * 4
        + 8 * 1024 * 4
        + 8 * 512 * 4
        + 8 * HIDDEN * 4
        + 8 * 4
        + 8 * 4
        + HIDDEN * 4
        + 256 * 4
        + 4
        + VOCAB * 4
        + 30 * 3 * 8192 * 2
        + 30 * 32 * 128 * 128 * 4
    )
    return resident + scratch + workspace_bytes() + 20 * kv_one + plane


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _f32(value: float) -> bytes:
    return struct.pack("<f", value)


def _file_buffer(session: Session, path: Path) -> Any:
    size = path.stat().st_size
    buffer = session.device.newBufferWithLength_options_(max(size, 16), session.shared)
    view = memoryview(buffer.contents().as_buffer(size))
    with path.open("rb") as handle:
        offset = 0
        while offset < size:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
    return buffer


class Qwen36Runtime:
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
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            message = f"{self.root.name} has no manifest.json"
            raise FileNotFoundError(message)
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete"):
            message = f"{self.root.name} is not a finished record"
            raise RuntimeError(message)
        self.max_pos = max_position(self.root)
        self.kv_record = kv_record_bytes(self.kv_cache)
        self.kv_bytes = kv_buffer_bytes(self.kv_cache, self.max_pos)
        self.session = Session()
        self.resident = _file_buffer(self.session, self.root / "resident.bin")
        resident_bytes = (self.root / "resident.bin").stat().st_size
        plane = 0 if self.kv_cache == "f32" else plane_bytes(self.max_pos)
        reserved = _reserved_bytes(resident_bytes, self.kv_bytes, plane)
        cap = parse_memory(max_memory)
        self.max_memory = available_bytes() if cap is None else cap
        self.atlas = ExpertAtlas(
            self.session,
            self.root,
            max_memory=self.max_memory,
            reserved=reserved,
        )
        self.tensors: dict[str, dict] = self.manifest["tensors"]
        self.hidden = self.session.zeros(HIDDEN * 4)
        self.normed = self.session.zeros(HIDDEN * 4)
        self.half = self.session.zeros(8192 * 2)
        self.wide = self.session.zeros(8192 * 4)
        self.z = self.session.zeros(4096 * 4)
        self.ab = self.session.zeros(32 * 4)
        self.ab2 = self.session.zeros(32 * 4)
        self.mixed = self.session.zeros(8192 * 4)
        self.gdn_y = self.session.zeros(4096 * 4)
        self.query = self.session.zeros(4096 * 4)
        self.gate = self.session.zeros(4096 * 4)
        self.kv = self.session.zeros(512 * 4)
        self.vv = self.session.zeros(512 * 4)
        self.attn = self.session.zeros(4096 * 4)
        self.expert = self.session.zeros(2048 * 4)
        self.mid = self.session.zeros(8 * 1024 * 4)
        self.gated = self.session.zeros(8 * 512 * 4)
        self.partial = self.session.zeros(8 * HIDDEN * 4)
        self.ids = self.session.zeros(8 * 4)
        self.mix_w = self.session.zeros(8 * 4)
        self.acc = self.session.zeros(HIDDEN * 4)
        self.router = self.session.zeros(256 * 4)
        self.coeff = self.session.zeros(4)
        self.logits = self.session.zeros(VOCAB * 4)
        self.history = self.session.zeros(30 * 3 * 8192 * 2)
        self.state = self.session.zeros(30 * 32 * 128 * 128 * 4)
        self.k_cache = [self.session.zeros(self.kv_bytes) for _ in range(10)]
        self.v_cache = [self.session.zeros(self.kv_bytes) for _ in range(10)]
        if plane:
            bind_plane(self)
        self.filled = 0
        self._prefill = None
        self._cached: list[int] = []
        used = reserved + self.atlas.slots * self.atlas.file_bytes
        print(
            f"loaded {self.root.name} kv {self.kv_cache} context {self.max_pos} "
            f"atlas {self.atlas.slots}/40 memory {used}/{self.max_memory}",
            flush=True,
        )
        self._held = [
            self.resident,
            *self.atlas.buffers,
            self.hidden,
            self.history,
            self.state,
            *self.k_cache,
            *self.v_cache,
        ]

    def close(self) -> None:
        self._prefill = None
        self._held.clear()
        self.atlas.close()
        self.resident = None
        self.session = None

    def expert_buffer(self, layer: int) -> Any:
        """The layer file in the atlas. A miss finishes the open command buffer first."""
        found = self.atlas.cached(layer)
        if found is not None:
            return found
        restart = self.session._recording
        if restart:
            self.session.end()
        found = self.atlas.load(layer)
        if restart:
            self.session.begin()
        return found

    def reset(self) -> None:
        for buffer, nbytes in (
            (self.history, 30 * 3 * 8192 * 2),
            (self.state, 30 * 32 * 128 * 128 * 4),
        ):
            memoryview(buffer.contents().as_buffer(nbytes))[:] = b"\x00" * nbytes
        for cache in (*self.k_cache, *self.v_cache):
            _clear(cache, self.kv_bytes)
        self.filled = 0

    def _loc(self, name: str) -> tuple[Any, int, dict]:
        entry = self.tensors[name]
        return self.resident, int(entry["offset"]), entry

    def _scale(self, entry: dict, offset: int) -> int:
        return offset + int(entry["data_nbytes"]) + int(entry["scale_pad"])

    def _gemv_fp4(self, act: Any, act_off: int, name: str, dest: Any, dest_off: int = 0) -> None:
        buf, offset, entry = self._loc(name)
        rows, inner = entry["stored_shape"]
        self.session.encode(
            "gemv_fp4_vec",
            [(act, act_off), (buf, offset), (buf, self._scale(entry, offset)), (dest, dest_off)],
            [_u32(int(rows)), _u32(int(inner))],
            (int(rows), 1, 1),
            (32, 1, 1),
        )

    def _weight(self, name: str) -> tuple[tuple[Any, int], tuple[Any, int]]:
        buf, offset, entry = self._loc(name)
        return (buf, offset), (buf, self._scale(entry, offset))

    def _gemv_fp4_acc(self, act: Any, act_off: int, name: str, dest: Any) -> None:
        buf, offset, entry = self._loc(name)
        rows, inner = entry["stored_shape"]
        self.session.encode(
            "gemv_fp4_vec_acc",
            [(act, act_off), (buf, offset), (buf, self._scale(entry, offset)), (dest, 0)],
            [_u32(int(rows)), _u32(int(inner))],
            (int(rows), 1, 1),
            (32, 1, 1),
        )

    def _rms(
        self, src: Any, src_off: int, name: str, width: int, dest: Any, dest_off: int = 0
    ) -> None:
        buf, offset, _entry = self._loc(name)
        self.session.encode(
            "rmsnorm_f32",
            [(src, src_off), (buf, offset), (dest, dest_off)],
            [_u32(width), _f32(EPS)],
            (1, 1, 1),
            (256, 1, 1),
        )

    def _embed(self, token: int) -> None:
        buf, offset, entry = self._loc("model.language_model.embed_tokens.weight")
        _rows, inner = entry["stored_shape"]
        self.session.encode(
            "gather_fp4_row",
            [(buf, offset), (buf, self._scale(entry, offset)), (self.hidden, 0)],
            [_u32(token), _u32(int(inner))],
            (1, 1, 1),
            (32, 1, 1),
        )

    def _gdn(self, layer: int) -> None:
        base = f"model.language_model.layers.{layer}.linear_attn."
        qkv_w, qkv_s = self._weight(base + "in_proj_qkv.weight")
        z_w, z_s = self._weight(base + "in_proj_z.weight")
        a_w, a_s = self._weight(base + "in_proj_a.weight")
        b_w, b_s = self._weight(base + "in_proj_b.weight")
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
            (12352, 1, 1),
            (32, 1, 1),
        )
        slot = layer - (layer // 4)
        hist = slot * 3 * 8192 * 2
        weight, woff, _entry = self._loc(base + "conv1d.weight")
        self.session.encode(
            "gdn_conv_shift",
            [(self.history, hist), (self.wide, 0), (weight, woff), (self.mixed, 0)],
            [_u32(8192)],
            (8192, 1, 1),
            (1, 1, 1),
        )
        along, aoff, _entry = self._loc(base + "A_log")
        bias, boff, _entry = self._loc(base + "dt_bias")
        norm, noff, _entry = self._loc(base + "norm.weight")
        state = slot * 32 * 128 * 128 * 4
        self.session.encode(
            "gdn_state",
            [
                (self.state, state),
                (self.mixed, 0),
                (self.ab, 0),
                (self.ab2, 0),
                (self.z, 0),
                (along, aoff),
                (bias, boff),
                (norm, noff),
                (self.gdn_y, 0),
            ],
            [_f32(EPS)],
            (32, 1, 1),
            (128, 1, 1),
        )
        self._gemv_fp4_acc(self.gdn_y, 0, base + "out_proj.weight", self.hidden)

    def _attend(self, layer: int, pos: int) -> None:
        base = f"model.language_model.layers.{layer}.self_attn."
        q_w, q_s = self._weight(base + "q_proj.weight")
        k_w, k_s = self._weight(base + "k_proj.weight")
        v_w, v_s = self._weight(base + "v_proj.weight")
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
            (9216, 1, 1),
            (32, 1, 1),
        )
        self.session.encode(
            "split_q_gate",
            [(self.wide, 0), (self.query, 0), (self.gate, 0)],
            [_u32(GATE_PER_HEAD)],
            (4096, 1, 1),
            (1, 1, 1),
        )
        q_buf, q_off, _entry = self._loc(base + "q_norm.weight")
        k_buf, k_off, _entry = self._loc(base + "k_norm.weight")
        positions = np.array([pos, pos, pos], dtype=np.uint32).tobytes()
        self.session.encode(
            "qk_norm_rope",
            [
                (self.query, 0),
                (self.kv, 0),
                (q_buf, q_off),
                (k_buf, k_off),
            ],
            [positions, _f32(1e7), _f32(EPS)],
            (Q_HEADS + KV, 1, 1),
            (HEAD, 1, 1),
        )
        slot = layer // 4
        if self.kv_cache == "f32":
            stride = self.max_pos * HEAD
            self.session.encode(
                "store_kv",
                [
                    (self.k_cache[slot], 0),
                    (self.v_cache[slot], 0),
                    (self.kv, 0),
                    (self.vv, 0),
                ],
                [_u32(pos), _u32(stride)],
                (KV, 1, 1),
                (HEAD, 1, 1),
            )
            self.session.encode(
                "attend_gqa_heads",
                [
                    (self.query, 0),
                    (self.k_cache[slot], 0),
                    (self.v_cache[slot], 0),
                    (self.gate, 0),
                    (self.attn, 0),
                ],
                [_u32(pos + 1), _u32(stride)],
                (Q_HEADS, 1, 1),
                (HEAD, 1, 1),
            )
        else:
            attend_plane(
                self,
                query=self.query,
                key=self.kv,
                value=self.vv,
                gate=self.gate,
                dest=self.attn,
                pos0=pos,
                count=1,
                slot=slot,
            )
        self._gemv_fp4_acc(self.attn, 0, base + "o_proj.weight", self.hidden)

    def _moe(self, layer: int, prefix: str) -> None:
        router, roff, _entry = self._loc(prefix + "gate.weight")
        coeff, coff, _entry = self._loc(prefix + "shared_expert_gate.weight")
        self.session.encode(
            "router_coeff",
            [
                (self.normed, 0),
                (router, roff),
                (coeff, coff),
                (self.router, 0),
                (self.coeff, 0),
            ],
            [],
            (257, 1, 1),
            (32, 1, 1),
        )
        self.session.encode(
            "router8",
            [(self.router, 0), (self.ids, 0), (self.mix_w, 0)],
            [],
            (1, 1, 1),
            (1, 1, 1),
        )
        gate_w, gate_s = self._weight(prefix + "shared_expert.gate_up")
        down_w, down_s = self._weight(prefix + "shared_expert.down_proj.weight")
        experts = self.expert_buffer(layer)
        self.session.encode(
            "moe_gate_silu",
            [
                (self.normed, 0),
                (experts, 0),
                (self.ids, 0),
                gate_w,
                gate_s,
                (self.gated, 0),
                (self.expert, 0),
            ],
            [_u32(EXPERT_IMAGE)],
            (4608, 1, 1),
            (32, 1, 1),
        )
        self.session.encode(
            "moe_down_mix",
            [
                (self.gated, 0),
                (self.expert, 0),
                (experts, 0),
                (self.ids, 0),
                (self.mix_w, 0),
                down_w,
                down_s,
                (self.coeff, 0),
                (self.hidden, 0),
            ],
            [_u32(EXPERT_IMAGE)],
            (HIDDEN, 1, 1),
            (32, 1, 1),
        )

    def step(self, token: int, pos: int, *, logits: bool) -> np.ndarray | None:
        if pos >= self.max_pos:
            message = f"position {pos} is past {self.max_pos}"
            raise RuntimeError(message)
        self.session.begin()
        self._embed(token)
        for layer in range(40):
            norm = f"model.language_model.layers.{layer}.input_layernorm.weight"
            self._rms(self.hidden, 0, norm, HIDDEN, self.normed)
            if layer % 4 == 3:
                self._attend(layer, pos)
            else:
                self._gdn(layer)
            prefix = f"model.language_model.layers.{layer}.mlp."
            self._rms(
                self.hidden,
                0,
                f"model.language_model.layers.{layer}.post_attention_layernorm.weight",
                HIDDEN,
                self.normed,
            )
            self._moe(layer, prefix)
        self._rms(self.hidden, 0, "model.language_model.norm.weight", HIDDEN, self.normed)
        if logits:
            self._gemv_fp4(self.normed, 0, "lm_head.weight", self.logits)
        self.session.end()
        self.filled = pos + 1
        if not logits:
            return None
        return self.session.download_f32(self.logits, VOCAB)

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        stop: set[int],
        forced: list[int] | None = None,
        stop_after_forced: bool = False,
    ) -> list[int]:
        self._hit_limit = False
        if not prompt:
            return []
        prompt = [int(token) for token in prompt]
        print(f"prompt tokens {len(prompt)}", flush=True)
        cached = self._cached
        self._cached = []
        if cached and prompt[: len(cached)] == cached and len(prompt) > len(cached):
            hit = len(cached)
            print(f"cache hit {hit}", flush=True)
        else:
            hit = 0
            self.reset()
            print("cache miss", flush=True)
        suffix = prompt[hit:]
        started = time.perf_counter()
        if len(suffix) > 1:
            from qwen36_35b_a3b.prefill import prefill_ids

            prefill_ids(self, suffix[:-1], hit)
        logits = self.step(suffix[-1], len(prompt) - 1, logits=True)
        assert logits is not None
        elapsed = time.perf_counter() - started
        print(
            f"prefill {len(suffix) / elapsed:.2f} tok/s ({len(suffix)} tok in {elapsed:.2f}s)",
            flush=True,
        )
        pending = list(forced or [])
        close_after = bool(stop_after_forced) and bool(pending)
        seen: set[int] = set()
        out: list[int] = []
        decode_started = time.perf_counter()
        for count in range(max_tokens):
            if pending:
                nxt = int(pending.pop(0))
                finished_forced = close_after and not pending
            else:
                nxt = _sample(logits, temperature, top_p, top_k, presence, seen)
                finished_forced = False
            if nxt in stop:
                break
            out.append(nxt)
            seen.add(nxt)
            if finished_forced:
                self.step(nxt, len(prompt) + count, logits=False)
                break
            if count + 1 == max_tokens:
                self._hit_limit = True
                break
            logits = self.step(nxt, len(prompt) + count, logits=True)
            assert logits is not None
        decode_elapsed = max(time.perf_counter() - decode_started, 1e-9)
        print(
            f"decode {len(out) / decode_elapsed:.2f} tok/s "
            f"({len(out)} tok in {decode_elapsed:.2f}s)",
            flush=True,
        )
        self._cached = prompt + out
        return out


def _sample(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    seen: set[int],
) -> int:
    scores = logits.astype(np.float64, copy=True)
    if presence:
        for token in seen:
            scores[token] -= presence
    if temperature == 0.0:
        return int(np.argmax(scores))
    if top_k > 0 and top_k < scores.size:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        banned = np.full(scores.shape, -np.inf)
        banned[keep] = scores[keep]
        scores = banned
    scores = scores / temperature
    scores -= np.max(scores)
    probs = np.exp(scores)
    total = np.sum(probs)
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probs /= total
    if 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        running = np.cumsum(probs[order])
        cut = int(np.searchsorted(running, top_p) + 1)
        probs[order[cut:]] = 0.0
        probs /= np.sum(probs)
    return int(np.random.choice(scores.size, p=probs))
