"""One resident fp4 record and the fp8 cache. Prefill and decode stay on Metal."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from favlo.console import format_note, format_rate
from favlo.memory import memory
from favlo.prefix import (
    DEFAULT_BUDGET,
    PrefixSession,
    PrefixStore,
    checkpoint_digest,
    default_prefix_dir,
)

from spark_x26_1_7b.cache import Kv, _slot_bytes
from spark_x26_1_7b.closed import RequestClosed
from spark_x26_1_7b.layout import (
    CHUNK,
    CONTEXT,
    EOS,
    FULL,
    HEADS,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    LAYERS,
    Q_DIM,
    RMS_EPS,
    ROPE_FULL,
    ROPE_SLIDE,
    THETA_FULL,
    THETA_SLIDE,
    VOCAB,
    WINDOW,
    full_token_bytes,
    is_full,
)
from spark_x26_1_7b.prefix import CacheAdapter
from spark_x26_1_7b.sample import Sampler
from spark_x26_1_7b.session import Session, f32, u32

_QKV = Q_DIM + 2 * KV_DIM


@dataclass
class Plane:
    data: Any
    scale: Any
    rows: int
    inner: int
    kind: str


@dataclass
class Layer:
    qkv: Plane
    out: Plane
    gate: Plane
    up: Plane
    down: Plane
    g: Plane
    in_norm: Plane
    post_norm: Plane
    full: bool
    index: int


def parse_memory(text: str | int | None) -> int | None:
    """``32GiB``, ``512MiB``, or a byte count. Empty leaves the machine budget."""
    if text is None or text == "":
        return None
    if isinstance(text, int):
        if text < 1:
            message = f"maximum memory {text} is not positive"
            raise ValueError(message)
        return text
    raw = str(text).strip().replace("_", "")
    suffix = ""
    digits = raw
    for ending, name in (
        ("gib", "GiB"),
        ("mib", "MiB"),
        ("gb", "GiB"),
        ("mb", "MiB"),
        ("g", "GiB"),
        ("m", "MiB"),
    ):
        if raw.lower().endswith(ending):
            suffix = name
            digits = raw[: -len(ending)]
            break
    try:
        number = int(digits)
    except ValueError as exc:
        message = f"maximum memory {text} is not a byte count"
        raise ValueError(message) from exc
    if number < 1:
        message = f"maximum memory {text} is not positive"
        raise ValueError(message)
    if suffix == "GiB":
        return number << 30
    if suffix == "MiB":
        return number << 20
    return number


def _workspace_bytes(capacity: int) -> int:
    rows = CHUNK
    floats = (
        rows * HIDDEN * 2
        + rows * _QKV
        + rows * INTERMEDIATE * 3
        + rows * HIDDEN * 2
        + rows * HEADS
        + VOCAB
        + HEADS * capacity
        + VOCAB * 4
        + 256 * 8
        + (VOCAB // 256) * 2
    )
    return floats * 4 + rows * INTERMEDIATE * 2


def _minimum_cache() -> int:
    full = sum(_slot_bytes(2048, full=True))
    slide = sum(_slot_bytes(WINDOW, full=False))
    return len(FULL) * full + (LAYERS - len(FULL)) * slide


class SparkRuntime:
    """The open record. ``generate`` applies tokens and samples the next id on device."""

    def __init__(
        self,
        path: Path,
        *,
        kv_cache: str = "e4m3",
        console: Any = None,
        max_memory: str | int | None = None,
        owner: str | None = None,
        mtp: bool = False,
    ) -> None:
        del mtp
        if kv_cache != "e4m3":
            message = f"spark-x26-1-7b keeps an e4m3 cache; {kv_cache} is not built"
            raise RuntimeError(message)
        self.root = Path(path)
        self.console = console
        self._memory_cap = parse_memory(max_memory)
        self._stop: Any = None
        self._halted = False
        self._hit_limit = False
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._cached_tokens = 0
        self._output_ids: list[int] = []
        self.limit = 2048
        self.owner = owner or f"spark-x26-1-7b:{id(self)}"
        self.session = Session(self.owner)
        self.prefix: PrefixSession | None = None
        try:
            self._open()
        except Exception:
            self.session.close()
            raise

    def _open(self) -> None:
        manifest_path = self.root / "manifest.json"
        weight_path = self.root / "weights.bin"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            message = f"{self.root} is not a finished Spark record"
            raise RuntimeError(message)
        record = weight_path.stat().st_size
        need = record + _workspace_bytes(2048) + _minimum_cache()
        room = self._room()
        if need > room:
            message = f"spark-x26-1-7b needs {need} bytes and the budget holds {room}"
            raise RuntimeError(message)
        extra = room - need
        per = full_token_bytes() + HEADS * 4
        span = 2048 + extra // per
        span = min(CONTEXT, span - span % 32)
        if span < 2048:
            message = f"spark-x26-1-7b needs {need} bytes and the budget holds {room}"
            raise RuntimeError(message)
        self.limit = span
        rows = CHUNK
        self.hidden = self.session.allocate(rows * HIDDEN * 4)
        self.normed = self.session.allocate(rows * HIDDEN * 4)
        self.qkv = self.session.allocate(rows * _QKV * 4)
        self.gate_act = self.session.allocate(rows * INTERMEDIATE * 4)
        self.up_act = self.session.allocate(rows * INTERMEDIATE * 4)
        self.mid = self.session.allocate(rows * INTERMEDIATE * 4)
        self.half = self.session.allocate(rows * INTERMEDIATE * 2)
        self.attn = self.session.allocate(rows * HIDDEN * 4)
        self.proj = self.session.allocate(rows * HIDDEN * 4)
        self.gates = self.session.allocate(rows * HEADS * 4)
        self.logits = self.session.allocate(VOCAB * 4)
        self.kv = Kv(self.session, 2048)
        self.scores = self.session.allocate(HEADS * self.kv.capacity * 4, zero=True)
        self._score_cap = self.kv.capacity
        self.attn_partial = None
        self.attn_state = None
        self._tile_cap = 0
        self._bind_tiles()
        blob = weight_path.read_bytes()
        tensors = manifest["tensors"]
        self.embedding = self._plane(blob, tensors["embedding"])
        self.final_norm = self._plane(blob, tensors["final_norm"])
        self.layers = [self._layer(blob, tensors, index) for index in range(LAYERS)]
        store = PrefixStore(
            default_prefix_dir(),
            model="spark-x26-1-7b",
            digest=checkpoint_digest(self.root),
            tag="fp4-e4m3b",
            budget=DEFAULT_BUDGET,
        )
        self.prefix = PrefixSession(store, CacheAdapter(self))
        self.sampler = Sampler(self.session)

    def _plane(self, blob: bytes, spec: dict[str, Any]) -> Plane:
        data = blob[spec["data"] : spec["data"] + spec["data_bytes"]]
        kind = spec["kind"]
        if kind == "fp4":
            scale = blob[spec["scale"] : spec["scale"] + spec["scale_bytes"]]
            data_h, scale_h = memory.upload_group([data, scale], owner=self.owner)
            return Plane(data_h, scale_h, int(spec["rows"]), int(spec["inner"]), kind)
        handle = self.session.upload(data)
        return Plane(handle, None, int(spec["rows"]), int(spec["inner"]), kind)

    def _layer(self, blob: bytes, tensors: dict[str, Any], index: int) -> Layer:
        prefix = f"layer.{index}."
        return Layer(
            qkv=self._plane(blob, tensors[prefix + "qkv"]),
            out=self._plane(blob, tensors[prefix + "out"]),
            gate=self._plane(blob, tensors[prefix + "gate"]),
            up=self._plane(blob, tensors[prefix + "up"]),
            down=self._plane(blob, tensors[prefix + "down"]),
            g=self._plane(blob, tensors[prefix + "g"]),
            in_norm=self._plane(blob, tensors[prefix + "in_norm"]),
            post_norm=self._plane(blob, tensors[prefix + "post_norm"]),
            full=is_full(index),
            index=index,
        )

    def _room(self) -> int:
        ceiling = memory.budget()
        if self._memory_cap is not None:
            ceiling = min(ceiling, self._memory_cap)
        return max(0, ceiling - memory.live - memory.pending)

    def close(self) -> None:
        prefix = self.prefix
        if prefix is not None:
            try:
                prefix.flush()
            except OSError:
                pass
        self.session.close()

    def reset(self) -> None:
        self.kv.length = 0
        self.sampler.clear()
        if self.prefix is not None:
            self.prefix.tokens = []
            self.prefix.dirty = False
            self.prefix.loaded = False

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
        seed: int | None = None,
    ) -> list[int]:
        self._hit_limit = False
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._output_ids = []
        self.sampler.reseed(0 if seed is None else int(seed))
        prompt = [int(token) for token in prompt]
        if not prompt:
            return []
        if len(prompt) > self.limit:
            message = f"prompt length {len(prompt)} is past {self.limit}"
            raise RuntimeError(message)
        self._status(format_note("prompt", tokens=len(prompt)))
        assert self.prefix is not None
        hit = self.prefix.align(prompt)
        if self.prefix.loaded:
            self._status(format_note("prefix_load", tokens=hit))
        self._status(format_note("cache", hit=hit))
        self._cached_tokens = hit
        suffix = prompt[hit:]
        if not suffix:
            message = "the prompt is already applied and has no next token to score"
            raise RuntimeError(message)
        boundary = False
        out: list[int] = []
        try:
            started = time.perf_counter()
            self._prefill(suffix, hit, True)
            self.sampler.clear()
            pending = list(forced or [])
            forced_now = bool(pending)
            if pending:
                chosen = int(pending.pop(0))
                self.sampler.observe(chosen)
                self.session.end()
            else:
                chosen = self._draw(temperature, top_p, top_k, presence, frequency)
            elapsed = max(time.perf_counter() - started, 0.0)
            self._prefill_seconds = elapsed
            print(format_rate("prefill", len(suffix), elapsed), flush=True)
            self.prefix.note(prompt)
            self._store_prefix()
            boundary = True
            if on_prefill is not None:
                on_prefill(len(suffix))
            room = max(0, self.limit - len(prompt))
            decode_total = room if max_tokens is None else max(0, min(int(max_tokens), room))
            decode_started = time.perf_counter()
            last_sample: float | None = None
            emitted = 0
            close_after = bool(stop_after_forced) and bool(forced)
            if decode_total == 0:
                self._hit_limit = True
            while emitted < decode_total:
                self._check_stop()
                if emitted > 0:
                    if pending:
                        chosen = int(pending.pop(0))
                        self.sampler.observe(chosen)
                        forced_now = True
                    else:
                        forced_now = False
                finished_forced = close_after and forced_now and not pending
                if chosen in stop:
                    break
                out.append(chosen)
                self._output_ids = list(out)
                emitted += 1
                if on_token is not None:
                    on_token(chosen)
                pos = len(prompt) + emitted - 1
                at_cap = emitted == decode_total
                if finished_forced or at_cap or chosen in stop:
                    self._forward([chosen], pos, False)
                    if at_cap:
                        self._hit_limit = True
                    break
                self.session.begin()
                self._encode([chosen], pos, True)
                self.prefix.note(prompt + out)
                chosen = self._draw(temperature, top_p, top_k, presence, frequency)
                last_sample = time.perf_counter()
            decode_elapsed = 0.0 if last_sample is None else max(last_sample - decode_started, 0.0)
            self._decode_seconds = decode_elapsed
            print(format_rate("decode", len(out), decode_elapsed), flush=True)
            self.prefix.note((prompt + out)[: self.kv.length])
            self._store_prefix()
        except RequestClosed:
            if self.session.active:
                self.session.end()
            if boundary:
                self.prefix.note((prompt + out)[: self.kv.length])
                self._store_prefix()
            else:
                self.prefix.tokens = []
                self.prefix.dirty = False
                self.kv.length = 0
            raise
        finally:
            if self.session.active:
                self.session.end()
        return out

    def _draw(self, temperature: float, top_p: float, top_k: int, presence: float, frequency: float) -> int:
        return self.sampler.draw(
            self.logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
        )

    def _prefill(self, tokens: list[int], pos: int, logits: bool) -> None:
        if not self._ensure(pos + len(tokens) - 1):
            message = f"spark-x26-1-7b context holds {self.limit} tokens"
            raise RuntimeError(message)
        self.session.begin()
        index = 0
        while index < len(tokens):
            left = len(tokens) - index
            take = CHUNK if left >= CHUNK else left
            last = index + take == len(tokens)
            self._encode(tokens[index : index + take], pos + index, logits and last)
            index += take
            if not last:
                self.session.end()
                self._check_stop()
                self.session.begin()
        if not logits:
            self.session.end()

    def _forward(self, tokens: list[int], pos: int, logits: bool) -> None:
        if not self._ensure(pos + len(tokens) - 1):
            message = f"spark-x26-1-7b context holds {self.limit} tokens"
            raise RuntimeError(message)
        self.session.begin()
        self._encode(tokens, pos, logits)
        self.session.end()

    def _encode(self, tokens: list[int], pos: int, logits: bool) -> None:
        count = len(tokens)
        if count < 1 or count > CHUNK:
            message = f"a Spark step is 1 to {CHUNK} tokens, not {count}"
            raise RuntimeError(message)
        physical = 1 if count == 1 else CHUNK
        if physical == CHUNK and count < CHUNK:
            tail = physical * HIDDEN
            groups = (tail + 255) // 256
            self.session.launch(
                "zero_f32",
                [self.hidden],
                [u32(tail)],
                (groups, 1, 1),
                (256, 1, 1),
            )
        for row, token in enumerate(tokens):
            self.session.launch(
                "embed_row",
                [self.embedding.data, self.embedding.scale, self.hidden],
                [u32(HIDDEN)],
                (1, 1, 1),
                (32, 1, 1),
                offsets=[int(token) * (HIDDEN // 2), int(token) * (HIDDEN // 32), row * HIDDEN * 4],
            )
        for layer in self.layers:
            self._layer_step(layer, count, physical, pos)
        if logits:
            if count == 1:
                self.session.launch(
                    "gemv_group_rms",
                    [self.hidden, self.final_norm.data, self.embedding.data, self.embedding.scale, self.logits, self.normed],
                    [u32(HIDDEN), f32(RMS_EPS)],
                    (VOCAB // 32, 1, 1),
                    (256, 1, 1),
                )
            else:
                self.session.launch(
                    "rmsnorm",
                    [self.hidden, self.final_norm.data, self.normed],
                    [u32(HIDDEN), f32(RMS_EPS)],
                    (1, 1, 1),
                    (32, 1, 1),
                    offsets=[(count - 1) * HIDDEN * 4, 0, 0],
                )
                self.session.launch(
                    "gemv_group",
                    [self.normed, self.embedding.data, self.embedding.scale, self.logits],
                    [u32(HIDDEN)],
                    (VOCAB // 32, 1, 1),
                    (256, 1, 1),
                )
        self.kv.length = pos + count

    def _layer_step(self, layer: Layer, count: int, physical: int, pos: int) -> None:
        if physical == 1:
            self._gemv_rms(self.hidden, layer.in_norm, layer.qkv, self.qkv)
        else:
            self._rmsnorm(self.hidden, layer.in_norm, self.normed, physical)
            self._linear(self.normed, physical, layer.qkv, self.qkv)
        self._rope(pos, count, layer.full)
        self._store_span(pos, count, layer)
        self._gproj(count, layer)
        self._attend_span(pos, count, layer)
        if physical > count:
            tail = (physical - count) * HIDDEN
            groups = (tail + 255) // 256
            self.session.launch(
                "zero_f32",
                [self.attn],
                [u32(tail)],
                (groups, 1, 1),
                (256, 1, 1),
                offsets=[count * HIDDEN * 4],
            )
        if physical == 1:
            self._gemv_add(self.attn, layer.out, self.hidden)
            self._gemv_rms(self.hidden, layer.post_norm, layer.gate, self.gate_act)
            self._gemv_rms(self.hidden, layer.post_norm, layer.up, self.up_act)
            self._geglu(INTERMEDIATE)
            self._gemv_add(self.mid, layer.down, self.hidden)
            return
        self._linear(self.attn, physical, layer.out, self.proj)
        self._residual(self.hidden, self.proj, physical)
        self._rmsnorm(self.hidden, layer.post_norm, self.normed, physical)
        self._linear(self.normed, physical, layer.gate, self.gate_act)
        self._linear(self.normed, physical, layer.up, self.up_act)
        self._geglu(physical * INTERMEDIATE)
        self._linear(self.mid, physical, layer.down, self.proj)
        self._residual(self.hidden, self.proj, physical)

    def _geglu(self, width: int) -> None:
        groups = (width + 255) // 256
        self.session.launch(
            "geglu",
            [self.gate_act, self.up_act, self.mid],
            [u32(width)],
            (groups, 1, 1),
            (256, 1, 1),
        )

    def _gemv_rms(self, source: Any, norm: Plane, plane: Plane, dest: Any) -> None:
        self.session.launch(
            "gemv_fp4_vec_rms",
            [source, norm.data, plane.data, plane.scale, dest, self.normed],
            [u32(plane.inner), f32(RMS_EPS)],
            (plane.rows // 4, 1, 1),
            (32, 1, 1),
        )

    def _gemv_add(self, source: Any, plane: Plane, hidden: Any) -> None:
        self.session.launch(
            "gemv_fp4_vec_add",
            [source, plane.data, plane.scale, hidden],
            [u32(plane.inner)],
            (plane.rows // 4, 1, 1),
            (32, 1, 1),
        )

    def _linear(self, source: Any, count: int, plane: Plane, dest: Any) -> None:
        if count == CHUNK:
            self._cast(source, count * plane.inner)
            self.session.launch(
                "matmul_fp4_32",
                [self.half, plane.data, plane.scale, dest],
                [u32(count), u32(plane.rows), u32(plane.inner)],
                (plane.rows // 128, count // 32, 1),
                (128, 1, 1),
            )
            return
        self.session.launch(
            "gemv_fp4",
            [source, plane.data, plane.scale, dest],
            [u32(plane.inner)],
            (plane.rows // 4, 1, 1),
            (32, 1, 1),
        )

    def _cast(self, source: Any, count: int) -> None:
        groups = (count + 255) // 256
        self.session.launch(
            "cast_f16",
            [source, self.half],
            [u32(count)],
            (groups, 1, 1),
            (256, 1, 1),
        )

    def _rmsnorm(self, source: Any, plane: Plane, dest: Any, count: int) -> None:
        self.session.launch(
            "rmsnorm",
            [source, plane.data, dest],
            [u32(HIDDEN), f32(RMS_EPS)],
            (count, 1, 1),
            (32, 1, 1),
        )

    def _residual(self, hidden: Any, update: Any, count: int) -> None:
        self.session.launch(
            "residual",
            [hidden, update],
            [u32(HIDDEN)],
            (count, 1, 1),
            (32, 1, 1),
        )

    def _rope(self, pos: int, count: int, full: bool) -> None:
        width = ROPE_FULL if full else ROPE_SLIDE
        theta = THETA_FULL if full else THETA_SLIDE
        self.session.launch(
            "rope_qk",
            [self.qkv],
            [struct.pack("<IIf", pos, width, theta)],
            (10, count, 1),
            (128, 1, 1),
        )

    def _store_span(self, pos: int, count: int, layer: Layer) -> None:
        slot = self.kv.slots[layer.index]
        capacity = slot.capacity if layer.full else WINDOW
        packed = struct.pack("<III", pos, capacity, count)
        keys = "store_full_keys" if layer.full else "store_slide_keys"
        values = "store_full_span" if layer.full else "store_slide_span"
        if count > 1:
            self.session.launch(
                keys,
                [self.qkv, slot.key, slot.key_scale],
                [packed],
                (count, 1, 1),
                (32, 1, 1),
            )
        self.session.launch(
            values,
            [self.qkv, slot.key, slot.key_scale, slot.value, slot.value_scale, slot.scratch],
            [packed],
            (1, 1, 1),
            (32, 1, 1),
        )

    def _attend_span(self, pos: int, count: int, layer: Layer) -> None:
        slot = self.kv.slots[layer.index]
        capacity = slot.capacity if layer.full else WINDOW
        window = 0 if layer.full else WINDOW
        cfg = struct.pack("<IIIII", pos, capacity, window, self.kv.capacity, count)
        nkeys = pos + count
        if window != 0:
            nkeys = min(nkeys, window)
        if count == 1 and nkeys > 64:
            used = (nkeys + 63) // 64
            self.session.launch(
                "attend_tile",
                [self.qkv, slot.key, slot.key_scale, slot.value, slot.value_scale, self.attn_partial, self.attn_state],
                [cfg, u32(self._tile_cap)],
                (HEADS, used, 1),
                (32, 1, 1),
            )
            self.session.launch(
                "attend_merge",
                [self.attn_partial, self.attn_state, self.gates, self.attn],
                [u32(self._tile_cap), u32(used)],
                (HEADS, 1, 1),
                (32, 1, 1),
            )
            return
        self.session.launch(
            "attend_span",
            [self.qkv, slot.key, slot.key_scale, slot.value, slot.value_scale, self.gates, self.attn],
            [cfg],
            (HEADS, count, 1),
            (32, 1, 1),
        )

    def _gproj(self, count: int, layer: Layer) -> None:
        self.session.launch(
            "gemv_f16",
            [self.normed, layer.g.data, self.gates],
            [u32(HIDDEN), u32(HEADS)],
            (HEADS, count, 1),
            (32, 1, 1),
        )

    def _ensure(self, pos: int) -> bool:
        if pos + 1 > self.limit:
            return False
        if pos < self.kv.capacity:
            return True
        capacity = self.kv.capacity
        while capacity <= pos:
            nxt = min(self.limit, capacity * 2)
            if nxt <= capacity:
                return False
            capacity = nxt
        try:
            self.kv._grow(capacity)
        except (RuntimeError, MemoryError):
            self.limit = self.kv.capacity
            return pos < self.kv.capacity
        self._bind_scores()
        self._bind_tiles()
        return True

    def _bind_scores(self) -> None:
        if self.scores is not None and self._score_cap == self.kv.capacity:
            return
        if self.scores is not None:
            memory.release(self.scores)
        self.scores = self.session.allocate(HEADS * self.kv.capacity * 4, zero=True)
        self._score_cap = self.kv.capacity

    def _bind_tiles(self) -> None:
        tiles = (self.kv.capacity + 63) // 64
        if self._tile_cap == tiles and self.attn_partial is not None:
            return
        if self.attn_partial is not None:
            memory.release(self.attn_partial)
            memory.release(self.attn_state)
        self.attn_partial = self.session.allocate(HEADS * tiles * KV_DIM * 4)
        self.attn_state = self.session.allocate(HEADS * tiles * 2 * 4)
        self._tile_cap = tiles

    def _check_stop(self) -> None:
        fn = self._stop
        if callable(fn) and fn():
            raise RequestClosed()

    def _store_prefix(self) -> None:
        assert self.prefix is not None
        try:
            path = self.prefix.flush()
        except OSError as exc:
            self._status(format_note("prefix_store_failed", error=exc))
            return
        if path is not None:
            self._status(format_note("prefix_store", tokens=len(self.prefix.tokens)))

    def _status(self, line: str) -> None:
        if self.console is None:
            print(line, flush=True)
            return
        self.console.log(line)


__all__ = ["EOS", "SparkRuntime", "parse_memory"]
