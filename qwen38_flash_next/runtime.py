"""Qwen3.8-Flash-Next forward on Metal."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import objc

from qwen38_flash_next.checks import _ngram_row, _sample, ngram_rows
from qwen38_flash_next.context import (
    BLOCK as INDEX_BLOCK,
)
from qwen38_flash_next.context import (
    DENSE_LIMIT,
    INDEX_STRIDE,
    PREFILL_CHUNK,
    atlas_budget,
    atlas_fallbacks,
    cache_growth_bytes,
    kv_bytes_per_token,
    machine_memory,
    process_footprint,
    resident_candidates,
    scratch_bytes,
)
from qwen38_flash_next.convert import ShardStore, _layer_f32_keys, _layer_int4_keys
from qwen38_flash_next.experts import ExpertAtlas, group_tier
from qwen38_flash_next.formats import (
    BLOCK,
    DEFAULT_KV_CACHE,
    INDEX_BF16_BYTES,
    KV_BF16_BYTES,
    KV_E4M3_BYTES,
    KV_TAIL_TOKENS,
    NGRAM_WIDTH,
    PAGE,
    ExpertPlanes,
    expert_planes,
    int4_plane_sizes,
    kv_cache_name,
    stored_k_int4,
)
from qwen38_flash_next.prefix_store import (
    DEFAULT_BUDGET,
    AttentionImage,
    FixedImage,
    PrefixHeader,
    PrefixState,
    PrefixStore,
    checkpoint_digest,
    default_prefix_dir,
    stored_bytes,
)
from qwen38_flash_next.record import QuantRecord, _Bf16Ngram, prefill_image_table
from qwen38_flash_next.session import (
    Dev,
    F16Matrix,
    Int4Matrix,
    MetalSession,
    _buffer_bytes,
    _buffer_slice,
    _buffer_view,
    _check_f32_gemv,
    _device_name,
    _fill_buffer,
    _log,
    _tok_s,
    _write_u32,
    commit_pages,
    discard_buffer,
    release_buffer,
)
from qwen38_flash_next.tools import (
    answer_repeat_id,
    canonical_continuation,
    continuing_ids,
    forced_call_options,
    function_name_options,
    looped_span,
    normalize_messages,
    penalty_ids,
    repeated_cycle,
    stuck_ids,
)


def mtp_starts(sequence: int, *, loaded: bool, limit: int = PREFILL_CHUNK) -> bool:
    """Drafts start whenever the MTP layer is loaded.

    One token cannot clear the rate a kept span reaches, so a long sequence
    drafts too. The rate window turns drafts off when they fall behind.
    ``limit`` stays for callers that still pass the prefill chunk.
    """
    if sequence < 0 or limit < 0:
        return False
    return bool(loaded)


def mtp_holds(rates: Sequence[float], *, window: int = 4, ratio: float = 0.5) -> bool:
    """The latest window is still at least ``ratio`` of the best window.

    Fewer than two windows has no earlier rate to fall behind, so drafts stay.
    """
    if window < 1 or len(rates) < window * 2:
        return True
    means = [
        sum(rates[index : index + window]) / window
        for index in range(len(rates) - window + 1)
    ]
    best = max(means)
    if best <= 0:
        return True
    return means[-1] >= best * ratio


def _growth_alloc(buf: Any, need: int, stride: int) -> int:
    """Bytes ``_reserve_bytes`` will request when ``buf`` is shorter than ``need``."""
    if need <= 0 or stride <= 0:
        return 0
    if buf is not None and int(buf.length()) >= need:
        return 0
    chunk = PREFILL_CHUNK * stride
    slack = 64 * stride
    return max(need * 2, stride) if need <= chunk else need + slack


def after_atlas_load(resident: Sequence[int]) -> int:
    """First decode step after the resident count stops growing.

    The steps before that index are the atlas fill. A later step may add a
    resident again when a hole is reused; the fill still ended at the first
    plateau.
    """
    for index in range(1, len(resident)):
        if resident[index] <= resident[index - 1]:
            return index
    return len(resident)


def span_expert_bits(hot: int, cold: int, super_bits: int) -> bool:
    """Short-span decode dots are fp8 E4M3 (8) and fp4 E2M1 (4)."""
    return hot in (4, 8) and cold in (4, 8) and super_bits in (4, 8)


class RequestClosed(Exception):
    """The client closed the socket while a forward was still running."""


def _poke_u32(buffer: Any, index: int, value: int) -> None:
    start = index * 4
    _buffer_view(buffer, start + 4)[start : start + 4] = int(value).to_bytes(4, "little")


class Qwen38Runtime:
    """One sequence of Qwen3.8-Flash-Next. Linear maps are Metal shader dots."""

    def __init__(
        self,
        root: Path,
        *,
        bf16: bool = False,
        resident_percent: float | None = None,
        kv_cache: str = DEFAULT_KV_CACHE,
        console: Any = None,
    ) -> None:
        self._closed = False
        self.console = console
        self._prefill_progress = None
        self._decode_progress = None
        self._decode_reported = True
        self._load_progress = None
        self._request_id = None
        self._cached_tokens = 0
        self._prompt_tokens = 0
        self._prefill_tokens = 0
        self._prefill_seconds = 0.0
        self.gpu = None
        self.store = None
        self.files = None
        try:
            self._open(root, bf16=bf16, resident_percent=resident_percent, kv_cache=kv_cache)
        except BaseException:
            self.close()
            raise

    def _open(
        self,
        root: Path,
        *,
        bf16: bool = False,
        resident_percent: float | None = None,
        kv_cache: str = DEFAULT_KV_CACHE,
    ) -> None:
        self.gpu = MetalSession()
        _log(f"shaders ready on {_device_name(self.gpu.device)}")
        self.root = root
        config = json.loads((root / "config.json").read_text())
        text = config["text_config"]
        self.hidden = int(text["hidden_size"])
        self.layers = int(text["num_hidden_layers"])
        self.experts = int(text["num_experts"])
        self.top_k = int(text["num_experts_per_tok"])
        self.moe = int(text["moe_intermediate_size"])
        self.vocab = int(text["vocab_size"])
        self.eos = int(text["eos_token_id"])
        self.stop_ids = {self.eos}
        self.eps = float(text["rms_norm_eps"])
        rope = text["rope_parameters"]
        self.rope_theta = float(rope["rope_theta"])
        self.rotary_dim = int(256 * float(rope["partial_rotary_factor"]))
        self.kinds = list(text["layer_types"])
        self.ple_layers = {int(item) for item in text["ple_layer_ids"]}
        self.branches = int(text["hc_count"])
        self.heads_per_ngram = int(text["heads_per_ngram"])
        self.ngram_size = int(text["ngram_size"])
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self._bf16 = bf16
        self.kv_cache = kv_cache_name(kv_cache)
        self.store: ShardStore | None = None
        self.files: QuantRecord | _Bf16Ngram
        self.weights: dict[str, Int4Matrix] = {}
        self.f16_weights: dict[str, F16Matrix] = {}
        self.f32: dict[str, np.ndarray] = {}
        self.i64: dict[str, np.ndarray] = {}
        self._embed_meta: dict[str, int] = {}
        self._resident: Any = None
        self._f32_meta: dict[str, dict[str, Any]] = {}
        self._f32_uploaded: dict[str, Any] = {}
        self.expert_cache: dict[tuple[int, int], tuple[Int4Matrix, Int4Matrix]] = {}
        self._choice: dict[int, Dev] = {}
        self._choice_ids = self.gpu.alloc(self.layers * self.top_k * 4)
        self._tier: Any = None
        self._atlas_trusted = False
        self._avoid_token: int | None = None
        self._avoid_until = 0
        self._block_tokens: set[int] = set()
        self._answer_mode = False
        self._presence_penalty = 0.0
        self._frequency_penalty = 0.0
        self._penalty_window = 20
        self._predicted: dict[int, list[int]] = {}
        self._span_rows = 1
        self._prefetch_events: dict[int, int] = {}
        self._prefetch_protect: set[tuple[int, int]] = set()
        self._residual_buf: Any = None
        self._residual_n = 0
        self._decoding = False
        self._force_sync = False
        self._stop = None
        self.atlas: ExpertAtlas | None = None
        self._streaming_prefill = False
        self._stage: list[Any] | None = None
        self._stage_table: Any = None
        self._stage_miss: Any = None
        self._stage_event = [0, 0]
        self._stage_bytes = 0
        self._hot_keep: list[Any] | None = None
        self._hot_fence = 0
        self._hold_xs: Any = None
        self._hold_out: Any = None
        self._hold_ids: Any = None
        self._hold_w: Any = None
        self._hold_cap = 0
        self._hot = expert_planes(4, self.hidden, self.moe)
        self._cold = expert_planes(2, self.hidden, self.moe)
        self._super_planes = expert_planes(8, self.hidden, self.moe)
        self._super_images: dict[tuple[int, int], Any] = {}
        self._int2_ids: Any = None
        self._int2_table: Any = None
        self._int2_miss: Any = None
        self._int2_one: Any = None
        self.tokens: list[int] = []
        self.last_logits: Dev | None = None
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self._decode_steps: list[tuple[float, float, int]] = []
        self._output_ids: list[int] = []
        self.rng = np.random.default_rng()
        self.tool_names: list[str] = []
        self.required_function: str | None = None
        self.required_closes = False
        self._token_pieces: list[str] | None = None
        self.gdn_state = {
            index: self.gpu.zeros(48 * 128 * 128 * 4)
            for index, kind in enumerate(self.kinds)
            if kind == "linear_attention"
        }
        self._conv_bytes = 3 * 10240 * 4
        self.conv_hist = {
            index: self.gpu.zeros(self._conv_bytes) for index in self.gdn_state
        }
        self._ple_span = (4 - 1) * self.ngram_size
        self._ple_bytes = self._ple_span * 10240 * 4
        self.ple_hist = self.gpu.zeros(self._ple_bytes)
        attention = [
            index for index, kind in enumerate(self.kinds) if kind == "full_attention"
        ]
        self.kv_len = dict.fromkeys(attention, 0)
        body = KV_BF16_BYTES if self.kv_cache == "bf16" else KV_E4M3_BYTES
        tail = KV_TAIL_TOKENS * KV_BF16_BYTES
        self.kv_k = {index: self.gpu.alloc(body) for index in attention}
        self.kv_v = {index: self.gpu.alloc(body) for index in attention}
        self.kv_k_tail = {index: self.gpu.alloc(tail) for index in attention}
        self.kv_v_tail = {index: self.gpu.alloc(tail) for index in attention}
        self.indexer_buf = {index: self.gpu.alloc(INDEX_BF16_BYTES) for index in attention}
        self._mtp_on = False
        self._span_snap = False
        self._span_snap_count = 0
        self._mtp_stopped = False
        self._mtp_layer = len(self.kinds)
        self._mtp_index_n = 0
        self._mtp_hidden: Any = None
        self._bundle: dict[int, dict[str, Any]] = {}
        self._final: dict[str, Any] | None = None
        self._head: Int4Matrix | None = None
        self._ple: dict[str, Any] | None = None
        if bf16:
            self.store = ShardStore(root)
            self.files = _Bf16Ngram(self.store)
            self._load_bf16()
        else:
            self.files = QuantRecord(root)
            self._hot = expert_planes(self.files.hot_bits, self.hidden, self.moe)
            self._cold = expert_planes(self.files.cold_bits, self.hidden, self.moe)
            self._super_planes = expert_planes(self.files.super_bits, self.hidden, self.moe)
            self._load_record()
            self._load_mtp()
            self._load_supers()
            self._upload_tiers()
            state_bytes = 48 * 128 * 128 * 4
            self._gdn_shadow = {
                index: self.gpu.alloc(state_bytes) for index in self.gdn_state
            }
            self._conv_shadow = {
                index: self.gpu.alloc(self._conv_bytes) for index in self.conv_hist
            }
            self._ple_shadow = self.gpu.alloc(self._ple_bytes)
            tail_bytes = KV_TAIL_TOKENS * KV_BF16_BYTES
            self._k_tail_shadow = {
                index: self.gpu.alloc(tail_bytes) for index in self.kv_k_tail
            }
            self._v_tail_shadow = {
                index: self.gpu.alloc(tail_bytes) for index in self.kv_v_tail
            }
        tok_cfg = json.loads((root / "tokenizer_config.json").read_text())
        self.chat_template = tok_cfg["chat_template"]
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._add_stop_token(tok_cfg.get("eos_token"))
        generation = root / "generation_config.json"
        if generation.is_file():
            eos_ids = json.loads(generation.read_text()).get("eos_token_id")
            if isinstance(eos_ids, int):
                self.stop_ids.add(eos_ids)
            elif isinstance(eos_ids, list):
                self.stop_ids.update(int(item) for item in eos_ids)
        self.prefixes = PrefixStore(
            default_prefix_dir(),
            digest=checkpoint_digest(root),
            tag=self.kv_cache,
            budget=DEFAULT_BUDGET,
        )
        self._set_prompt_limit(resident_percent)

    def _set_prompt_limit(self, resident_percent: float | None) -> None:
        self.resident_percent = resident_percent
        self.prefill_chunk = PREFILL_CHUNK
        self._tier_bytes = 0
        self._hot_bytes = 0
        if isinstance(self.files, QuantRecord):
            self._hot_bytes = sum(os.fstat(fd).st_size for fd in self.files.hot_fds)
            cold = sum(os.fstat(fd).st_size for fd in self.files.cold_fds)
            self._tier_bytes = self._hot_bytes + cold
            if resident_percent is not None:
                self.resident_candidates = resident_candidates(self._hot_bytes, resident_percent)
                self.resident_budget = self.resident_candidates[0]
            else:
                self.resident_candidates = ()
                self.resident_budget = 0
        else:
            self.resident_candidates = ()
            self.resident_budget = 0
        if self.resident_candidates:
            _log(f"kv {self.kv_cache}, resident budget {self.resident_budget >> 20} MiB")
        else:
            _log(f"kv {self.kv_cache}, resident atlas follows the live sequence")

    def _add_stop_token(self, piece: Any) -> None:
        """A single-id end marker stops generation. The n-gram boundary stays `eos`."""
        if not isinstance(piece, str) or piece == "":
            return
        encoded = self.tokenizer.encode(piece).ids
        if len(encoded) == 1:
            self.stop_ids.add(int(encoded[0]))

    def halt(self) -> None:
        """The next layer boundary ends the forward."""
        self._stop = lambda: True

    def _discard_device_buffers(self) -> None:
        """Mark shared buffers empty so the next runtime can size a new atlas."""
        seen: set[int] = set()

        def walk(value: Any) -> None:
            if value is None or isinstance(value, (str, bytes, int, float, np.ndarray)):
                return
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if hasattr(value, "setPurgeableState_"):
                discard_buffer(value)
                return
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)
                return
            if isinstance(value, (list, tuple)):
                for child in value:
                    walk(child)
                return
            for name in ("buf", "weights", "scales", "atlas", "table", "miss"):
                walk(getattr(value, name, None))

        walk(getattr(self, "atlas", None))
        for name in (
            "weights",
            "f16_weights",
            "expert_cache",
            "gdn_state",
            "conv_hist",
            "kv_k",
            "kv_v",
            "kv_k_tail",
            "kv_v_tail",
            "indexer_buf",
            "_bundle",
            "_super_images",
            "_f32_uploaded",
            "_gdn_shadow",
            "_conv_shadow",
            "_k_tail_shadow",
            "_v_tail_shadow",
            "_stage",
            "_stage_table",
            "_stage_miss",
            "_resident",
            "_hot_keep",
            "_hold_xs",
            "_hold_out",
            "_hold_ids",
            "_hold_w",
            "_final",
            "_head",
            "_ple",
            "_ple_shadow",
        ):
            walk(getattr(self, name, None))

    def close(self) -> None:
        """Write the live prefix, then release the device, the atlas, and the record."""
        if self._closed:
            return
        self._persist_prefix()
        self._closed = True
        self._drop_prefill_progress()
        progress = getattr(self, "_decode_progress", None)
        self._decode_progress = None
        self._decode_reported = True
        console = getattr(self, "console", None)
        if progress is not None and console is not None and not progress.closed:
            console.end(progress, "")
        load = getattr(self, "_load_progress", None)
        self._load_progress = None
        if load is not None and console is not None and not load.closed:
            console.end(load, "")
        gpu = getattr(self, "gpu", None)
        if gpu is not None:
            try:
                gpu.flush()
                gpu.finish_io()
            except Exception:
                pass
        if getattr(self, "_streaming_prefill", False):
            try:
                self._end_prefill_stream()
            except Exception:
                pass
        files = getattr(self, "files", None)
        if isinstance(files, QuantRecord):
            files.release_io()
        self._discard_device_buffers()
        self.atlas = None
        self._stage = None
        self._stage_table = None
        self._stage_miss = None
        self._resident = None
        self._hot_keep = None
        self._hold_xs = None
        self._hold_out = None
        self._hold_ids = None
        self._hold_w = None
        self.weights = {}
        self.f16_weights = {}
        self.expert_cache = {}
        self.gdn_state = {}
        self.conv_hist = {}
        self.kv_k = {}
        self.kv_v = {}
        self.kv_k_tail = {}
        self.kv_v_tail = {}
        self.indexer_buf = {}
        self._bundle = {}
        self._final = None
        self._head = None
        self._ple = None
        self._super_images = {}
        self._f32_uploaded = {}
        self._choice = {}
        self.last_logits = None
        self._gdn_shadow = {}
        self._conv_shadow = {}
        self._ple_shadow = None
        self._k_tail_shadow = {}
        self._v_tail_shadow = {}
        gc.collect()
        if gpu is not None:
            try:
                gpu.close()
            except Exception:
                pass
            self.gpu = None
        if isinstance(files, QuantRecord):
            files.close()
        store = getattr(self, "store", None)
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
            self.store = None

    def _check_stop(self) -> None:
        stop = self._stop
        if stop is not None and stop():
            raise RequestClosed()

    def _cancel_forward(self) -> None:
        try:
            if self.gpu is not None:
                self.gpu.flush()
        finally:
            if self._streaming_prefill:
                self._end_prefill_stream()
            elif self.atlas is not None:
                self.atlas.release_unused()
            if getattr(self, "_kv_snap", None):
                self._restore_recurrent()
                self._kv_snap = None
            else:
                self.reset()
            self._forward_open = False
            self._persist_prefix()

    def reset(self) -> None:
        self.tokens = []
        self.last_logits = None
        self._prefix_dirty = True
        for buffer in self.gdn_state.values():
            _fill_buffer(buffer, 48 * 128 * 128 * 4, 0)
        for history in self.conv_hist.values():
            _fill_buffer(history, self._conv_bytes, 0)
        _fill_buffer(self.ple_hist, self._ple_bytes, 0)
        for index in self.kv_len:
            self.kv_len[index] = 0

    def align(self, prompt: list[int]) -> int:
        if self.tokens and prompt[: len(self.tokens)] == self.tokens:
            cached = len(self.tokens)
        else:
            # A cache hit decodes with the experts already mapped. A streamed
            # prefill evicts the atlas tail until its staging buffers fit.
            self._flush_prefix(release_atlas=False)
            self.reset()
            cached = self._load_prefix(prompt)
        self._cached_tokens = cached
        self._prompt_tokens = len(prompt)
        if self.console is None or self.console.keeps_lines:
            _log(f"prompt tokens {len(prompt)}")
            if cached:
                _log(f"cache hit {cached}")
            else:
                _log("cache miss")
            _log(f"incoming tokens {len(prompt) - cached}")
        elif cached and cached == len(prompt):
            # Pretty mode has no prefill row when every token was cached.
            _log(f"cache hit {cached}")
        return cached

    def _flush_prefix(self, *, release_atlas: bool = False) -> None:
        if not self.tokens:
            return
        self.gpu.flush()
        self._rewind_uncommitted()
        try:
            state = self._prefix_state()
        except RuntimeError as exc:
            _log(f"prefix store failed {exc}")
            return
        # The atlas stays mapped across the prefix write. Decode still needs
        # those expert images. Stopping the process drops the device.
        if release_atlas:
            self._release_atlas()
        try:
            path = self.prefixes.save(state)
        except OSError as exc:
            _log(f"prefix store failed {exc}")
            return
        if path is not None:
            self._prefix_dirty = False
            _log(f"prefix store {len(self.tokens)} tokens")

    def _rewind_uncommitted(self) -> None:
        """Drop a draft span that reached the cache and not the token list.

        The verify forward writes the span before those tokens are committed.
        A client that leaves during that window still has the committed
        sequence. The snapshot from before the span is that sequence.
        """
        lengths = getattr(self, "kv_len", None)
        tokens = getattr(self, "tokens", None)
        if not lengths or not tokens:
            return
        count = len(tokens)
        if all(length == count for length in lengths.values()):
            return
        layer = getattr(self, "_mtp_layer", None)
        snap = getattr(self, "_kv_snap", None)
        if snap:
            backbone = {index: item for index, item in snap.items() if index != layer}
            if backbone and all(item[5] == count for item in backbone.values()):
                self._restore_recurrent()
        mtp = getattr(self, "_mtp_kv", None)
        if (
            mtp is not None
            and layer in self.kv_len
            and mtp[5] == count
            and self.kv_len[layer] != count
        ):
            self._mtp_restore_kv()
        if any(length != count for length in self.kv_len.values()):
            return
        logits = getattr(self, "last_logits", None)
        if logits is not None and getattr(logits, "n", None) != getattr(self, "vocab", None):
            self.last_logits = None

    def _persist_prefix(self, *, release_atlas: bool = False) -> None:
        """Write the last consistent sequence. The atlas stays mapped."""
        if getattr(self, "_forward_open", False):
            if not getattr(self, "_kv_snap", None):
                return
            try:
                if getattr(self, "gpu", None) is not None:
                    self.gpu.flush()
                self._restore_recurrent()
            except Exception as exc:
                _log(f"prefix store failed {exc}")
                return
            self._kv_snap = None
            self._forward_open = False
        tokens = getattr(self, "tokens", None)
        if (
            not tokens
            or getattr(self, "gpu", None) is None
            or getattr(self, "prefixes", None) is None
        ):
            return
        if getattr(self, "_prefix_dirty", True) is False:
            return
        try:
            self._flush_prefix(release_atlas=release_atlas)
        except Exception as exc:
            _log(f"prefix store failed {exc}")

    def _prefix_state(self) -> PrefixState:
        count = len(self.tokens)
        body, _stride = self._kv_span(count)
        tail = KV_TAIL_TOKENS * KV_BF16_BYTES
        attention: list[AttentionImage] = []
        for layer, length in self.kv_len.items():
            if length != count:
                message = f"attention cache length {length} does not match {count} tokens"
                raise RuntimeError(message)
            attention.append(
                AttentionImage(
                    layer,
                    self._span_view(self.kv_k[layer], 0, body),
                    self._span_view(self.kv_v[layer], 0, body),
                    self._span_view(self.kv_k_tail[layer], 0, tail),
                    self._span_view(self.kv_v_tail[layer], 0, tail),
                    self._span_view(self.indexer_buf[layer], 0, count * INDEX_BF16_BYTES),
                )
            )
        state_bytes = 48 * 128 * 128 * 4
        gdn = [
            FixedImage(layer, self._span_view(buf, 0, state_bytes))
            for layer, buf in self.gdn_state.items()
        ]
        conv = [
            FixedImage(layer, self._span_view(buf, 0, self._conv_bytes))
            for layer, buf in self.conv_hist.items()
        ]
        if self.last_logits is None:
            logits: bytes | memoryview = b""
        else:
            if self.last_logits.n != self.vocab:
                message = "cached logits do not match the vocabulary"
                raise RuntimeError(message)
            logits = self._span_view(
                self.last_logits.buf, self.last_logits.offset * 4, self.vocab * 4
            )
        return PrefixState(
            list(self.tokens),
            attention,
            gdn,
            conv,
            self._span_view(self.ple_hist, 0, self._ple_bytes),
            logits,
        )

    def _span_view(self, buf: Any, byte_off: int, nbytes: int) -> memoryview:
        if nbytes <= 0:
            return memoryview(b"")
        if int(buf.length()) < byte_off + nbytes:
            message = f"cache buffer is {int(buf.length())} bytes, need {byte_off + nbytes}"
            raise RuntimeError(message)
        view = _buffer_view(buf, byte_off + nbytes)
        return view[byte_off : byte_off + nbytes]

    def _load_prefix(self, prompt: list[int]) -> int:
        for header in self.prefixes.matches(prompt):
            if not self._prefix_fits(header):
                continue
            try:
                self._read_prefix(header)
            except (OSError, RuntimeError) as exc:
                _log(f"prefix load failed {exc}")
                self.reset()
                continue
            self._prefix_dirty = False
            _log(f"prefix load {len(header.tokens)} tokens")
            return len(header.tokens)
        return 0

    def _prefix_fits(self, header: PrefixHeader) -> bool:
        count = len(header.tokens)
        attention = {image.layer: image for image in header.attention}
        if set(attention) != set(self.kv_len):
            return False
        body, _stride = self._kv_span(count)
        tail = KV_TAIL_TOKENS * KV_BF16_BYTES
        index = count * INDEX_BF16_BYTES
        for image in attention.values():
            if (
                image.key.length != body
                or image.value.length != body
                or image.key_tail.length != tail
                or image.value_tail.length != tail
                or image.indexer.length != index
            ):
                return False
        gdn = {image.layer: image for image in header.gdn}
        if set(gdn) != set(self.gdn_state):
            return False
        state_bytes = 48 * 128 * 128 * 4
        if any(image.blob.length != state_bytes for image in gdn.values()):
            return False
        conv = {image.layer: image for image in header.conv}
        if set(conv) != set(self.conv_hist):
            return False
        if any(image.blob.length != self._conv_bytes for image in conv.values()):
            return False
        if header.ple.length != self._ple_bytes:
            return False
        return header.logits.length in (0, self.vocab * 4)

    def _read_prefix(self, header: PrefixHeader) -> None:
        loads: list[tuple[Any, int, int, int]] = []
        for image in header.attention:
            self.kv_k[image.layer] = self._stage_prefix(
                self.kv_k, image.layer, image.key, loads
            )
            self.kv_v[image.layer] = self._stage_prefix(
                self.kv_v, image.layer, image.value, loads
            )
            self.kv_k_tail[image.layer] = self._stage_prefix(
                self.kv_k_tail, image.layer, image.key_tail, loads
            )
            self.kv_v_tail[image.layer] = self._stage_prefix(
                self.kv_v_tail, image.layer, image.value_tail, loads
            )
            self.indexer_buf[image.layer] = self._stage_prefix(
                self.indexer_buf, image.layer, image.indexer, loads
            )
        for image in header.gdn:
            self.gdn_state[image.layer] = self._stage_prefix(
                self.gdn_state, image.layer, image.blob, loads
            )
        for image in header.conv:
            self.conv_hist[image.layer] = self._stage_prefix(
                self.conv_hist, image.layer, image.blob, loads
            )
        self.ple_hist = self._stage_one(self.ple_hist, header.ple, loads)
        logits = self._stage_one(None, header.logits, loads)
        handle = self._prefix_io(header.path)
        try:
            if loads:
                self.gpu.io_read(handle, loads)
        finally:
            del handle
        self.tokens = list(header.tokens)
        for layer in self.kv_len:
            self.kv_len[layer] = len(self.tokens)
        if header.logits.length:
            self.last_logits = Dev(logits, self.vocab)
        else:
            self.last_logits = None

    def _stage_prefix(
        self,
        table: dict[int, Any],
        layer: int,
        ref: Any,
        loads: list[tuple[Any, int, int, int]],
    ) -> Any:
        stored = stored_bytes(ref.length)
        if stored == 0:
            return table[layer]
        buf = self._replace_cache(table, layer, stored)
        loads.append((buf, 0, stored, ref.offset))
        return buf

    def _stage_one(
        self, buf: Any, ref: Any, loads: list[tuple[Any, int, int, int]]
    ) -> Any:
        stored = stored_bytes(ref.length)
        if stored == 0:
            return buf
        if buf is None or int(buf.length()) < stored:
            buf = self.gpu.alloc(stored)
            if buf is None:
                message = f"prefix buffer of {stored} bytes was not allocated"
                raise RuntimeError(message)
        loads.append((buf, 0, stored, ref.offset))
        return buf

    def _replace_cache(self, table: dict[int, Any], layer: int, nbytes: int) -> Any:
        current = table[layer]
        if current is not None and int(current.length()) >= nbytes:
            return current
        allocated = self.gpu.alloc(nbytes)
        if allocated is None:
            message = f"prefix buffer of {nbytes} bytes was not allocated"
            raise RuntimeError(message)
        table[layer] = allocated
        return allocated

    def _prefix_io(self, path: Path) -> Any:
        import Foundation

        self.gpu.ensure_io()
        url = Foundation.NSURL.fileURLWithPath_(str(path))
        opened = self.gpu.device.newUncachedIOHandleWithURL_error_(url, None)
        if opened is None:
            message = f"prefix IO handle failed for {path.name}"
            raise RuntimeError(message)
        return opened

    def _prompt_logits(self, prompt: list[int]) -> Dev:
        """Logits for ``prompt``. A boundary with no logits is prefilled again."""
        start = self.align(prompt)
        if start == len(prompt) and self.last_logits is None:
            _log("prefix has no logits")
            self.reset()
            start = 0
        if start == len(prompt):
            logits = self.last_logits
            if not isinstance(logits, Dev):
                message = "the cached prefix has no logits"
                raise RuntimeError(message)
            return logits
        return self.forward(prompt[start:])

    def generate(
        self,
        prompt: list[int],
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[int]:
        # Drain Objective-C temporaries at the end of the request. The process
        # has no AppKit run loop to do that.
        with objc.autorelease_pool():
            logits = self._prompt_logits(prompt)
            tokens = list(self._new_tokens(logits, max_tokens, temperature, top_p, top_k))
            self._log_decode()
            return tokens

    def _new_tokens(
        self,
        logits: Any,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        self._close_decode_row()
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self._decode_steps = []
        self._output_ids = []
        self._hit_limit = False
        self._begin_decode(max_tokens)
        self._begin_drafts()
        # The first sampled token fills the atlas. Later tokens reuse it.
        self._atlas_trusted = False
        produced_ids: list[int] = []
        produced = 0
        try:
            while True:
                self._check_stop()
                if max_tokens is not None and produced >= max_tokens:
                    self._hit_limit = True
                    return
                token = self._pick(logits, temperature, top_p, top_k, produced_ids)
                if token in self.stop_ids:
                    return
                room = getattr(self, "_draft_limit", 3) + 1
                if max_tokens is not None:
                    room = max_tokens - produced
                if room <= 0:
                    self._hit_limit = True
                    return
                # Prefill logits keep that scratch alive. Drop them before decode arenas.
                self.last_logits = None
                logits = None
                if self._mtp_drafts:
                    pending = self._mtp_cycle(
                        token, produced_ids, room, temperature, top_p, top_k
                    )
                    logits = self.last_logits
                    stopped = self._mtp_stopped
                else:
                    pending = [token]
                    logits = self.forward([token], decode=True)
                    # The backbone cache advanced with this token. The MTP
                    # layer has to follow, or the next prefill finds it behind.
                    if self._mtp_on:
                        self._mtp_step(token, reuse=False, head=False)
                    stopped = False
                for item in pending:
                    self._check_stop()
                    self._output_ids.append(item)
                    yield item
                    produced += 1
                    produced_ids.append(item)
                    if self.required_function and "</tool_call>" in self.decode(produced_ids):
                        return
                if stopped:
                    return
        finally:
            self._log_decode()

    def _begin_drafts(self) -> None:
        self._mtp_rates: list[float] = []
        self._mtp_drafts = mtp_starts(len(self.tokens), loaded=bool(self._mtp_on))

    def _note_draft_rate(self, tokens: int, seconds: float) -> None:
        if not getattr(self, "_mtp_drafts", False) or tokens <= 0 or seconds <= 0:
            return
        rates = getattr(self, "_mtp_rates", None)
        if rates is None:
            return
        rates.append(tokens / seconds)
        if mtp_holds(rates):
            return
        self._mtp_drafts = False
        progress = getattr(self, "_decode_progress", None)
        if progress is not None and "mtp off" not in progress.detail:
            progress.detail = f"{progress.detail} · mtp off" if progress.detail else "mtp off"
        _log("mtp off")

    def _drop_prefill_progress(self) -> None:
        progress = getattr(self, "_prefill_progress", None)
        self._prefill_progress = None
        console = getattr(self, "console", None)
        if progress is not None and console is not None and not progress.closed:
            console.end(progress, "")

    def _close_decode_row(self) -> None:
        """End a decode row left open when the request stopped early."""
        if not self._decode_reported:
            self._log_decode()

    def _begin_decode(self, max_tokens: int | None) -> None:
        self._close_decode_row()
        self._decode_reported = False
        console = self.console
        if console is None or not console.live_decode:
            self._decode_progress = None
            return
        detail = f"req #{self._request_id}" if self._request_id else ""
        self._decode_progress = console.begin(
            "decode",
            total=0 if max_tokens is None else max_tokens,
            unit="tok",
            detail=detail,
            request=self._request_id,
        )

    def _note_decode(self, resident: int) -> None:
        progress = self._decode_progress
        if progress is None:
            return
        progress.done = len(self._output_ids)
        progress.gauge = ("resident", resident)

    def _note_prefill(self, layer: int, seen: int, count: int) -> None:
        progress = self._prefill_progress
        if progress is None or count <= 0 or self.layers <= 0:
            return
        frac = (layer + seen / count) / self.layers
        done = min(progress.total, int(frac * progress.total))
        if done > progress.done:
            progress.done = done

    def _log_decode(self) -> None:
        if self._decode_reported:
            return
        self._decode_reported = True
        summary = _tok_s("decode", self._decode_tokens, self._decode_seconds)
        progress = self._decode_progress
        self._decode_progress = None
        if progress is not None and self.console is not None:
            if self.console.keeps_lines:
                _log(f"output tokens {len(self._output_ids)}")
            self.console.end(progress, summary)
            return
        _log(f"output tokens {len(self._output_ids)}")
        _log(summary)

    def _pick(
        self,
        logits: Any,
        temperature: float,
        top_p: float,
        top_k: int,
        produced: list[int],
    ) -> int:
        prepared = self._prepare(logits, temperature, produced)
        if isinstance(prepared, int):
            return prepared
        if isinstance(prepared, Dev):
            return self.gpu.sample(prepared, temperature, top_p, top_k, self.rng)
        return _sample(prepared, temperature, top_p, top_k, self.rng)

    def _prepare(self, logits: Any, temperature: float, produced: list[int]) -> Any:
        if self.tool_names and isinstance(logits, Dev):
            constrained = self._constrained_name(logits, temperature, produced)
            if constrained is not None:
                return constrained
        if (
            self._avoid_token is not None
            and self._avoid_token not in produced
            and len(produced) >= 512
            and looped_span(produced)
        ):
            return self._avoid_token
        fence_open = False
        if (
            self._avoid_token is not None
            and self._avoid_token in produced
            and len(produced) >= 16
        ):
            reply = self.decode(produced).split("</think>", 1)[-1]
            fence_open = reply.count("```") % 2 == 1
        banned = repeated_cycle(produced)
        narrow = stuck_ids(produced)
        closed = self._avoid_token is not None and self._avoid_token in produced
        repeat = answer_repeat_id(produced) if self._answer_mode or closed else None
        penalized: list[int] = []
        penalty = 0.0
        if self._frequency_penalty and produced:
            penalized = penalty_ids(produced, self._penalty_window, once=False)
            penalty = self._frequency_penalty
        elif self._presence_penalty and produced:
            penalized = penalty_ids(produced, self._penalty_window, once=True)
            penalty = self._presence_penalty
        if isinstance(logits, Dev):
            if penalized:
                self.gpu.presence(logits, penalized, penalty)
            if (
                self._avoid_token is not None
                and len(produced) < self._avoid_until
                and self._avoid_token < logits.n
            ):
                self.gpu.suppress_logit(logits, self._avoid_token)
            for token in self._block_tokens:
                if token < logits.n:
                    self.gpu.suppress_logit(logits, token)
            if banned is not None and banned < logits.n:
                self.gpu.suppress_logit(logits, banned)
            for token in narrow:
                if token < logits.n:
                    self.gpu.suppress_logit(logits, token)
            if repeat is not None and repeat < logits.n:
                self.gpu.suppress_logit(logits, repeat)
            if self._avoid_token is not None:
                answered = 0
                closed = self._avoid_token in produced
                if closed:
                    answered = len(produced) - 1 - produced.index(self._avoid_token)
                hold_stop = (
                    fence_open
                    or (not closed and len(produced) < 64)
                    or (closed and answered < 16)
                )
                if hold_stop:
                    for stop in self.stop_ids:
                        if stop < logits.n:
                            self.gpu.suppress_logit(logits, stop)
        elif penalized or banned is not None or narrow or self._block_tokens or repeat is not None:
            logits = np.array(logits, copy=True)
            for token in penalized:
                if token < int(logits.size):
                    logits[int(token)] -= penalty
            if banned is not None and banned < int(logits.size):
                logits[int(banned)] = -1.0e30
            for token in narrow:
                if token < int(logits.size):
                    logits[int(token)] = -1.0e30
            for token in self._block_tokens:
                if token < int(logits.size):
                    logits[int(token)] = -1.0e30
            if repeat is not None and repeat < int(logits.size):
                logits[int(repeat)] = -1.0e30
        return logits

    def _constrained_name(self, logits: Dev, temperature: float, produced: list[int]) -> int | None:
        """The next id of a required opening, or a token that continues a declared name.

        A required opening uses the tokenizer's encoding of that text.
        """
        text = self.decode(produced) if produced else ""
        options = forced_call_options(
            text, self.required_function, close=self.required_closes
        )
        forced = options is not None
        if options is None and produced:
            options = function_name_options(text, self.tool_names)
        if not options:
            return None
        if forced and len(options) == 1:
            nxt = canonical_continuation(
                produced,
                options[0],
                lambda piece: self.tokenizer.encode(piece).ids,
                self.decode,
            )
            if nxt is not None:
                return nxt
        allowed = continuing_ids(options, self._pieces())
        if not allowed:
            return None
        raw = _buffer_slice(logits.buf, logits.offset * 4, logits.n * 4)
        values = np.frombuffer(raw, dtype=np.float32)
        ids = np.asarray(allowed, dtype=np.int64)
        chosen = values[ids]
        if temperature <= 0.0 or not np.isfinite(chosen).any():
            return int(ids[int(np.argmax(chosen))])
        scaled = chosen / temperature
        scaled = scaled - np.max(scaled)
        weights = np.exp(scaled)
        total = float(np.sum(weights))
        if total == 0.0 or not np.isfinite(total):
            return int(ids[int(np.argmax(chosen))])
        return int(self.rng.choice(ids, p=weights / total))

    def _pieces(self) -> list[str]:
        if self._token_pieces is None:
            size = int(self.tokenizer.get_vocab_size())
            self._token_pieces = [self.tokenizer.decode([index]) for index in range(size)]
        return self._token_pieces

    def forward(
        self,
        tokens: list[int],
        *,
        decode: bool = False,
        commit: bool = True,
        account: bool = True,
    ) -> Dev:
        if not decode:
            self._close_decode_row()
        self._check_stop()
        started = time.perf_counter()
        gpu_at = self.gpu.gpu_seconds
        self._decoding = decode
        self._prefill_progress = None
        if self.console is not None and not decode and tokens:
            detail = f"req #{self._request_id}" if self._request_id else ""
            self._prefill_progress = self.console.begin(
                "prefill",
                total=len(tokens),
                unit="tok",
                detail=detail,
                request=self._request_id,
            )
        total = len(self.tokens) + len(tokens)
        chunk = len(tokens) if decode else min(len(tokens), self.prefill_chunk)
        scratch = scratch_bytes(max(chunk, 1))
        if not decode and total > DENSE_LIMIT:
            blocks = max(total // INDEX_BLOCK, 1)
            scratch += chunk * blocks * 4 + chunk * INDEX_STRIDE * 4
        self._rewind_uncommitted()
        self.gpu.begin_scratch(scratch, copies=2 if decode else 1)
        if decode:
            self._make_room_for_cache(len(tokens))
        self._prepare_experts(decode=decode, count=len(tokens))
        if self.atlas is not None:
            _fill_buffer(self.atlas.miss, self.layers * 4, 0)
        if decode:
            self._span_rows = len(tokens) if 1 < len(tokens) <= 4 else 1
        if decode and self.atlas is not None:
            self._prefetch_decode()
        self._capture_recurrent()
        self._forward_open = bool(getattr(self, "_kv_snap", None))
        resident_before = self.atlas.resident_count if decode and self.atlas is not None else 0
        try:
            logits = self._forward_body(tokens)
            self.gpu.flush()
            if self._streaming_prefill:
                missed = self._stage_missed()
                self._end_prefill_stream()
                if missed:
                    message = "prefill referenced an expert that was not loaded"
                    raise RuntimeError(message)
            elif self.atlas is not None and self._expert_miss():
                if not decode:
                    message = "prefill referenced an expert that was not loaded"
                    raise RuntimeError(message)
                self._atlas_trusted = False
                logits = self._rerun_decode(tokens)
        except RequestClosed:
            self._cancel_forward()
            self._drop_prefill_progress()
            raise
        if decode and self.atlas is not None and not self._expert_miss():
            self._note_routes()
            # A token that loaded experts is still filling the atlas. Trusting
            # the next one would compute it twice, once before the miss replay.
            self._atlas_trusted = self.atlas.resident_count == resident_before
        if commit:
            self.tokens.extend(tokens)
            self._prefix_dirty = True
        self._forward_open = False
        self.last_logits = logits
        elapsed = time.perf_counter() - started
        gpu_elapsed = self.gpu.gpu_seconds - gpu_at
        self._last_elapsed = elapsed
        self._last_gpu = gpu_elapsed
        if decode and account:
            self._decode_tokens += len(tokens)
            self._decode_seconds += elapsed
            resident = self.atlas.resident_count if self.atlas is not None else 0
            self._decode_steps.append((elapsed, gpu_elapsed, resident))
            self._note_decode(resident)
        elif not decode:
            self._prefill_tokens = len(tokens)
            self._prefill_seconds = elapsed
            summary = _tok_s("prefill", len(tokens), elapsed)
            progress = self._prefill_progress
            self._prefill_progress = None
            if progress is not None and self.console is not None:
                progress.done = len(tokens)
                if self.console.fold_turns:
                    self.console.end(progress, "")
                else:
                    self.console.end(progress, summary)
            else:
                _log(summary)
            if commit:
                self._persist_prefix(release_atlas=False)
        if self.atlas is not None:
            self.atlas.release_unused()
        return logits

    def _load_mtp(self) -> None:
        """Keep the MTP expert file resident and give that layer its own cache."""
        if not isinstance(self.files, QuantRecord) or self.files.mtp_io is None:
            return
        if "mtp.fc_hidden.weight" not in self.weights:
            return
        if "mtp.layers.0.mlp.gate.weight" not in self.f16_weights:
            return
        path = self.files.root / "mtp-experts.bin"
        size = path.stat().st_size
        slot = int(self.files.manifest.get("mtp_slot") or 0)
        if slot <= 0 or size % slot:
            return
        bits = int(self.files.manifest.get("mtp_bits") or 4)
        buffer = self.gpu.alloc(size)
        if buffer is None:
            message = "MTP expert file was not allocated"
            raise RuntimeError(message)
        self.gpu.io_read(self.files.mtp_io, [(buffer, 0, size, 0)])
        self._mtp_experts = buffer
        self._mtp_slot = slot
        self._mtp_planes = expert_planes(bits, self.hidden, self.moe)
        self._mtp_members = set(range(self.experts))
        table = (np.arange(self.experts, dtype=np.int64) * slot).tobytes()
        self._mtp_table = self.gpu.upload(table)
        self._mtp_miss = self.gpu.zeros(4)
        self._mtp_indices = self.gpu.zeros(INDEX_STRIDE * 4)
        self._mtp_count = self.gpu.zeros(4)
        span = self.branches * self.hidden
        self._mtp_hidden = self.gpu.zeros(span * 4)
        self._mtp_hidden_snap = self.gpu.zeros(span * 4)
        self._mtp_backbone = self.gpu.zeros(span * 4)
        self._mtp_in = self.gpu.alloc(PREFILL_CHUNK * span * 4)
        self._mtp_proj = self.gpu.alloc(PREFILL_CHUNK * span * 4)
        ones = np.ones(PREFILL_CHUNK * span, dtype=np.float32)
        self._mtp_ones = self.gpu.upload(ones.tobytes())
        held = (
            self._mtp_table,
            self._mtp_miss,
            self._mtp_indices,
            self._mtp_count,
            self._mtp_hidden,
            self._mtp_hidden_snap,
            self._mtp_backbone,
            self._mtp_in,
            self._mtp_proj,
            self._mtp_ones,
        )
        if any(buf is None for buf in held):
            message = "MTP scratch was not allocated"
            raise RuntimeError(message)
        body = KV_BF16_BYTES if self.kv_cache == "bf16" else KV_E4M3_BYTES
        tail = KV_TAIL_TOKENS * KV_BF16_BYTES
        layer = self._mtp_layer
        self.kv_len[layer] = 0
        self.kv_k[layer] = self.gpu.alloc(body)
        self.kv_v[layer] = self.gpu.alloc(body)
        self.kv_k_tail[layer] = self.gpu.alloc(tail)
        self.kv_v_tail[layer] = self.gpu.alloc(tail)
        self.indexer_buf[layer] = self.gpu.alloc(INDEX_BF16_BYTES)
        self._mtp_on = True
        self._mtp_pack: dict[str, Any] | None = None
        self._mtp_logit_buf: Any = None
        self._mtp_logit_n = 0
        self._mtp_tail_snap: tuple[Any, Any] | None = None

    def _mtp_bundle(self) -> dict[str, Any]:
        if self._mtp_pack is not None:
            return self._mtp_pack
        prefix = "mtp.layers.0"
        mixer = "mtp.hyper_connection_mixer"
        self._mtp_pack = {
            "attn_norm": self._weight(f"{prefix}.attn_hyper_connection.hc_norm.weight"),
            "attn_down": self._int4(f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight"),
            "attn_up": self._int4(f"{prefix}.attn_hyper_connection.input_mix_weight_up.weight"),
            "attn_inject": self._int4(f"{prefix}.attn_hyper_connection.block_inject_weight.weight"),
            "mlp_norm": self._weight(f"{prefix}.mlp_hyper_connection.hc_norm.weight"),
            "mlp_down": self._int4(f"{prefix}.mlp_hyper_connection.input_mix_weight_down.weight"),
            "mlp_up": self._int4(f"{prefix}.mlp_hyper_connection.input_mix_weight_up.weight"),
            "mlp_inject": self._int4(f"{prefix}.mlp_hyper_connection.block_inject_weight.weight"),
            "shared_gate": self._int4(f"{prefix}.mlp.shared_expert.gate_proj.weight"),
            "shared_up": self._int4(f"{prefix}.mlp.shared_expert.up_proj.weight"),
            "shared_down": self._int4(f"{prefix}.mlp.shared_expert.down_proj.weight"),
            "shared_coeff": self._int4(f"{prefix}.mlp.shared_expert_gate.weight"),
            "q_proj": self._int4(f"{prefix}.self_attn.q_proj.weight"),
            "k_proj": self._int4(f"{prefix}.self_attn.k_proj.weight"),
            "v_proj": self._int4(f"{prefix}.self_attn.v_proj.weight"),
            "o_proj": self._int4(f"{prefix}.self_attn.o_proj.weight"),
            "q_norm": self._weight(f"{prefix}.self_attn.q_norm.weight"),
            "k_norm": self._weight(f"{prefix}.self_attn.k_norm.weight"),
            "index_qk": self._int4(f"{prefix}.self_attn.indexer.index_qk_proj.weight"),
            "pre_h": self._weight("mtp.pre_fc_norm_hidden.weight"),
            "pre_e": self._weight("mtp.pre_fc_norm_embedding.weight"),
            "fc_h": self._int4("mtp.fc_hidden.weight"),
            "fc_e": self._int4("mtp.fc_embedding.weight"),
            "final_norm": self._weight(f"{mixer}.hc_norm.weight"),
            "final_down": self._int4(f"{mixer}.input_mix_weight_down.weight"),
            "final_up": self._int4(f"{mixer}.input_mix_weight_up.weight"),
        }
        return self._mtp_pack

    def _mtp_copy(self, src: Any, dst: Any, n: int) -> None:
        self.gpu.copy_bytes(src, 0, dst, 0, n * 4)

    def _mtp_keep_residual(self, residual: Dev, count: int) -> None:
        span = self.branches * self.hidden
        last = residual.view((count - 1) * span, span)
        self.gpu.copy_span(last, 0, self._mtp_hidden, 0, span)

    def _mtp_snap_kv(self) -> None:
        layer = self._mtp_layer
        self._mtp_kv = (
            self.kv_k[layer],
            self.kv_v[layer],
            self.kv_k_tail[layer],
            self.kv_v_tail[layer],
            self.indexer_buf[layer],
            self.kv_len[layer],
        )
        tail = KV_TAIL_TOKENS * KV_BF16_BYTES
        if self._mtp_tail_snap is None:
            key_snap = self.gpu.alloc(tail)
            value_snap = self.gpu.alloc(tail)
            if key_snap is None or value_snap is None:
                message = "MTP tail snapshot was not allocated"
                raise RuntimeError(message)
            self._mtp_tail_snap = (key_snap, value_snap)
        key_snap, value_snap = self._mtp_tail_snap
        self.gpu.copy_bytes(self.kv_k_tail[layer], 0, key_snap, 0, tail)
        self.gpu.copy_bytes(self.kv_v_tail[layer], 0, value_snap, 0, tail)
        self.gpu.barrier()

    def _mtp_restore_kv(self) -> None:
        layer = self._mtp_layer
        key, value, key_tail, value_tail, indexer, length = self._mtp_kv
        self.kv_k[layer] = key
        self.kv_v[layer] = value
        self.kv_k_tail[layer] = key_tail
        self.kv_v_tail[layer] = value_tail
        self.indexer_buf[layer] = indexer
        self.kv_len[layer] = length
        if self._mtp_tail_snap is None:
            return
        tail = KV_TAIL_TOKENS * KV_BF16_BYTES
        key_snap, value_snap = self._mtp_tail_snap
        self.gpu.copy_bytes(key_snap, 0, self.kv_k_tail[layer], 0, tail)
        self.gpu.copy_bytes(value_snap, 0, self.kv_v_tail[layer], 0, tail)
        self.gpu.barrier()

    def _mtp_attend(
        self,
        query: Dev,
        gate: Dev,
        key: Any,
        value: Any,
        count: int,
        mode: int,
        complete: int,
        layer: int,
    ) -> Dev:
        return self.gpu.qsa_attend_selected(
            query,
            gate,
            key,
            value,
            Dev(self._mtp_indices, INDEX_STRIDE),
            Dev(self._mtp_count, 1),
            count,
            INDEX_STRIDE,
            mode=mode,
            complete=complete,
            key_tail=self.kv_k_tail[layer],
            value_tail=self.kv_v_tail[layer],
        )

    def _mtp_project(self, hidden: Dev, tokens: list[int]) -> Dev:
        """Norm the residual and the token embedding, then add the two projections."""
        count = len(tokens)
        bundle = self._mtp_bundle()
        span = self.branches * self.hidden
        hats = self._rms_branches(hidden, bundle["pre_h"], count)
        reduced = self.gpu.branch_mean(
            Dev(self._mtp_ones, count * span), hats, count, self.branches, self.hidden
        )
        from_hidden = self.gpu.linear(bundle["fc_h"], reduced, count, self.hidden)
        packed = self.gpu.tmp(count * self.hidden)
        meta = self._embed_meta
        stored = int(meta["stored_inner"])
        self.gpu.embed_gather(
            self._resident,
            int(meta["offset"]),
            self.gpu.dev_u32(tokens),
            packed,
            count,
            self.hidden,
            stored // 2,
            stored // BLOCK,
            int(meta["data"]),
            1,
        )
        weight, offset = bundle["pre_e"]
        normed = self.gpu.rms(packed, weight, offset, self.hidden, count, self.eps, True, 0)
        from_embed = self.gpu.linear(bundle["fc_e"], normed, count, self.hidden)
        mixed = self.gpu.add(from_hidden, from_embed)
        dst = Dev(self._mtp_proj, count * span)
        for index in range(count):
            row = mixed.view(index * self.hidden, self.hidden)
            self.gpu.broadcast(row, dst, index * span, self.branches, self.hidden)
        return dst

    def _mtp_moe(self, bundle: dict[str, Any], xs: Dev, count: int) -> Dev:
        router = self.f16_weights["mtp.layers.0.mlp.gate.weight"]
        choice, weight = self.gpu.topk_batch(
            self.gpu.linear(router, xs, count, self.hidden), count, self.experts, self.top_k
        )
        host_ids = np.asarray(self.gpu.read_u32(choice), dtype=np.uint32)
        shared = self._shared_expert(bundle, xs, count)
        return self._moe_tier(
            0,
            xs,
            weight,
            shared,
            host_ids,
            count,
            self._mtp_members,
            self._mtp_planes,
            self._mtp_experts,
            self._mtp_table,
            0,
            self._mtp_miss,
        )

    def _mtp_layer_pass(self, residual: Dev, count: int, base: int, *, reuse: bool) -> None:
        bundle = self._mtp_bundle()
        # Commit the projection into the other scratch arena. A flush here would
        # wait out the draft before the layer is even encoded.
        self.gpu.submit()
        if len(self.gpu._arenas) < 2:
            self.gpu.rewind_scratch()
        with objc.autorelease_pool():
            hats = self._rms_branches(residual, bundle["attn_norm"], count)
            xs = self._gr_dev(bundle["attn_down"], bundle["attn_up"], hats, count)
            mixed = self._qsa_dev(
                self._mtp_layer, bundle, xs, count, base=base, reuse=reuse
            )
            self._inject_dev(residual, hats, mixed, bundle["attn_inject"], count)
            hats = self._rms_branches(residual, bundle["mlp_norm"], count)
            xs = self._gr_dev(bundle["mlp_down"], bundle["mlp_up"], hats, count)
            mixed = self._mtp_moe(bundle, xs, count)
            self._inject_dev(residual, hats, mixed, bundle["mlp_inject"], count)

    def _mtp_head(self, residual: Dev) -> Dev:
        bundle = self._mtp_bundle()
        hat = self._rms_branches(residual, bundle["final_norm"], 1)
        hidden = self._gr_dev(bundle["final_down"], bundle["final_up"], hat, 1)
        return self.gpu.linear(self._lm_head(), hidden, 1, self.hidden)

    def _mtp_step(self, token: int, *, reuse: bool, head: bool) -> Dev | None:
        span = self.branches * self.hidden
        projected = self._mtp_project(Dev(self._mtp_hidden, span), [token])
        self._mtp_layer_pass(projected, 1, self.kv_len[self._mtp_layer], reuse=reuse)
        self._mtp_copy(self._mtp_proj, self._mtp_hidden, span)
        if not head:
            return None
        return self._mtp_head(Dev(self._mtp_proj, span))

    def _mtp_shift(self, residual: Dev, offset: int, count: int, base: int) -> None:
        span = self.branches * self.hidden
        dst = self._mtp_in
        if offset == 0 and base == 0:
            _fill_buffer(dst, span * 4, 0)
            if count > 1:
                self.gpu.copy_bytes(
                    residual.buf,
                    residual.offset * 4,
                    dst,
                    span * 4,
                    (count - 1) * span * 4,
                )
            return
        if offset == 0:
            self._mtp_copy(self._mtp_hidden, dst, span)
            if count > 1:
                self.gpu.copy_bytes(
                    residual.buf,
                    residual.offset * 4,
                    dst,
                    span * 4,
                    (count - 1) * span * 4,
                )
            return
        start = (offset - 1) * span
        self.gpu.copy_bytes(
            residual.buf,
            (residual.offset + start) * 4,
            dst,
            0,
            count * span * 4,
        )

    def _mtp_prefill(self, residual: Dev, tokens: list[int]) -> None:
        """Write one MTP cache entry per prompt token before the first draft."""
        count = len(tokens)
        origin = len(self.tokens)
        span = self.branches * self.hidden
        self.gpu.rewind_scratch()
        for offset in range(0, count, self.prefill_chunk):
            self._check_stop()
            n = min(self.prefill_chunk, count - offset)
            self._mtp_shift(residual, offset, n, origin)
            ids = tokens[offset : offset + n]
            projected = self._mtp_project(Dev(self._mtp_in, n * span), ids)
            self._mtp_layer_pass(projected, n, origin + offset, reuse=False)

    def _span_experts(self) -> bool:
        if self.atlas is None or not isinstance(self.files, QuantRecord):
            return False
        return span_expert_bits(self.files.hot_bits, self.files.cold_bits, self.files.super_bits)

    def _name_constrained(self, produced: list[int]) -> bool:
        if not self.tool_names:
            return False
        text = self.decode(produced) if produced else ""
        options = forced_call_options(
            text, self.required_function, close=self.required_closes
        )
        if options is None and produced:
            options = function_name_options(text, self.tool_names)
        return bool(options)

    def _draft_blocked(self, produced: list[int]) -> bool:
        if self._name_constrained(produced):
            return True
        return bool(
            self._avoid_token is not None
            and self._avoid_token not in produced
            and len(produced) >= 512
            and looped_span(produced)
        )

    def _mtp_alloc(self, name: str, words: int) -> Any:
        buf = self.gpu.alloc(words * 4)
        if buf is None:
            message = f"MTP {name} buffer was not allocated"
            raise RuntimeError(message)
        return buf

    def _mtp_hold_draft(self, index: int, logits: Dev) -> Dev:
        words = 3 * self.vocab
        if getattr(self, "_mtp_draft_buf", None) is None:
            self._mtp_draft_buf = self._mtp_alloc("draft", words)
        self.gpu.copy_span(logits, 0, self._mtp_draft_buf, index * self.vocab, self.vocab)
        return Dev(self._mtp_draft_buf, self.vocab, index * self.vocab)

    def _mtp_hold_rows(self, logits: Dev, count: int) -> Dev:
        words = count * self.vocab
        if getattr(self, "_mtp_span_buf", None) is None or self._mtp_span_n < words:
            self._mtp_span_buf = self._mtp_alloc("span", 4 * self.vocab)
            self._mtp_span_n = 4 * self.vocab
        self.gpu.copy_span(logits, 0, self._mtp_span_buf, 0, words)
        self.gpu.flush()
        return Dev(self._mtp_span_buf, words)

    def _mtp_row(self, src: Dev, index: int) -> Dev:
        if getattr(self, "_mtp_work", None) is None:
            self._mtp_work = self._mtp_alloc("row", self.vocab)
        self.gpu.copy_span(src, index * self.vocab, self._mtp_work, 0, self.vocab)
        return Dev(self._mtp_work, self.vocab)

    def _mtp_propose(
        self,
        first: int,
        produced: list[int],
        extras: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[list[int], list[Dev]]:
        drafts: list[int] = []
        views: list[Dev] = []
        self._mtp_draft_clean = False
        self.gpu.begin_scratch(scratch_bytes(1), copies=2)
        try:
            chosen = [first]
            for _step in range(min(getattr(self, "_draft_limit", 3), extras)):
                self._check_stop()
                if self._draft_blocked(produced + chosen):
                    break
                logits = self._mtp_step(chosen[-1], reuse=True, head=True)
                if logits is None:
                    break
                saved = self._mtp_hold_draft(len(drafts), logits)
                prepared = self._prepare(saved, temperature, produced + chosen)
                if not isinstance(prepared, Dev):
                    break
                guess = self.gpu.sample(prepared, temperature, top_p, top_k, self.rng)
                if guess in self.stop_ids:
                    break
                drafts.append(guess)
                views.append(prepared)
                chosen.append(guess)
                if self.required_function and "</tool_call>" in self.decode(produced + chosen):
                    break
            else:
                # Every drafted step was kept. One more step records the last
                # guess, so an accepted span does not replay the draft.
                if drafts:
                    self._mtp_step(drafts[-1], reuse=True, head=False)
                self._mtp_draft_clean = True
            return drafts, views
        finally:
            if not self._mtp_draft_clean:
                self._mtp_restore_kv()
            span = self.branches * self.hidden
            self._mtp_copy(self._mtp_hidden_snap, self._mtp_hidden, span)
            self._mtp_index_n = 0
            # The backbone forward may replace this scratch. Finish the draft first.
            self.gpu.flush()

    def _mtp_replay(self, kept: list[int]) -> None:
        span = self.branches * self.hidden
        self._mtp_copy(self._mtp_hidden, self._mtp_backbone, span)
        self._mtp_restore_kv()
        self._mtp_copy(self._mtp_hidden_snap, self._mtp_hidden, span)
        self._mtp_index_n = 0
        for token in kept:
            self._mtp_step(token, reuse=False, head=False)
        self._mtp_copy(self._mtp_backbone, self._mtp_hidden, span)
        self.gpu.flush()

    def _mtp_cycle(
        self,
        first: int,
        produced: list[int],
        room: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[int]:
        """One backbone forward of the drafted span. Rejected tokens are not committed."""
        started = time.perf_counter()
        # The previous span's experts load while the draft steps run. Grow the
        # cache first so that load is not issued into an atlas about to drop.
        span_room = min(max(room, 1), getattr(self, "_draft_limit", 3) + 1)
        if self._predicted:
            self._make_room_for_cache(span_room)
            if self.atlas is not None:
                self._prefetch_decode()
        self._mtp_stopped = False
        self._mtp_draft_clean = False
        if not hasattr(self, "_mtp_full"):
            self._mtp_full = 0
            self._mtp_partial = 0
        span_n = self.branches * self.hidden
        self._mtp_copy(self._mtp_hidden, self._mtp_hidden_snap, span_n)
        self._mtp_snap_kv()
        self._mtp_index_n = 0
        drafts: list[int] = []
        views: list[Dev] = []
        opened = produced + [first]
        closed = bool(self.required_function) and "</tool_call>" in self.decode(opened)
        if room > 1 and not closed and self._span_experts() and not self._draft_blocked(opened):
            drafts, views = self._mtp_propose(
                first, produced, room - 1, temperature, top_p, top_k
            )
        span = [first, *drafts]

        def bill(count: int) -> None:
            elapsed = time.perf_counter() - started
            self._decode_tokens += count
            self._decode_seconds += elapsed
            resident = self.atlas.resident_count if self.atlas is not None else 0
            gpu_elapsed = getattr(self, "_last_gpu", 0.0)
            self._decode_steps.append((elapsed, gpu_elapsed, resident))
            self._note_decode(resident)
            self._note_draft_rate(count, elapsed)

        if len(drafts) == 0:
            self.forward([first], decode=True, account=False)
            self._mtp_hold_logits()
            self._mtp_replay([first])
            bill(1)
            return [first]
        logits = self.forward(span, decode=True, commit=False, account=False)
        held = self._mtp_hold_rows(logits, len(span))
        kept = [first]
        correction: int | None = None
        for index, draft in enumerate(drafts):
            row = self._mtp_row(held, index)
            prepared = self._prepare(row, temperature, produced + kept)
            if not isinstance(prepared, Dev):
                if prepared == draft:
                    kept.append(draft)
                    continue
                correction = int(prepared)
                break
            token, accepted = self.gpu.spec_accept(
                prepared, views[index], draft, temperature, top_p, top_k, self.rng
            )
            if not accepted:
                correction = token
                break
            kept.append(draft)
        if correction is None and len(kept) == len(span):
            self.tokens.extend(span)
            self._prefix_dirty = True
            self.last_logits = held.view((len(span) - 1) * self.vocab, self.vocab)
            self._mtp_hold_logits()
            if not self._mtp_draft_clean:
                self._mtp_replay(span)
            self._mtp_full += 1
            bill(len(span))
            return span
        commit = list(kept)
        if correction is not None and correction in self.stop_ids:
            self._mtp_stopped = True
        elif correction is not None:
            commit.append(correction)
        # The span already applied the kept tokens. Put that state back and
        # forward only the token that replaced a rejected guess.
        if (
            correction is not None
            and correction not in self.stop_ids
            and self._span_snap_count == len(span)
            and len(kept) >= 1
        ):
            self._restore_span_prefix(len(kept))
            self.tokens.extend(kept)
            self._prefix_dirty = True
            self.forward([correction], decode=True, account=False)
        else:
            if self._mtp_draft_clean:
                self._mtp_restore_kv()
            self._restore_recurrent()
            self.forward(commit, decode=True, account=False)
        self._mtp_hold_logits()
        self._mtp_replay(commit)
        self._mtp_partial += 1
        bill(len(commit))
        return commit

    def _mtp_hold_logits(self) -> None:
        """Keep the backbone logits off the scratch arena the draft reuses."""
        logits = self.last_logits
        if not isinstance(logits, Dev):
            return
        if logits.n > self.vocab and logits.n % self.vocab == 0:
            logits = logits.view(logits.n - self.vocab, self.vocab)
            self.last_logits = logits
        if self._mtp_logit_buf is None or self._mtp_logit_n < logits.n:
            buf = self.gpu.alloc(logits.n * 4)
            if buf is None:
                message = "MTP logit buffer was not allocated"
                raise RuntimeError(message)
            self._mtp_logit_buf = buf
            self._mtp_logit_n = logits.n
        self.gpu.copy_span(logits, 0, self._mtp_logit_buf, 0, logits.n)
        self.gpu.flush()
        self.last_logits = Dev(self._mtp_logit_buf, logits.n)

    def _forward_body(self, tokens: list[int]) -> Dev:
        count = len(tokens)
        self._span_snap_count = 0
        self._span_snap = False
        if self._span_snap:
            self._span_snap = self._ensure_tok_snaps()
        if self._span_snap:
            self._tok_kv_seen = set()
        self._reserve_caches(len(self.tokens) + count)
        residual = self._embed_dev(tokens)
        span = self.branches * self.hidden
        step = count if self._decoding else self.prefill_chunk
        wide = self._streaming_prefill and count > self.prefill_chunk
        if wide:
            self._ensure_hold(count)
        for layer in range(self.layers):
            self._check_stop()
            submitted = False
            bundle = None
            if wide and layer + 1 < self.layers:
                self._submit_prefill_layer(layer + 1)
                submitted = True
            for offset in range(0, count, step):
                n = min(step, count - offset)
                view = residual.view(offset * span, n * span)
                self.gpu.rewind_scratch()
                # Objective-C temporaries from this chunk. The command buffer stays
                # open until prefill needs expert ids, the host reads the token, or
                # the next n-gram chunk reuses this scratch.
                with objc.autorelease_pool():
                    chunk_tokens = tokens[offset : offset + n]
                    if (layer + 1) in self.ple_layers:
                        self._ple_dev(view, chunk_tokens, n, prior=tokens[:offset])
                    bundle = self._layer(layer)
                    hats = self._rms_branches(view, bundle["attn_norm"], n)
                    xs = self._gr_dev(bundle["attn_down"], bundle["attn_up"], hats, n)
                    if self.kinds[layer] == "linear_attention":
                        mixed = self._gdn_dev(layer, bundle, xs, n)
                    elif self.kinds[layer] == "full_attention":
                        mixed = self._qsa_dev(
                            layer, bundle, xs, n, base=len(self.tokens) + offset
                        )
                    else:
                        message = f"unknown layer type {self.kinds[layer]}"
                        raise RuntimeError(message)
                    self._inject_dev(view, hats, mixed, bundle["attn_inject"], n)
                    hats = self._rms_branches(view, bundle["mlp_norm"], n)
                    xs = self._gr_dev(bundle["mlp_down"], bundle["mlp_up"], hats, n)
                    if wide:
                        self.gpu.copy_bytes(
                            xs.buf,
                            xs.offset * 4,
                            self._hold_xs,
                            offset * self.hidden * 4,
                            n * self.hidden * 4,
                        )
                        self.gpu.barrier()
                    else:
                        mixed = self._moe_dev(
                            layer,
                            bundle,
                            xs,
                            n,
                            submit_next=self._streaming_prefill and not submitted,
                        )
                        submitted = True
                        if self._bf16:
                            self.expert_cache.clear()
                        self._inject_dev(view, hats, mixed, bundle["mlp_inject"], n)
                    if not self._decoding:
                        self._note_prefill(layer, offset + n, count)
                    if self._decoding and self.atlas is not None:
                        self.gpu.submit()
                # Packed n-gram loads land in this scratch arena. The next rewind
                # reuses those offsets, so this chunk finishes before that rewind.
                if wide and (layer + 1) in self.ple_layers:
                    self.gpu.flush()
            if wide:
                if bundle is None:
                    message = "prefill layer had no tokens"
                    raise RuntimeError(message)
                self._moe_wide(layer, bundle, residual, count)
        if self._mtp_on and not self._decoding and tokens:
            self._mtp_prefill(residual, tokens)
        if self._mtp_on and tokens:
            self._mtp_keep_residual(residual, count)
        self.gpu.rewind_scratch()
        final = self._final_bundle()
        positions = count if self._decoding and count > 1 else 1
        if positions == 1:
            last = residual.view(
                (count - 1) * self.branches * self.hidden, self.branches * self.hidden
            )
        else:
            last = residual
        hat = self._rms_branches(last, final["norm"], positions)
        hidden = self._gr_dev(final["down"], final["up"], hat, positions)
        if self._span_snap:
            self._span_snap_count = count
        return self.gpu.linear(self._lm_head(), hidden, positions, self.hidden)

    def _rerun_decode(self, tokens: list[int]) -> Dev:
        """Load whatever this token routed to, then run it again from the snapshot."""
        self._atlas_trusted = False
        logits = self.last_logits
        for _ in range(self.layers):
            self._reload_misses()
            if self.atlas is not None:
                _fill_buffer(self.atlas.miss, self.layers * 4, 0)
            self._restore_recurrent()
            logits = self._forward_body(tokens)
            self.gpu.flush()
            if not self._expert_miss():
                return logits
        self._force_sync = True
        if self.atlas is not None:
            _fill_buffer(self.atlas.miss, self.layers * 4, 0)
        self._restore_recurrent()
        logits = self._forward_body(tokens)
        self.gpu.flush()
        self._force_sync = False
        if self._expert_miss():
            message = "expert atlas missed after reloading this token"
            raise RuntimeError(message)
        return logits

    def _prepare_experts(self, *, decode: bool, count: int) -> None:
        if self._bf16:
            return
        if decode or count <= self.prefill_chunk:
            # Cold and warm prompts that fit in one chunk miss into the atlas.
            # Decode then reads those images from RAM.
            self._ensure_atlas()
            return
        self._begin_prefill_stream()

    def _cache_covers(self, tokens: int) -> bool:
        if tokens <= 0 or not self.kv_k:
            return True
        for layer, buf in self.kv_k.items():
            need, _stride = self._kv_span(tokens)
            value = self.kv_v.get(layer)
            indexer = self.indexer_buf.get(layer)
            if buf is None or int(buf.length()) < need:
                return False
            if value is None or int(value.length()) < need:
                return False
            if indexer is None or int(indexer.length()) < tokens * INDEX_BF16_BYTES:
                return False
        return True

    def _make_room_for_cache(self, extra: int) -> None:
        """Grow the cache beside the atlas.

        When the new cache buffers do not fit, victim runs are discarded and
        the atlas stays mapped.
        """
        end = len(self.tokens) + max(0, extra)
        if self._cache_covers(end):
            return
        try:
            self._grow_cache_to(end)
        except RuntimeError:
            if self.atlas is None:
                raise
            self._evict_atlas_victims(end)
            self._grow_cache_to(end)

    def _grow_cache_to(self, end: int) -> None:
        for layer in list(self.kv_k):
            length = int(self.kv_len.get(layer, 0))
            target = max(end, length)
            if target > length:
                self._grow_kv(layer, length, target - length)

    def _evict_atlas_victims(self, end: int) -> None:
        need = self._pending_cache_bytes(end)
        protect: set[tuple[int, int]] = set(getattr(self, "_prefetch_protect", ()) or ())
        for layer, ids in getattr(self, "_predicted", {}).items():
            protect.update((layer, int(expert)) for expert in ids)
        self.gpu.flush()
        self.gpu.finish_io()
        if self.atlas is None:
            return
        self.atlas.release_unused()
        freed = self.atlas.evict_bytes(need, protect)
        if freed < need:
            self.atlas.evict_bytes(need - freed, set())

    def _pending_cache_bytes(self, end: int) -> int:
        total = 0
        for layer in list(self.kv_k):
            length = int(self.kv_len.get(layer, 0))
            target = max(end, length)
            _used, stride = self._kv_span(length)
            need, _stride = self._kv_span(target)
            total += _growth_alloc(self.kv_k.get(layer), need, stride)
            total += _growth_alloc(self.kv_v.get(layer), need, stride)
            total += _growth_alloc(
                self.indexer_buf.get(layer),
                target * INDEX_BF16_BYTES,
                INDEX_BF16_BYTES,
            )
        return total

    def _live_atlas_candidates(self) -> tuple[int, ...]:
        self.gpu.flush()
        gc.collect()
        layers = sum(kind == "full_attention" for kind in self.kinds)
        if self._mtp_on:
            layers += 1
        per_token = kv_bytes_per_token(layers, cache=self.kv_cache)
        growth = cache_growth_bytes(len(self.tokens), per_token)
        budget = atlas_budget(
            memory=machine_memory(),
            content_bytes=getattr(self, "_tier_bytes", 0),
            footprint=process_footprint(),
            growth=growth,
        )
        return atlas_fallbacks(budget)

    def _ensure_atlas(self) -> None:
        if self._bf16 or self.atlas is not None:
            return
        if not isinstance(self.files, QuantRecord):
            return
        candidates = self.resident_candidates
        if not candidates:
            candidates = self._live_atlas_candidates()
            self.resident_budget = candidates[0]
            _log(
                f"resident atlas {self.resident_budget >> 20} MiB "
                f"for {len(self.tokens)} tokens"
            )
        self.atlas = ExpertAtlas(
            self.gpu, self.files, self.layers, self.experts, candidates
        )

    def _stage_budget(self) -> int:
        if not isinstance(self.files, QuantRecord):
            return 0
        budget = 0
        for layer in range(self.layers):
            hot_bytes, cold_bytes, _table = prefill_image_table(
                self.files.hot[layer],
                self.files.super[layer],
                self.experts,
                self.files.slot,
                self.files.cold_slot,
            )
            budget = max(budget, hot_bytes + cold_bytes)
        return budget

    def _release_atlas(self) -> None:
        """Drop the decode atlas so the next allocation can use that memory.

        A pending Metal IO command retains its destination. Wait for those
        commands before the Python reference is the last one.
        """
        atlas = getattr(self, "atlas", None)
        self.atlas = None
        if atlas is None:
            return
        gpu = getattr(self, "gpu", None)
        if gpu is None:
            return
        gpu.flush()
        gpu.finish_io()
        discard_buffer(getattr(atlas, "atlas", None))
        discard_buffer(getattr(atlas, "table", None))
        discard_buffer(getattr(atlas, "miss", None))
        for name in ("atlas", "table", "miss"):
            if hasattr(atlas, name):
                setattr(atlas, name, None)
        del atlas
        gc.collect()

    def _shrink_atlas(self, nbytes: int) -> None:
        """Evict the least-fired tail until ``nbytes`` of images are released.

        The atlas buffer stays. Runs that have been routed again stay until the
        tail is gone.
        """
        if self.atlas is None or nbytes <= 0:
            return
        self.gpu.flush()
        self.gpu.finish_io()
        self.atlas.release_unused()
        self.atlas.evict_bytes(nbytes, set())

    def _alloc_stage_pair(self, budget: int) -> list[Any]:
        stage = [self.gpu.alloc(budget), self.gpu.alloc(budget)]
        if stage[0] is None or stage[1] is None:
            for buf in stage:
                if buf is not None:
                    discard_buffer(buf)
            gc.collect()
            return [None, None]
        return stage

    def _begin_prefill_stream(self) -> None:
        """Two staging buffers. The atlas stays, and its tail is evicted until they fit."""
        if not isinstance(self.files, QuantRecord):
            return
        self._end_prefill_stream()
        self._hot_keep = None
        budget = 0
        for layer in range(self.layers):
            hot_bytes, cold_bytes, _table = self._prefill_layout(layer)
            budget = max(budget, hot_bytes + cold_bytes)
        self._stage = self._alloc_stage_pair(budget)
        if self._stage[0] is None and self.atlas is not None:
            self._shrink_atlas(budget * 2)
            self._stage = self._alloc_stage_pair(budget)
        if self._stage[0] is None and self.atlas is not None and self.atlas.resident_count:
            self._shrink_atlas(self.atlas.budget)
            self._stage = self._alloc_stage_pair(budget)
        if self._stage[0] is None or self._stage[1] is None:
            message = f"prefill stage of {budget} bytes was not allocated"
            raise RuntimeError(message)
        release_buffer(self._stage[0])
        release_buffer(self._stage[1])
        self._stage_table = self.gpu.alloc(self.experts * 8)
        self._stage_miss = self.gpu.zeros(self.layers * 4)
        self._stage_event = [0, 0]
        self._stage_bytes = 0
        self._streaming_prefill = True
        self._submit_prefill_layer(0)

    def _prefill_layout(self, layer: int) -> tuple[int, int, np.ndarray]:
        """Offsets in the staging buffer. A resident hot file is omitted from that buffer."""
        if not isinstance(self.files, QuantRecord):
            message = "prefill layout requires a quantized record"
            raise RuntimeError(message)
        hot_bytes, cold_bytes, table = prefill_image_table(
            self.files.hot[layer],
            self.files.super[layer],
            self.experts,
            self.files.slot,
            self.files.cold_slot,
        )
        if self._hot_keep is None:
            return hot_bytes, cold_bytes, table
        adjusted = np.array(table, copy=True)
        hot_here = (adjusted >= 0) & (adjusted < hot_bytes)
        adjusted[hot_here] = -1
        cold_here = adjusted >= hot_bytes
        adjusted[cold_here] = adjusted[cold_here] - hot_bytes
        return 0, cold_bytes, adjusted

    def _end_prefill_stream(self) -> None:
        stage = self._stage
        self._streaming_prefill = False
        self._stage = None
        self._stage_table = None
        self._stage_miss = None
        self._stage_event = [0, 0]
        self._hot_keep = None
        self._hot_fence = 0
        if stage is None:
            return
        # The layer loads were submitted into these buffers. Finish that IO
        # before dropping them, or the next atlas is allocated beside them.
        # Discard the pages first so the dirty staging window is not swapped.
        self.gpu.flush()
        self.gpu.finish_io()
        for buf in stage:
            if buf is not None:
                discard_buffer(buf)
        del stage
        gc.collect()

    def _submit_prefill_layer(self, layer: int) -> None:
        if not isinstance(self.files, QuantRecord) or self._stage is None:
            message = "prefill stream has no tier files"
            raise RuntimeError(message)
        hot_bytes, cold_bytes, _table = self._prefill_layout(layer)
        buf = self._stage[layer % 2]
        hot_size = os.fstat(self.files.hot_fds[layer]).st_size
        cold_size = os.fstat(self.files.cold_fds[layer]).st_size
        if hot_size < hot_bytes or cold_size < cold_bytes:
            message = f"layer {layer} tier file is shorter than the image table"
            raise RuntimeError(message)
        used = hot_bytes + cold_bytes
        if used % PAGE:
            message = f"layer {layer} stage of {used} bytes is not a multiple of {PAGE}"
            raise RuntimeError(message)
        if used:
            commit_pages(buf, 0, used)
        groups: list[tuple[Any, list[tuple[Any, int, int, int]]]] = []
        if hot_bytes:
            groups.append((self.files.hot_io[layer], [(buf, 0, hot_bytes, 0)]))
        if cold_bytes:
            groups.append((self.files.cold_io[layer], [(buf, hot_bytes, cold_bytes, 0)]))
        self._stage_event[layer % 2] = self.gpu.io_submit_mixed(groups)
        self._stage_bytes += hot_bytes + cold_bytes

    def _write_stage_table(self, layer: int) -> None:
        if not isinstance(self.files, QuantRecord) or self._stage_table is None:
            message = "prefill stream has no offset table"
            raise RuntimeError(message)
        _hot, _cold, table = self._prefill_layout(layer)
        raw = np.ascontiguousarray(table, dtype=np.int64).tobytes()
        _buffer_view(self._stage_table, len(raw))[: len(raw)] = raw

    def _stage_missed(self) -> bool:
        if self._stage_miss is None:
            return False
        raw = _buffer_bytes(self._stage_miss, self.layers * 4)
        return bool(np.frombuffer(raw, dtype=np.uint32).any())

    def _expert_miss(self) -> bool:
        if self.atlas is None:
            return False
        raw = _buffer_bytes(self.atlas.miss, self.layers * 4)
        return bool(np.frombuffer(raw, dtype=np.uint32).any())

    def _prefetch_decode(self) -> None:
        """Queue the previous span's experts before this token's shaders start."""
        earlier = dict(getattr(self, "_prefetch_events", {}))
        self._prefetch_events = {}
        if self.atlas is None or not self._predicted:
            self._prefetch_protect = set()
            return
        protect = {
            (layer, expert) for layer, ids in self._predicted.items() for expert in ids
        }
        self._prefetch_protect = protect
        for layer, ids in self._predicted.items():
            try:
                event = self.atlas.stage(layer, ids, protect)
            except RuntimeError:
                # The forward stages what it needs and can evict the tail.
                # Pinning every predicted id here is what fills the atlas.
                self._prefetch_protect = set()
                return
            queued = earlier.get(layer, 0)
            if queued > event:
                event = queued
            self._prefetch_events[layer] = event

    def _capture_recurrent(self) -> None:
        if not getattr(self, "_gdn_shadow", None):
            self._kv_snap = None
            return
        self._kv_snap = {
            index: (
                self.kv_k[index],
                self.kv_v[index],
                self.kv_k_tail[index],
                self.kv_v_tail[index],
                self.indexer_buf[index],
                self.kv_len[index],
            )
            for index in self.kv_len
        }
        state_bytes = 48 * 128 * 128 * 4
        for layer, buf in self.gdn_state.items():
            self.gpu.copy_bytes(buf, 0, self._gdn_shadow[layer], 0, state_bytes)
        for layer, buf in self.conv_hist.items():
            self.gpu.copy_bytes(buf, 0, self._conv_shadow[layer], 0, self._conv_bytes)
        self.gpu.copy_bytes(self.ple_hist, 0, self._ple_shadow, 0, self._ple_bytes)
        # The store rewrites this tail in place. The length snap does not keep the bytes.
        tail_bytes = KV_TAIL_TOKENS * KV_BF16_BYTES
        for index, buf in self.kv_k_tail.items():
            self.gpu.copy_bytes(buf, 0, self._k_tail_shadow[index], 0, tail_bytes)
        for index, buf in self.kv_v_tail.items():
            self.gpu.copy_bytes(buf, 0, self._v_tail_shadow[index], 0, tail_bytes)
        self.gpu.barrier()

    def _restore_recurrent(self) -> None:
        for index, (key, value, key_tail, value_tail, indexer, length) in self._kv_snap.items():
            self.kv_k[index] = key
            self.kv_v[index] = value
            self.kv_k_tail[index] = key_tail
            self.kv_v_tail[index] = value_tail
            self.indexer_buf[index] = indexer
            self.kv_len[index] = length
        state_bytes = 48 * 128 * 128 * 4
        for layer, buf in self.gdn_state.items():
            self.gpu.copy_bytes(self._gdn_shadow[layer], 0, buf, 0, state_bytes)
        for layer, buf in self.conv_hist.items():
            self.gpu.copy_bytes(self._conv_shadow[layer], 0, buf, 0, self._conv_bytes)
        self.gpu.copy_bytes(self._ple_shadow, 0, self.ple_hist, 0, self._ple_bytes)
        tail_bytes = KV_TAIL_TOKENS * KV_BF16_BYTES
        for index, buf in self.kv_k_tail.items():
            self.gpu.copy_bytes(self._k_tail_shadow[index], 0, buf, 0, tail_bytes)
        for index, buf in self.kv_v_tail.items():
            self.gpu.copy_bytes(self._v_tail_shadow[index], 0, buf, 0, tail_bytes)
        self.gpu.barrier()

    def _reload_misses(self) -> None:
        if self.atlas is None:
            return
        pending: list[tuple[int, list[int]]] = []
        protect: set[tuple[int, int]] = set()
        for layer, choice in self._choice.items():
            ids = [int(expert) for expert in self.gpu.read_u32(choice).tolist()]
            pending.append((layer, ids))
            protect.update((layer, expert) for expert in ids)
        for layer, ids in pending:
            event = self.atlas.stage(layer, ids, protect)
            self.gpu.io_fence(event)

    def _load_record(self) -> None:
        if not isinstance(self.files, QuantRecord):
            message = "quantized load has no record"
            raise RuntimeError(message)
        size = (self.files.root / "resident.bin").stat().st_size
        progress = None
        if self.console is not None and not self.console.keeps_lines:
            detail = getattr(self.console, "load_detail", "") or "record"
            progress = self.console.begin("load", total=size, unit="bytes", detail=detail)
            self._load_progress = progress
        else:
            _log("loading quantized record")
        self.files.open_io(self.gpu)
        manifest = self.files.manifest
        self._resident = self.gpu.alloc(size)
        self.gpu.io_read(self.files.resident_io, [(self._resident, 0, size, 0)])
        if progress is not None:
            progress.done = size
        for key, meta in manifest["int4"].items():
            if key == manifest["embed_key"]:
                self._embed_meta = meta
                continue
            rows = int(meta["rows"])
            offset = int(meta["offset"])
            self.weights[key] = Int4Matrix(
                rows,
                int(meta["logical_inner"]),
                ((rows + 31) // 32) * 32,
                self._resident,
                self._resident,
                int(meta["stored_inner"]),
                offset,
                offset + int(meta["data"]),
            )
        for key, meta in manifest["f16"].items():
            self.f16_weights[key] = F16Matrix(
                int(meta["rows"]), int(meta["inner"]), self._resident, int(meta["offset"])
            )
        self._f32_meta = manifest["f32"]
        for key, meta in manifest["i64"].items():
            count = int(np.prod(meta["shape"]))
            # Address constants for the host n-gram hash. The resident load
            # already placed these bytes in the shared buffer.
            blob = _buffer_slice(self._resident, int(meta["offset"]), count * 8)
            self.i64[key] = np.frombuffer(blob, dtype=np.int64).copy().reshape(meta["shape"])
        summary = f"loaded {len(self.weights)} fp4 matrices"
        if progress is not None and self.console is not None:
            self._load_progress = None
            self.console.end(progress, summary)
        else:
            _log(summary)

    def _load_bf16(self) -> None:
        """Upload the checkpoint's own values. Experts and n-gram rows stay on disk."""
        store = self.store
        if store is None:
            message = "BF16 load has no shard store"
            raise RuntimeError(message)
        _check_f32_gemv(self.gpu)
        progress = None
        if self.console is not None and not self.console.keeps_lines:
            progress = self.console.begin(
                "load", total=len(self.kinds), unit="layers", detail="bf16"
            )
            self._load_progress = progress
        else:
            _log("loading BF16 checkpoint onto Metal")
        self._head = self.gpu.load_f32(store.matrix("lm_head.weight"))
        for layer, kind in enumerate(self.kinds):
            prefix = f"model.language_model.layers.{layer}"
            for key in _layer_int4_keys(prefix, kind):
                self.weights[key] = self.gpu.load_f32(store.matrix(key))
            gate_key = f"{prefix}.mlp.gate.weight"
            self.f16_weights[gate_key] = self.gpu.load_f32(store.matrix(gate_key))
            for key in _layer_f32_keys(prefix, kind):
                self.f32[key] = store.matrix(key)
            if progress is not None:
                progress.done = layer + 1
            else:
                _log(f"bf16 layer {layer}")
        ple = "model.language_model.layers.1.ple"
        table = f"{ple}.ple_embedding"
        for key in (f"{ple}.key_proj.weight", f"{ple}.value_proj.weight"):
            self.weights[key] = self.gpu.load_f32(store.matrix(key))
        for key in (
            f"{ple}.conv1d.weight",
            f"{ple}.norm_conv.weight",
            f"{ple}.norm_key.weight",
            f"{ple}.norm_query.weight",
        ):
            self.f32[key] = store.matrix(key)
        for key in (
            f"{table}.layer_multipliers",
            f"{table}.ngram_heads_offsets",
            f"{table}.ngram_heads_vocab_sizes",
        ):
            self.i64[key] = store.i64(key)
        mixer = "model.language_model.hyper_connection_mixer"
        self.f32[f"{mixer}.hc_norm.weight"] = store.matrix(f"{mixer}.hc_norm.weight")
        for key in (
            f"{mixer}.input_mix_weight_down.weight",
            f"{mixer}.input_mix_weight_up.weight",
        ):
            self.weights[key] = self.gpu.load_f32(store.matrix(key))
        summary = f"loaded {len(self.weights)} bf16 matrices"
        if progress is not None and self.console is not None:
            self._load_progress = None
            self.console.end(progress, summary)
        else:
            _log(summary)

    def _embed_dev(self, tokens: list[int]) -> Dev:
        count = len(tokens)
        span = self.branches * self.hidden
        dst = self._residual(count)
        if self._bf16:
            if self.store is None:
                message = "BF16 embed has no shard store"
                raise RuntimeError(message)
            key = "model.language_model.embed_tokens.weight"
            for index, token in enumerate(tokens):
                vector = self.store.read_bf16(
                    key, token * self.hidden, self.hidden, (self.hidden,)
                )
                self.gpu.broadcast(
                    self.gpu.dev_from(vector), dst, index * span, self.branches, self.hidden
                )
            return dst
        meta = self._embed_meta
        stored = int(meta["stored_inner"])
        self.gpu.embed_gather(
            self._resident,
            int(meta["offset"]),
            self.gpu.dev_u32(tokens),
            dst,
            count,
            self.hidden,
            stored // 2,
            stored // BLOCK,
            int(meta["data"]),
            self.branches,
        )
        return dst

    def _rms_branches(self, residual: Dev, binding: tuple[Any, int], count: int) -> Dev:
        weight, offset = binding
        return self.gpu.rms(
            residual,
            weight,
            offset,
            self.hidden,
            count * self.branches,
            self.eps,
            True,
            self.branches,
        )

    def _residual(self, count: int) -> Dev:
        n = count * self.branches * self.hidden
        if self._residual_buf is None or self._residual_n < n:
            buf = self.gpu.alloc(n * 4)
            if buf is None:
                message = f"residual of {n * 4} bytes was not allocated"
                raise RuntimeError(message)
            self._residual_buf = buf
            self._residual_n = n
        return Dev(self._residual_buf, n)

    def _reserve_caches(self, total: int) -> None:
        for layer, length in self.kv_len.items():
            extra = total - length
            if extra <= 0:
                continue
            self._grow_kv(layer, length, extra)

    def _gr_dev(self, down: Any, up: Any, hats: Dev, count: int) -> Dev:
        # Tech report eq. (31): SiLU((1/n_r) W_d vec(R̂)). The branch mean is the second 1/n_r.
        flat = self.branches * self.hidden
        hidden = self.gpu.linear(
            down, hats, count, flat, epilogue=3, epilogue_scale=1.0 / self.branches
        )
        gate = self.gpu.linear(up, hidden, count, int(up.inner), epilogue=2)
        return self.gpu.branch_mean(gate, hats, count, self.branches, self.hidden)

    def _inject_dev(self, residual: Dev, hats: Dev, mixed: Dev, weight: Any, count: int) -> None:
        flat = self.branches * self.hidden
        scales = self.gpu.linear(weight, hats, count, flat, epilogue=4)
        self.gpu.residual_write(residual, scales, mixed, count, self.branches, self.hidden)

    def _layer(self, layer: int) -> dict[str, Any]:
        cached = self._bundle.get(layer)
        if cached is not None:
            return cached
        prefix = f"model.language_model.layers.{layer}"
        bundle: dict[str, Any] = {
            "attn_norm": self._weight(f"{prefix}.attn_hyper_connection.hc_norm.weight"),
            "attn_down": self._int4(f"{prefix}.attn_hyper_connection.input_mix_weight_down.weight"),
            "attn_up": self._int4(f"{prefix}.attn_hyper_connection.input_mix_weight_up.weight"),
            "attn_inject": self._int4(f"{prefix}.attn_hyper_connection.block_inject_weight.weight"),
            "mlp_norm": self._weight(f"{prefix}.mlp_hyper_connection.hc_norm.weight"),
            "mlp_down": self._int4(f"{prefix}.mlp_hyper_connection.input_mix_weight_down.weight"),
            "mlp_up": self._int4(f"{prefix}.mlp_hyper_connection.input_mix_weight_up.weight"),
            "mlp_inject": self._int4(f"{prefix}.mlp_hyper_connection.block_inject_weight.weight"),
            "shared_gate": self._int4(f"{prefix}.mlp.shared_expert.gate_proj.weight"),
            "shared_up": self._int4(f"{prefix}.mlp.shared_expert.up_proj.weight"),
            "shared_down": self._int4(f"{prefix}.mlp.shared_expert.down_proj.weight"),
            "shared_coeff": self._int4(f"{prefix}.mlp.shared_expert_gate.weight"),
        }
        if self.kinds[layer] == "linear_attention":
            bundle.update(
                {
                    "qkv": self._int4(f"{prefix}.linear_attn.in_proj_qkv.weight"),
                    "z": self._int4(f"{prefix}.linear_attn.in_proj_z.weight"),
                    "a": self._int4(f"{prefix}.linear_attn.in_proj_a.weight"),
                    "b": self._int4(f"{prefix}.linear_attn.in_proj_b.weight"),
                    "out": self._int4(f"{prefix}.linear_attn.out_proj.weight"),
                    "conv": self._weight(f"{prefix}.linear_attn.conv1d.weight"),
                    "a_log": self._weight(f"{prefix}.linear_attn.A_log"),
                    "dt_bias": self._weight(f"{prefix}.linear_attn.dt_bias"),
                    "norm": self._weight(f"{prefix}.linear_attn.norm.weight"),
                }
            )
        else:
            bundle.update(
                {
                    "q_proj": self._int4(f"{prefix}.self_attn.q_proj.weight"),
                    "k_proj": self._int4(f"{prefix}.self_attn.k_proj.weight"),
                    "v_proj": self._int4(f"{prefix}.self_attn.v_proj.weight"),
                    "o_proj": self._int4(f"{prefix}.self_attn.o_proj.weight"),
                    "q_norm": self._weight(f"{prefix}.self_attn.q_norm.weight"),
                    "k_norm": self._weight(f"{prefix}.self_attn.k_norm.weight"),
                    "index_qk": self._int4(f"{prefix}.self_attn.indexer.index_qk_proj.weight"),
                }
            )
        self._bundle[layer] = bundle
        return bundle

    def _weight(self, key: str) -> tuple[Any, int]:
        if self._bf16:
            cached = self._f32_uploaded.get(key)
            if cached is None:
                array = np.ascontiguousarray(self.f32[key], dtype=np.float32)
                cached = self.gpu.upload(array.tobytes())
                self._f32_uploaded[key] = cached
            return cached, 0
        meta = self._f32_meta[key]
        return self._resident, int(meta["offset"])

    def _int4(self, key: str) -> Int4Matrix:
        return self.weights[key]

    def _span_choice_buf(self) -> Any:
        words = self.layers * 4 * self.top_k
        buf = getattr(self, "_span_choice", None)
        if buf is None:
            buf = self.gpu.alloc(words * 4)
            if buf is None:
                message = "span expert ids were not allocated"
                raise RuntimeError(message)
            self._span_choice = buf
        return buf

    def _ensure_tok_snaps(self) -> bool:
        # Four copies of the gated-delta state are about 432 MiB. That does not
        # fit beside the resident atlas on a 48 GiB machine.
        return False

    def _mark_gdn_token(self, layer: int, token: int) -> None:
        self.gpu.barrier()
        state_bytes = 48 * 128 * 128 * 4
        self.gpu.copy_bytes(self.gdn_state[layer], 0, self._tok_gdn[layer][token], 0, state_bytes)
        self.gpu.copy_bytes(
            self.conv_hist[layer], 0, self._tok_conv[layer][token], 0, self._conv_bytes
        )
        self.gpu.barrier()

    def _mark_kv_token(self, layer: int, token: int, origin: int) -> None:
        need = ((origin & ~3) + 4) * KV_E4M3_BYTES
        if int(self.kv_k[layer].length()) < need or int(self.kv_v[layer].length()) < need:
            self._span_snap = False
            return
        self.gpu.barrier()
        tail_bytes = KV_TAIL_TOKENS * KV_BF16_BYTES
        body_bytes = 4 * KV_E4M3_BYTES
        key_tail, value_tail, key_body, value_body = self._tok_kv[layer][token]
        self.gpu.copy_bytes(self.kv_k_tail[layer], 0, key_tail, 0, tail_bytes)
        self.gpu.copy_bytes(self.kv_v_tail[layer], 0, value_tail, 0, tail_bytes)
        start = (origin & ~3) * KV_E4M3_BYTES
        self.gpu.copy_bytes(self.kv_k[layer], start, key_body, 0, body_bytes)
        self.gpu.copy_bytes(self.kv_v[layer], start, value_body, 0, body_bytes)
        self._tok_kv_len[layer][token] = self.kv_len[layer]
        self._tok_origin[layer] = origin
        self._tok_kv_seen.add(layer)
        self.gpu.barrier()

    def _mark_ple_token(self, token: int) -> None:
        if not getattr(self, "_span_snap", False):
            return
        self.gpu.barrier()
        self.gpu.copy_bytes(self.ple_hist, 0, self._tok_ple[token], 0, self._ple_bytes)
        self.gpu.barrier()

    def _restore_span_prefix(self, tokens: int) -> None:
        """Recurrent state after the first ``tokens`` positions of the span."""
        index = tokens - 1
        state_bytes = 48 * 128 * 128 * 4
        for layer, buf in self.gdn_state.items():
            self.gpu.copy_bytes(self._tok_gdn[layer][index], 0, buf, 0, state_bytes)
            self.gpu.copy_bytes(
                self._tok_conv[layer][index], 0, self.conv_hist[layer], 0, self._conv_bytes
            )
        self.gpu.copy_bytes(self._tok_ple[index], 0, self.ple_hist, 0, self._ple_bytes)
        tail_bytes = KV_TAIL_TOKENS * KV_BF16_BYTES
        body_bytes = 4 * KV_E4M3_BYTES
        for layer in self._tok_kv_seen:
            slots = self._tok_kv[layer]
            key_tail, value_tail, key_body, value_body = slots[index]
            self.gpu.copy_bytes(key_tail, 0, self.kv_k_tail[layer], 0, tail_bytes)
            self.gpu.copy_bytes(value_tail, 0, self.kv_v_tail[layer], 0, tail_bytes)
            start = (self._tok_origin[layer] & ~3) * KV_E4M3_BYTES
            self.gpu.copy_bytes(key_body, 0, self.kv_k[layer], start, body_bytes)
            self.gpu.copy_bytes(value_body, 0, self.kv_v[layer], start, body_bytes)
            self.kv_len[layer] = self._tok_kv_len[layer][index]
        self.gpu.barrier()

    def _gdn_dev(self, layer: int, bundle: dict[str, Any], xs: Dev, count: int) -> Dev:
        hidden = self.hidden
        qkv = self.gpu.linear(bundle["qkv"], xs, count, hidden)
        z = self.gpu.linear(bundle["z"], xs, count, hidden)
        raw_a = self.gpu.linear(bundle["a"], xs, count, hidden)
        raw_b = self.gpu.linear(bundle["b"], xs, count, hidden)
        conv_w, conv_off = bundle["conv"]
        dt_w, dt_off = bundle["dt_bias"]
        a_w, a_off = bundle["a_log"]
        norm_w, norm_off = bundle["norm"]
        mixed = self.gpu.tmp(count * 10240)
        heads = self.gpu.tmp(count * 6144)
        if self._span_snap and count > 1:
            for token in range(count):
                self.gpu.gdn_conv_scan(
                    self.conv_hist[layer],
                    qkv.view(token * 10240, 10240),
                    conv_w,
                    conv_off,
                    mixed.view(token * 10240, 10240),
                    1,
                )
                self.gpu.gdn_state_scan(
                    self.gdn_state[layer],
                    mixed.view(token * 10240, 10240),
                    raw_a.view(token * 48, 48),
                    raw_b.view(token * 48, 48),
                    z.view(token * 6144, 6144),
                    dt_w,
                    dt_off,
                    a_w,
                    a_off,
                    norm_w,
                    norm_off,
                    heads.view(token * 6144, 6144),
                    1,
                    self.eps,
                )
                self._mark_gdn_token(layer, token)
        else:
            self.gpu.gdn_conv_scan(self.conv_hist[layer], qkv, conv_w, conv_off, mixed, count)
            self.gpu.gdn_state_scan(
                self.gdn_state[layer],
                mixed,
                raw_a,
                raw_b,
                z,
                dt_w,
                dt_off,
                a_w,
                a_off,
                norm_w,
                norm_off,
                heads,
                count,
                self.eps,
            )
        return self.gpu.linear(bundle["out"], heads, count, 6144)

    def _grow_slot(
        self, table: dict[int, Any], layer: int, length: int, width: int
    ) -> tuple[Any, int]:
        return self._reserve(table, layer, length, 1, width), length * width

    def _grow_kv(self, layer: int, length: int, extra: int) -> tuple[Any, Any, Any]:
        end = length + extra
        used, stride = self._kv_span(length)
        need, _stride = self._kv_span(end)
        key = self._reserve_bytes(self.kv_k, layer, used, need, stride)
        value = self._reserve_bytes(self.kv_v, layer, used, need, stride)
        indexer = self._reserve_bytes(
            self.indexer_buf,
            layer,
            length * INDEX_BF16_BYTES,
            end * INDEX_BF16_BYTES,
            INDEX_BF16_BYTES,
        )
        return key, value, indexer

    def _kv_span(self, tokens: int) -> tuple[int, int]:
        if self.kv_cache == "bf16":
            return tokens * KV_BF16_BYTES, KV_BF16_BYTES
        complete = (tokens // 4) * 4
        return complete * KV_E4M3_BYTES, KV_E4M3_BYTES

    def _reserve_bytes(
        self, table: dict[int, Any], layer: int, used: int, need: int, stride: int
    ) -> Any:
        buf = table[layer]
        if need <= 0:
            return buf
        if buf is not None and buf.length() >= need:
            return buf
        chunk = PREFILL_CHUNK * stride
        slack = 64 * stride
        alloc_n = max(need * 2, stride) if need <= chunk else need + slack
        grown = self.gpu.alloc(alloc_n)
        if grown is None:
            message = f"cache of {alloc_n} bytes was not allocated"
            raise RuntimeError(message)
        if used and buf is not None:
            self.gpu._retain.append(buf)
            self.gpu.copy_bytes(buf, 0, grown, 0, used)
        table[layer] = grown
        return grown

    def _store_kv(self, key: Dev, value: Dev, layer: int, count: int, base: int) -> None:
        tails = (
            key,
            value,
            self.kv_k[layer],
            self.kv_v[layer],
            self.kv_k_tail[layer],
            self.kv_v_tail[layer],
            count,
            base,
            self.kv_cache,
        )
        self.gpu.kv_store(*tails, 0)
        if self.kv_cache == "e4m3":
            # Phase 0 reads the unfinished tail into the completed block.
            # Phase 1 replaces that tail. The two dispatches share this encoder.
            self.gpu.barrier()
            self.gpu.kv_store(*tails, 1)

    def _reserve(
        self, table: dict[int, Any], layer: int, length: int, extra: int, width: int
    ) -> Any:
        need = (length + extra) * width * 4
        buf = table[layer]
        if buf is not None and buf.length() >= need:
            return buf
        # Short sequences keep the doubled buffer. A long prompt allocates the
        # request plus a few decode tokens, so the next step does not copy it.
        chunk = PREFILL_CHUNK * width * 4
        slack = 64 * width * 4
        alloc_n = max(need * 2, width * 4) if need <= chunk else need + slack
        grown = self.gpu.alloc(alloc_n)
        if grown is None:
            message = f"cache of {alloc_n} bytes was not allocated"
            raise RuntimeError(message)
        if length and buf is not None:
            self.gpu._retain.append(buf)
            self.gpu.copy_span(Dev(buf, length * width), 0, grown, 0, length * width)
        table[layer] = grown
        return grown

    def _qsa_dev(
        self,
        layer: int,
        bundle: dict[str, Any],
        xs: Dev,
        count: int,
        *,
        base: int,
        reuse: bool = False,
    ) -> Dev:
        return self._qsa_rows(layer, bundle, xs, count, base=base, reuse=reuse)

    def _qsa_rows(
        self,
        layer: int,
        bundle: dict[str, Any],
        xs: Dev,
        count: int,
        *,
        base: int,
        reuse: bool = False,
    ) -> Dev:
        hidden = self.hidden
        query = self.gpu.linear(bundle["q_proj"], xs, count, hidden)
        key = self.gpu.linear(bundle["k_proj"], xs, count, hidden)
        value = self.gpu.linear(bundle["v_proj"], xs, count, hidden)
        index_qk = self.gpu.linear(bundle["index_qk"], xs, count, hidden)
        q_w, q_off = bundle["q_norm"]
        k_w, k_off = bundle["k_norm"]
        q_part, gate_pre = self.gpu.split_q_batch(query, count)
        q = self.gpu.rms(q_part, q_w, q_off, 256, count * 24, self.eps, True, 0)
        k = self.gpu.rms(key, k_w, k_off, 256, count * 2, self.eps, True, 0)
        q = self.gpu.rope_batch(q, count, 24, 256, self.rotary_dim, base, self.rope_theta)
        k = self.gpu.rope_batch(k, count, 2, 256, self.rotary_dim, base, self.rope_theta)
        length = self.kv_len[layer]
        if length != base:
            self.gpu.flush()
            message = f"attention cache length {length} does not match position {base}"
            raise RuntimeError(message)
        k_buf, v_buf, i_buf = self._grow_kv(layer, length, count)
        if self._span_snap and count > 1:
            k_stride = k.n // count
            v_stride = value.n // count
            for token in range(count):
                self._store_kv(
                    k.view(token * k_stride, k_stride),
                    value.view(token * v_stride, v_stride),
                    layer,
                    1,
                    self.kv_len[layer],
                )
                self.kv_len[layer] += 1
                self._mark_kv_token(layer, token, length)
        else:
            self._store_kv(k, value, layer, count, length)
            self.kv_len[layer] = length + count
        self.gpu.indexer_write(index_qk, i_buf, count, length)
        # The attend in this encoder reads the keys and values just stored.
        self.gpu.barrier()
        complete = 0 if self.kv_cache == "bf16" else ((length + count) // 4) * 4
        mode = 0 if self.kv_cache == "bf16" else 1
        dense = base + count <= DENSE_LIMIT
        saved = reuse and 0 < self._mtp_index_n < INDEX_STRIDE and not dense
        if dense:
            if reuse:
                self._mtp_index_n = 0
            gated = self.gpu.qsa_attend(
                q,
                gate_pre,
                k_buf,
                v_buf,
                count,
                length,
                mode=mode,
                complete=complete,
                key_tail=self.kv_k_tail[layer],
                value_tail=self.kv_v_tail[layer],
            )
        elif saved:
            self.gpu.flush()
            for pos in range(base, base + count):
                if self._mtp_index_n >= INDEX_STRIDE:
                    break
                _poke_u32(self._mtp_indices, self._mtp_index_n, pos)
                self._mtp_index_n += 1
            _write_u32(self._mtp_count, self._mtp_index_n)
            gated = self._mtp_attend(
                q, gate_pre, k_buf, v_buf, count, mode, complete, layer
            )
        else:
            n_blocks = max((base + count - 1) // INDEX_BLOCK, 1)
            scores = self.gpu.tmp(count * n_blocks)
            indices = self.gpu.tmp(count * INDEX_STRIDE)
            selected = self.gpu.tmp(count)
            self.gpu.indexer_block_score(index_qk, i_buf, scores, count, length, n_blocks)
            self.gpu.indexer_select(
                scores, indices, selected, count, length, n_blocks, INDEX_STRIDE
            )
            if reuse:
                self.gpu.copy_bytes(
                    indices.buf,
                    indices.offset * 4,
                    self._mtp_indices,
                    0,
                    INDEX_STRIDE * 4,
                )
                self._mtp_index_n = int(self.gpu.read_u32(selected)[0])
                _write_u32(self._mtp_count, self._mtp_index_n)
                gated = self._mtp_attend(
                    q, gate_pre, k_buf, v_buf, count, mode, complete, layer
                )
            else:
                gated = self.gpu.qsa_attend_selected(
                    q,
                    gate_pre,
                    k_buf,
                    v_buf,
                    indices,
                    selected,
                    count,
                    INDEX_STRIDE,
                    mode=mode,
                    complete=complete,
                    key_tail=self.kv_k_tail[layer],
                    value_tail=self.kv_v_tail[layer],
                )
        return self.gpu.linear(bundle["o_proj"], gated, count, 6144)

    def _moe_dev(
        self,
        layer: int,
        bundle: dict[str, Any],
        xs: Dev,
        count: int,
        *,
        submit_next: bool = True,
    ) -> Dev:
        if self._bf16 or (self.atlas is None and not self._streaming_prefill):
            return self._moe_bf16(layer, bundle, xs, count)
        router = self._router(layer)
        choice, weight = self.gpu.topk_batch(
            self.gpu.linear(router, xs, count, self.hidden), count, self.experts, self.top_k
        )
        if count == 1 and not self._streaming_prefill:
            slot = layer * self.top_k
            self.gpu.copy_bytes(
                choice.buf, choice.offset * 4, self._choice_ids, slot * 4, self.top_k * 4
            )
            choice = Dev(self._choice_ids, self.top_k, slot)
        elif self._decoding and not self._streaming_prefill:
            slot = layer * self.top_k
            row_off = (count - 1) * self.top_k
            self.gpu.copy_bytes(
                choice.buf,
                (choice.offset + row_off) * 4,
                self._choice_ids,
                slot * 4,
                self.top_k * 4,
            )
            if count <= 4:
                kept = self._span_choice_buf()
                base = layer * 4 * self.top_k
                self.gpu.copy_bytes(
                    choice.buf,
                    choice.offset * 4,
                    kept,
                    base * 4,
                    count * self.top_k * 4,
                )
                choice = Dev(kept, count * self.top_k, base)
        self._choice[layer] = choice
        fp4_tiers = (
            isinstance(self.files, QuantRecord)
            and self.files.hot_bits == 4
            and self.files.cold_bits in (2, 4)
        )
        if (
            count == 1
            and self._decoding
            and self._atlas_trusted
            and self._tier is not None
            and fp4_tiers
        ):
            shared = self._shared_expert(bundle, xs, count)
            return self._moe_decode_trusted(layer, xs, weight, shared)
        # The host reads the route. A miss would zero that layer and change
        # every later route, so the token would have to be replayed.
        host_ids = self.gpu.read_u32(choice)
        rows = np.asarray(host_ids, dtype=np.uint32).reshape(count, self.top_k)
        # A decode span is at most four tokens. A prefill chunk can route
        # thousands of ids; the next token only follows the last row.
        chosen = rows.reshape(-1) if self._decoding and count <= 4 else rows[-1]
        self._remember_ids(layer, chosen.tolist())
        if self._streaming_prefill:
            return self._moe_from_stage(
                layer, bundle, xs, count, weight, host_ids, submit_next=submit_next
            )
        if self.atlas is None:
            message = "decode has no expert atlas"
            raise RuntimeError(message)
        protect = self._prefetch_protect if self._decoding else None
        event = self.atlas.stage(layer, host_ids.tolist(), protect)
        queued = self._prefetch_events.pop(layer, 0)
        if queued > event:
            event = queued
        shared = self._shared_expert(bundle, xs, count)
        self.gpu.io_fence(event)
        if self._decoding and 1 < count <= 4 and self._span_experts():
            return self._moe_span(layer, xs, weight, shared, host_ids, count)
        if count == 1:
            return self._moe_decode(layer, xs, weight, shared, host_ids)
        return self._moe_grouped(layer, xs, weight, shared, host_ids, count)

    def _moe_from_stage(
        self,
        layer: int,
        bundle: dict[str, Any],
        xs: Dev,
        count: int,
        weight: Dev,
        host_ids: np.ndarray,
        *,
        submit_next: bool,
    ) -> Dev:
        shared = self._shared_expert(bundle, xs, count)
        self.gpu.io_fence(self._stage_event[layer % 2])
        self._write_stage_table(layer)
        mixed = self._moe_grouped(layer, xs, weight, shared, host_ids, count)
        if submit_next and layer + 1 < self.layers:
            self._submit_prefill_layer(layer + 1)
        return mixed

    def _ensure_hold(self, count: int) -> None:
        if self._hold_cap >= count and self._hold_xs is not None:
            return
        xs_bytes = count * self.hidden * 4
        id_bytes = count * self.top_k * 4
        self._hold_xs = self.gpu.alloc(xs_bytes)
        self._hold_out = self.gpu.alloc(xs_bytes)
        self._hold_ids = self.gpu.alloc(id_bytes)
        self._hold_w = self.gpu.alloc(id_bytes)
        held = (self._hold_xs, self._hold_out, self._hold_ids, self._hold_w)
        if any(buf is None for buf in held):
            message = f"prefill expert hold of {count} tokens was not allocated"
            raise RuntimeError(message)
        self._hold_cap = count

    def _room_for_expert(self) -> None:
        arena = self.gpu._arena
        if arena is None or self.gpu._arena_used <= int(arena.length()) // 2:
            return
        self.gpu.flush()
        self.gpu.rewind_scratch()

    def _keep_mixed(self, mixed: Dev, count: int) -> Dev:
        n = count * self.hidden
        if mixed.buf is not self._hold_out or mixed.offset != 0 or mixed.n != n:
            self.gpu.copy_bytes(mixed.buf, mixed.offset * 4, self._hold_out, 0, n * 4)
        self.gpu.flush()
        self.gpu.rewind_scratch()
        return Dev(self._hold_out, n)

    def _moe_wide(self, layer: int, bundle: dict[str, Any], residual: Dev, count: int) -> None:
        """Int4 experts for the whole layer, after every attention chunk."""
        self._route_hold(layer, bundle, count)
        if not isinstance(self.files, QuantRecord):
            message = "prefill has no expert record"
            raise RuntimeError(message)
        if self._stage is None or self._stage_table is None or self._stage_miss is None:
            message = "prefill stream has no staging buffer"
            raise RuntimeError(message)
        raw = _buffer_bytes(self._hold_ids, count * self.top_k * 4)
        host_ids = np.frombuffer(raw, dtype=np.uint32).copy()
        rows = host_ids.reshape(count, self.top_k)
        self._predicted[layer] = [int(item) for item in rows[-1].tolist()]
        self._choice[layer] = Dev(self._hold_ids, self.top_k, (count - 1) * self.top_k)
        self.gpu.io_fence(self._stage_event[layer % 2])
        self._write_stage_table(layer)
        xs = Dev(self._hold_xs, count * self.hidden)
        weight = Dev(self._hold_w, count * self.top_k)
        running = Dev(self._hold_out, count * self.hidden)
        running = self._prefill_tier(
            layer, xs, weight, running, host_ids, count, self.files.hot_set(layer), self._hot
        )
        running = self._prefill_tier(
            layer, xs, weight, running, host_ids, count, self.files.cold_set(layer), self._cold
        )
        running = self._prefill_super(layer, xs, weight, running, host_ids, count)
        running = self._keep_mixed(running, count)
        span = self.branches * self.hidden
        for offset in range(0, count, self.prefill_chunk):
            n = min(self.prefill_chunk, count - offset)
            self.gpu.rewind_scratch()
            view = residual.view(offset * span, n * span)
            hats = self._rms_branches(view, bundle["mlp_norm"], n)
            self._inject_dev(
                view,
                hats,
                running.view(offset * self.hidden, n * self.hidden),
                bundle["mlp_inject"],
                n,
            )

    def _route_hold(self, layer: int, bundle: dict[str, Any], count: int) -> None:
        router = self._router(layer)
        hidden = self.hidden
        for offset in range(0, count, self.prefill_chunk):
            n = min(self.prefill_chunk, count - offset)
            self.gpu.rewind_scratch()
            xs = Dev(self._hold_xs, count * hidden).view(offset * hidden, n * hidden)
            shared = self._shared_expert(bundle, xs, n)
            self.gpu.copy_bytes(
                shared.buf,
                shared.offset * 4,
                self._hold_out,
                offset * hidden * 4,
                n * hidden * 4,
            )
            choice, weight = self.gpu.topk_batch(
                self.gpu.linear(router, xs, n, hidden), n, self.experts, self.top_k
            )
            self.gpu.copy_bytes(
                choice.buf,
                choice.offset * 4,
                self._hold_ids,
                offset * self.top_k * 4,
                n * self.top_k * 4,
            )
            self.gpu.copy_bytes(
                weight.buf,
                weight.offset * 4,
                self._hold_w,
                offset * self.top_k * 4,
                n * self.top_k * 4,
            )
            self.gpu.barrier()
        self.gpu.flush()

    def _prefill_tier(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        running: Dev,
        host_ids: np.ndarray,
        count: int,
        members: set[int],
        planes: ExpertPlanes,
    ) -> Dev:
        if self._stage is None or self._stage_table is None or self._stage_miss is None:
            message = "prefill stream has no staging buffer"
            raise RuntimeError(message)
        if planes.bits in (4, 8):
            return self._moe_int4_matmul(
                layer, xs, weight, running, host_ids, count, members, planes
            )
        return self._moe_tier_batched(
            layer,
            xs,
            weight,
            running,
            host_ids,
            count,
            members,
            planes,
            self._stage[layer % 2],
            self._stage_table,
            0,
            self._stage_miss,
        )

    def _moe_int4_matmul(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        running: Dev,
        host_ids: np.ndarray,
        count: int,
        members: set[int],
        planes: ExpertPlanes,
    ) -> Dev:
        if planes.bits not in (4, 8):
            message = "the wide matmul path was given another plane width"
            raise RuntimeError(message)
        if not isinstance(self.files, QuantRecord) or self._stage is None:
            message = "prefill stream has no staging buffer"
            raise RuntimeError(message)
        flat = np.asarray(host_ids, dtype=np.uint32).reshape(count, self.top_k)
        chosen: dict[int, list[tuple[int, int]]] = {}
        for token in range(count):
            row = flat[token]
            for slot in range(self.top_k):
                expert = int(row[slot])
                if expert in members:
                    chosen.setdefault(expert, []).append((token, slot))
        large: list[tuple[int, list[tuple[int, int]]]] = []
        small: set[int] = set()
        wide_hot = self._hot_keep is not None and planes.bits == 8
        threshold = 1 if wide_hot else 64
        for expert, pairs in chosen.items():
            if len(pairs) >= threshold:
                large.append((expert, pairs))
            else:
                small.add(expert)
        if large:
            if self._hot_fence:
                self.gpu.io_fence(self._hot_fence)
                self._hot_fence = 0
            if wide_hot:
                order = {expert: row for row, expert in enumerate(self.files.hot[layer])}
                image = None
                stage = self._hot_keep[layer]
            else:
                order = {}
                _hot_bytes, _cold_bytes, image = self._prefill_layout(layer)
                stage = self._stage[layer % 2]
            for expert, pairs in large:
                self._room_for_expert()
                if wide_hot:
                    base = order[expert] * self.files.slot
                else:
                    base = int(image[expert])
                if base < 0:
                    message = f"layer {layer} expert {expert} is not in the staging buffer"
                    raise RuntimeError(message)
                gate, down = self._plane_views(stage, planes)
                gate.weight_off += base
                gate.scale_off += base
                down.weight_off += base
                down.scale_off += base
                # One expert can be selected by every token of a long prompt.
                # The scratch arena is sized for one prefill chunk, so the
                # matmul is split there instead of growing the arena.
                for start in range(0, len(pairs), self.prefill_chunk):
                    if start:
                        self.gpu.flush()
                        self.gpu.rewind_scratch()
                    part = pairs[start : start + self.prefill_chunk]
                    tokens = np.asarray([item[0] for item in part], dtype=np.uint32)
                    slots = np.asarray([item[1] for item in part], dtype=np.uint32)
                    n = int(tokens.size)
                    token_buf = self._upload_u32(tokens)
                    slot_buf = self._upload_u32(slots)
                    gathered = self.gpu.gather_half(
                        xs, token_buf, n, self.hidden, gate.stored_inner
                    )
                    activated = self.gpu.moe_activate(
                        self.gpu.gemm_int4_half(gate, gathered, n, fp8=planes.bits == 8),
                        n,
                        self.moe,
                        down.stored_inner,
                    )
                    mixed = self.gpu.gemm_int4_half(
                        down, activated, n, fp8=planes.bits == 8
                    )
                    self.gpu.moe_add_weighted(
                        mixed, token_buf, slot_buf, weight, running, n, self.hidden, self.top_k
                    )
        if small:
            self.gpu.flush()
            self.gpu.rewind_scratch()
            table = self._stage_table
            miss = self._stage_miss
            if table is None or miss is None:
                message = "prefill stream has no offset table"
                raise RuntimeError(message)
            return self._moe_tier_batched(
                layer,
                xs,
                weight,
                running,
                host_ids,
                count,
                small,
                planes,
                stage if large else self._stage[layer % 2],
                table,
                0,
                miss,
            )
        if large:
            self.gpu.flush()
            self.gpu.rewind_scratch()
        return running

    def _moe_tier_batched(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
        members: set[int],
        planes: ExpertPlanes,
        weights: Any,
        table: Any,
        table_base: int,
        miss: Any,
    ) -> Dev:
        if not members:
            return shared
        flat = np.asarray(host_ids, dtype=np.uint32).reshape(count, self.top_k)
        sizes: dict[int, int] = {}
        for token in range(count):
            for slot in range(self.top_k):
                expert = int(flat[token, slot])
                if expert in members:
                    sizes[expert] = sizes.get(expert, 0) + 1
        if not sizes:
            return shared
        running = shared
        batch: set[int] = set()
        assigned = 0
        # int8 walks one token at a time. Keep one command buffer inside the GPU timeout.
        cap = 2048 if planes.bits == 8 else 65536
        for expert, size in sizes.items():
            if batch and assigned + size > cap:
                running = self._moe_tier(
                    layer,
                    xs,
                    weight,
                    running,
                    host_ids,
                    count,
                    batch,
                    planes,
                    weights,
                    table,
                    table_base,
                    miss,
                )
                running = self._keep_mixed(running, count)
                batch = set()
                assigned = 0
            batch.add(expert)
            assigned += size
        if batch:
            running = self._moe_tier(
                layer,
                xs,
                weight,
                running,
                host_ids,
                count,
                batch,
                planes,
                weights,
                table,
                table_base,
                miss,
            )
            running = self._keep_mixed(running, count)
        return running

    def _shared_expert(self, bundle: dict[str, Any], xs: Dev, count: int) -> Dev:
        hidden = self.hidden
        gate = self.gpu.linear(bundle["shared_gate"], xs, count, hidden, epilogue=1)
        up = self.gpu.linear(bundle["shared_up"], xs, count, hidden)
        shared = self.gpu.linear(bundle["shared_down"], self.gpu.mul(gate, up), count, self.moe)
        coeff = self.gpu.linear(bundle["shared_coeff"], xs, count, hidden, epilogue=2)
        return self.gpu.scale_rows(shared, coeff, count, hidden)

    def _moe_bf16(self, layer: int, bundle: dict[str, Any], xs: Dev, count: int) -> Dev:
        hidden = self.hidden
        shared = self._shared_expert(bundle, xs, count)
        router = self._router(layer)
        out = self.gpu.tmp(count * hidden)
        for index in range(count):
            row = xs.view(index * hidden, hidden)
            logits = self.gpu.linear(router, row, 1, hidden)
            choice, weight = self.gpu.topk(logits, logits.n, self.top_k)
            picked = [int(expert) for expert in self.gpu.read_u32(choice).tolist()]
            self._load_experts(layer, picked)
            mixed = self.gpu.copy_dev(shared.view(index * hidden, hidden))
            for slot, expert in enumerate(picked):
                routed = self._routed_dev(layer, expert, row)
                self.gpu.axpy(mixed, routed, weight, slot)
            self.gpu.copy_into(mixed, out, index * hidden)
        return out

    def _moe_decode(
        self, layer: int, xs: Dev, weight: Dev, shared: Dev, host_ids: np.ndarray
    ) -> Dev:
        if self.atlas is None or not isinstance(self.files, QuantRecord):
            message = "decode has no expert atlas"
            raise RuntimeError(message)
        row = xs if xs.n == self.hidden else xs.view(0, self.hidden)
        shared_row = shared if shared.n == self.hidden else shared.view(0, self.hidden)
        ids = [int(expert) for expert in np.asarray(host_ids).reshape(-1).tolist()]
        running = shared_row
        for members, planes in (
            (self.files.hot_set(layer), self._hot),
            (self.files.cold_set(layer), self._cold),
        ):
            slots = [index for index, expert in enumerate(ids) if expert in members]
            if not slots:
                continue
            running = self._decode_tier(
                layer,
                row,
                running,
                [ids[index] for index in slots],
                weight,
                slots,
                planes,
            )
        return self._decode_super(layer, row, running, ids, weight)

    def _moe_decode_trusted(self, layer: int, xs: Dev, weight: Dev, shared: Dev) -> Dev:
        """Expert GEMVs from the device ids. The host does not read them between layers."""
        if self.atlas is None or self._tier is None or not isinstance(self.files, QuantRecord):
            message = "decode has no tier table"
            raise RuntimeError(message)
        choice = Dev(self._choice_ids, self.top_k, layer * self.top_k)
        row = xs if xs.n == self.hidden else xs.view(0, self.hidden)
        shared_row = shared if shared.n == self.hidden else shared.view(0, self.hidden)
        running = self._decode_tier_ids(layer, row, shared_row, choice, weight, self._hot, 0)
        running = self._decode_tier_ids(layer, row, running, choice, weight, self._cold, 1)
        return self._decode_super_trusted(layer, row, running, choice, weight)

    def _decode_tier_ids(
        self,
        layer: int,
        row: Dev,
        shared_row: Dev,
        choice: Dev,
        weight: Dev,
        planes: ExpertPlanes,
        want: int,
    ) -> Dev:
        if self.atlas is None or self._tier is None:
            message = "decode has no tier table"
            raise RuntimeError(message)
        gate_fn = self.gpu.moe_decode_gate if planes.bits == 4 else self.gpu.moe_decode_gate_int2
        down_fn = self.gpu.moe_decode_down if planes.bits == 4 else self.gpu.moe_decode_down_int2
        gate = gate_fn(
            row,
            self.atlas.atlas,
            self.atlas.table,
            choice,
            self.top_k,
            layer * self.experts,
            planes.gate_rows,
            planes.gate_inner,
            0,
            planes.gate_scale,
            self.atlas.miss,
            layer,
            want=want,
            tier=self._tier,
        )
        activated = self.gpu.moe_activate(gate, self.top_k, self.moe, planes.down_inner)
        return down_fn(
            activated,
            self.atlas.atlas,
            self.atlas.table,
            choice,
            weight,
            shared_row,
            self.top_k,
            layer * self.experts,
            self.hidden,
            planes.down_inner,
            planes.down_weight,
            planes.down_scale,
            self.atlas.miss,
            layer,
            want=want,
            tier=self._tier,
        )

    def _decode_super_trusted(
        self, layer: int, row: Dev, running: Dev, choice: Dev, weight: Dev
    ) -> Dev:
        if not isinstance(self.files, QuantRecord):
            return running
        pins = self.files.super[layer]
        if not pins:
            return running
        running = self.gpu.copy_dev(running)
        planes = self._super_planes
        for expert in pins:
            image = self._super_images[(layer, expert)]
            gate = self._super_gemv(
                row,
                image,
                planes.gate_rows,
                planes.gate_inner,
                0,
                planes.gate_scale,
                self.hidden,
                half=False,
            )
            activated = self.gpu.moe_activate(gate, 1, self.moe, planes.down_inner)
            down = self._super_gemv(
                activated,
                image,
                self.hidden,
                planes.down_inner,
                planes.down_weight,
                planes.down_scale,
                self.moe,
                half=True,
            )
            self.gpu.axpy_matching(running, down, weight, choice, expert)
        return running

    def _decode_int8_atlas(
        self,
        layer: int,
        row: Dev,
        shared_row: Dev,
        ids: list[int],
        weight: Dev,
        slots: list[int],
        planes: ExpertPlanes,
    ) -> Dev:
        if self.atlas is None:
            message = "decode has no expert atlas"
            raise RuntimeError(message)
        running = self.gpu.copy_dev(shared_row)
        atlas = self.atlas.atlas
        for expert, slot in zip(ids, slots, strict=True):
            base = self.atlas.resident.get((layer, expert))
            if base is None:
                message = f"layer {layer} expert {expert} is not in the atlas"
                raise RuntimeError(message)
            gate = self.gpu.gemv_int8(
                row,
                atlas,
                planes.gate_rows,
                planes.gate_inner,
                base,
                base + planes.gate_scale,
                self.hidden,
                half=False,
            )
            activated = self.gpu.moe_activate(gate, 1, self.moe, planes.down_inner)
            down = self.gpu.gemv_int8(
                activated,
                atlas,
                self.hidden,
                planes.down_inner,
                base + planes.down_weight,
                base + planes.down_scale,
                self.moe,
                half=True,
            )
            self.gpu.axpy(running, down, weight, slot)
        return running

    def _decode_tier(
        self,
        layer: int,
        row: Dev,
        shared_row: Dev,
        ids: list[int],
        weight: Dev,
        slots: list[int],
        planes: ExpertPlanes,
    ) -> Dev:
        if self.atlas is None:
            message = "decode has no expert atlas"
            raise RuntimeError(message)
        if planes.bits == 8:
            return self._decode_int8_atlas(layer, row, shared_row, ids, weight, slots, planes)
        choice = self.gpu.dev_u32(ids)
        compact = self.gpu.tmp(len(slots))
        self.gpu.gather_f32(weight, self.gpu.dev_u32(slots), compact, len(slots))
        weight = compact
        gate_fn = self.gpu.moe_decode_gate if planes.bits == 4 else self.gpu.moe_decode_gate_int2
        down_fn = self.gpu.moe_decode_down if planes.bits == 4 else self.gpu.moe_decode_down_int2
        gate = gate_fn(
            row,
            self.atlas.atlas,
            self.atlas.table,
            choice,
            len(ids),
            layer * self.experts,
            planes.gate_rows,
            planes.gate_inner,
            0,
            planes.gate_scale,
            self.atlas.miss,
            layer,
        )
        activated = self.gpu.moe_activate(gate, len(ids), self.moe, planes.down_inner)
        return down_fn(
            activated,
            self.atlas.atlas,
            self.atlas.table,
            choice,
            weight,
            shared_row,
            len(ids),
            layer * self.experts,
            self.hidden,
            planes.down_inner,
            planes.down_weight,
            planes.down_scale,
            self.atlas.miss,
            layer,
        )

    def _moe_span_device(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        choice: Dev,
        count: int,
    ) -> Dev:
        """Grouped experts from device ids. The host does not read the routes."""
        if self.atlas is None:
            message = "span decode has no expert atlas"
            raise RuntimeError(message)
        running = self._span_device_tier(
            layer, xs, weight, shared, choice, count, self._hot, self.atlas.atlas, 0
        )
        running = self._span_device_tier(
            layer, xs, weight, running, choice, count, self._cold, self.atlas.atlas, 1
        )
        return self._span_super_pins(layer, xs, weight, running, choice, count)

    def _span_device_tier(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        choice: Dev,
        count: int,
        planes: ExpertPlanes,
        weights: Any,
        want: int,
    ) -> Dev:
        if self.atlas is None or planes.bits not in (4, 8):
            return shared
        slots = count * self.top_k
        experts = self.gpu.span(slots * 4)
        offsets = self.gpu.span((slots + 1) * 4)
        tokens = self.gpu.span(slots * 4)
        inverse = self.gpu.span(slots * 4)
        for buf, nbytes in (
            (experts, slots * 4),
            (offsets, (slots + 1) * 4),
            (tokens, slots * 4),
            (inverse, slots * 4),
        ):
            view = _buffer_view(buf.buf, buf.offset + nbytes)
            view[buf.offset : buf.offset + nbytes] = bytes(nbytes)
        self.gpu.group_span(
            choice,
            self._tier,
            experts,
            offsets,
            tokens,
            inverse,
            count,
            self.top_k,
            layer * self.experts,
            want,
            slots,
        )
        self.gpu.barrier()
        gate_kernel = "moe_span_fp4_gate" if planes.bits == 4 else "moe_span_fp8_gate"
        down_kernel = "moe_span_fp4_down" if planes.bits == 4 else "moe_span_fp8_down"
        logical = None if planes.bits == 4 else self.hidden
        gathered = self.gpu.gather_rows_f32(xs, tokens, slots, self.hidden, planes.gate_inner)
        gate = self.gpu.span(max(slots, 1) * planes.gate_rows * 4)
        self.gpu.moe_grouped(
            gathered,
            weights,
            self.atlas.table,
            experts,
            offsets,
            gate,
            self.atlas.miss,
            slots,
            layer * self.experts,
            planes.gate_rows,
            planes.gate_inner,
            0,
            planes.gate_scale,
            layer,
            kernel=gate_kernel,
            logical=logical,
        )
        activated = self.gpu.moe_activate(gate, slots, self.moe, planes.down_inner)
        down = self.gpu.span(max(slots, 1) * self.hidden * 4)
        down_logical = None if planes.bits == 4 else self.moe
        self.gpu.moe_grouped(
            activated,
            weights,
            self.atlas.table,
            experts,
            offsets,
            down,
            self.atlas.miss,
            slots,
            layer * self.experts,
            self.hidden,
            planes.down_inner,
            planes.down_weight,
            planes.down_scale,
            layer,
            kernel=down_kernel,
            logical=down_logical,
        )
        return self.gpu.moe_scatter(shared, down, inverse, weight, count, self.top_k, self.hidden)

    def _span_super_pins(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        running: Dev,
        choice: Dev,
        count: int,
    ) -> Dev:
        if not isinstance(self.files, QuantRecord):
            return running
        pins = self.files.super[layer]
        if not pins:
            return running
        planes = self._super_planes
        for token in range(count):
            row = xs.view(token * self.hidden, self.hidden)
            acc = running.view(token * self.hidden, self.hidden)
            ids = choice.view(token * self.top_k, self.top_k)
            scales = weight.view(token * self.top_k, self.top_k)
            for expert in pins:
                image = self._super_images[(layer, int(expert))]
                gate = self._super_gemv(
                    row,
                    image,
                    planes.gate_rows,
                    planes.gate_inner,
                    0,
                    planes.gate_scale,
                    self.hidden,
                    half=False,
                )
                activated = self.gpu.moe_activate(gate, 1, self.moe, planes.down_inner)
                down = self._super_gemv(
                    activated,
                    image,
                    self.hidden,
                    planes.down_inner,
                    planes.down_weight,
                    planes.down_scale,
                    self.moe,
                    half=True,
                )
                self.gpu.axpy_matching(acc, down, scales, ids, int(expert))
        return running

    def _moe_span(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
    ) -> Dev:
        if self.atlas is None or not isinstance(self.files, QuantRecord):
            message = "span decode has no expert atlas"
            raise RuntimeError(message)
        running = self._span_tier(
            layer,
            xs,
            weight,
            shared,
            host_ids,
            count,
            self.files.hot_set(layer),
            self._hot,
            self.atlas.atlas,
            self.atlas.table,
            layer * self.experts,
            self.atlas.miss,
        )
        running = self._span_tier(
            layer,
            xs,
            weight,
            running,
            host_ids,
            count,
            self.files.cold_set(layer),
            self._cold,
            self.atlas.atlas,
            self.atlas.table,
            layer * self.experts,
            self.atlas.miss,
        )
        return self._span_super(layer, xs, weight, running, host_ids, count)

    def _span_tier(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
        members: set[int],
        planes: ExpertPlanes,
        weights: Any,
        table: Any,
        table_base: int,
        miss: Any,
    ) -> Dev:
        experts, offsets, tokens, inverse = group_tier(host_ids, count, self.top_k, members)
        if int(experts.size) == 0:
            return shared
        if planes.bits not in (4, 8):
            message = f"span decode does not support width {planes.bits}"
            raise RuntimeError(message)
        return self._span_groups(
            layer,
            xs,
            weight,
            shared,
            tokens,
            experts,
            offsets,
            inverse,
            count,
            planes,
            weights,
            table,
            table_base,
            miss,
        )

    def _span_groups(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        tokens: np.ndarray,
        experts: np.ndarray,
        offsets: np.ndarray,
        inverse: np.ndarray,
        count: int,
        planes: ExpertPlanes,
        weights: Any,
        table: Any,
        table_base: int,
        miss: Any,
    ) -> Dev:
        n_assign = int(tokens.size)
        token_buf = self._upload_u32(tokens)
        expert_buf = self._upload_u32(experts)
        offset_buf = self._upload_u32(offsets)
        inverse_buf = self._upload_u32(inverse)
        gate_kernel = "moe_span_fp4_gate" if planes.bits == 4 else "moe_span_fp8_gate"
        down_kernel = "moe_span_fp4_down" if planes.bits == 4 else "moe_span_fp8_down"
        logical = None if planes.bits == 4 else self.hidden
        gathered = self.gpu.gather_rows_f32(
            xs, token_buf, n_assign, self.hidden, planes.gate_inner
        )
        gate = self.gpu.span(n_assign * planes.gate_rows * 4)
        self.gpu.moe_grouped(
            gathered,
            weights,
            table,
            expert_buf,
            offset_buf,
            gate,
            miss,
            int(experts.size),
            table_base,
            planes.gate_rows,
            planes.gate_inner,
            0,
            planes.gate_scale,
            layer,
            kernel=gate_kernel,
            logical=logical,
        )
        activated = self.gpu.moe_activate(gate, n_assign, self.moe, planes.down_inner)
        down = self.gpu.span(n_assign * self.hidden * 4)
        down_logical = None if planes.bits == 4 else self.moe
        self.gpu.moe_grouped(
            activated,
            weights,
            table,
            expert_buf,
            offset_buf,
            down,
            miss,
            int(experts.size),
            table_base,
            self.hidden,
            planes.down_inner,
            planes.down_weight,
            planes.down_scale,
            layer,
            kernel=down_kernel,
            logical=down_logical,
        )
        return self.gpu.moe_scatter(
            shared, down, inverse_buf, weight, count, self.top_k, self.hidden
        )

    def _span_super(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
    ) -> Dev:
        if not isinstance(self.files, QuantRecord) or not self.files.super_set(layer):
            return shared
        experts, offsets, tokens, inverse = group_tier(
            host_ids, count, self.top_k, self.files.super_set(layer)
        )
        if int(experts.size) == 0:
            return shared
        planes = self._super_planes
        if planes.bits not in (4, 8):
            message = f"span decode does not support width {planes.bits}"
            raise RuntimeError(message)
        if self.atlas is None:
            message = "span decode has no expert atlas"
            raise RuntimeError(message)
        if getattr(self, "_span_base", None) is None:
            self._span_base = self.gpu.zeros(8)
        n_assign = int(tokens.size)
        partial = self.gpu.tmp(n_assign * self.hidden)
        gate_kernel = "moe_span_fp4_gate" if planes.bits == 4 else "moe_span_fp8_gate"
        down_kernel = "moe_span_fp4_down" if planes.bits == 4 else "moe_span_fp8_down"
        logical = None if planes.bits == 4 else self.hidden
        down_logical = None if planes.bits == 4 else self.moe
        one = self._upload_u32(np.asarray([0], dtype=np.uint32))
        for index, expert in enumerate(experts.tolist()):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            picked = np.asarray(tokens[start:stop], dtype=np.uint32)
            n = int(picked.size)
            token_buf = self._upload_u32(picked)
            bounds = self._upload_u32(np.asarray([0, n], dtype=np.uint32))
            image = self._super_images[(layer, int(expert))]
            gathered = self.gpu.gather_rows_f32(
                xs, token_buf, n, self.hidden, planes.gate_inner
            )
            gate = self.gpu.span(n * planes.gate_rows * 4)
            self.gpu.moe_grouped(
                gathered,
                image,
                self._span_base,
                one,
                bounds,
                gate,
                self.atlas.miss,
                1,
                0,
                planes.gate_rows,
                planes.gate_inner,
                0,
                planes.gate_scale,
                layer,
                kernel=gate_kernel,
                logical=logical,
            )
            activated = self.gpu.moe_activate(gate, n, self.moe, planes.down_inner)
            down = self.gpu.span(n * self.hidden * 4)
            self.gpu.moe_grouped(
                activated,
                image,
                self._span_base,
                one,
                bounds,
                down,
                self.atlas.miss,
                1,
                0,
                self.hidden,
                planes.down_inner,
                planes.down_weight,
                planes.down_scale,
                layer,
                kernel=down_kernel,
                logical=down_logical,
            )
            placed = Dev(down.buf, n * self.hidden, down.offset // 4)
            self.gpu.copy_into(placed, partial, start * self.hidden)
        return self.gpu.moe_scatter(
            shared, partial, self._upload_u32(inverse), weight, count, self.top_k, self.hidden
        )

    def _moe_grouped(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
    ) -> Dev:
        if not isinstance(self.files, QuantRecord):
            message = "prefill has no expert record"
            raise RuntimeError(message)
        if self._streaming_prefill:
            if self._stage is None or self._stage_table is None or self._stage_miss is None:
                message = "prefill stream has no staging buffer"
                raise RuntimeError(message)
            weights = self._stage[layer % 2]
            table = self._stage_table
            table_base = 0
            miss = self._stage_miss
        elif self.atlas is None:
            message = "prefill has no expert atlas"
            raise RuntimeError(message)
        else:
            weights = self.atlas.atlas
            table = self.atlas.table
            table_base = layer * self.experts
            miss = self.atlas.miss
        mixed = self._moe_tier(
            layer,
            xs,
            weight,
            shared,
            host_ids,
            count,
            self.files.hot_set(layer),
            self._hot,
            weights,
            table,
            table_base,
            miss,
        )
        mixed = self._moe_tier(
            layer,
            xs,
            weight,
            mixed,
            host_ids,
            count,
            self.files.cold_set(layer),
            self._cold,
            weights,
            table,
            table_base,
            miss,
        )
        return self._prefill_super(layer, xs, weight, mixed, host_ids, count)

    def _moe_tier(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
        members: set[int],
        planes: ExpertPlanes,
        weights: Any,
        table: Any,
        table_base: int,
        miss: Any,
    ) -> Dev:
        experts, offsets, tokens, inverse = group_tier(host_ids, count, self.top_k, members)
        if int(experts.size) == 0:
            return shared
        n_assign = int(tokens.size)
        n_groups = int(experts.size)
        token_buf = self._upload_u32(tokens)
        expert_buf = self._upload_u32(experts)
        offset_buf = self._upload_u32(offsets)
        inverse_buf = self._upload_u32(inverse)
        gathered = self.gpu.gather_half(xs, token_buf, n_assign, self.hidden, self.hidden)
        gate = self.gpu.span(n_assign * planes.gate_rows * 4)
        if planes.bits == 8:
            gemm = self.gpu.moe_grouped_int8
        elif planes.bits == 2:
            gemm = self.gpu.moe_grouped_int2
        else:
            gemm = self.gpu.moe_grouped
        gemm(
            gathered,
            weights,
            table,
            expert_buf,
            offset_buf,
            gate,
            miss,
            n_groups,
            table_base,
            planes.gate_rows,
            planes.gate_inner,
            0,
            planes.gate_scale,
            layer,
        )
        activated = self.gpu.moe_activate(gate, n_assign, self.moe, planes.down_inner)
        down = self.gpu.span(n_assign * self.hidden * 4)
        gemm(
            activated,
            weights,
            table,
            expert_buf,
            offset_buf,
            down,
            miss,
            n_groups,
            table_base,
            self.hidden,
            planes.down_inner,
            planes.down_weight,
            planes.down_scale,
            layer,
        )
        return self.gpu.moe_scatter(
            shared, down, inverse_buf, weight, count, self.top_k, self.hidden
        )

    def _upload_tiers(self) -> None:
        """One byte per expert: 0 hot, 1 cold, 2 super."""
        if not isinstance(self.files, QuantRecord):
            return
        raw = bytearray(self.layers * self.experts)
        for layer in range(self.layers):
            base = layer * self.experts
            for expert in self.files.hot_set(layer):
                raw[base + expert] = 0
            for expert in self.files.cold_set(layer):
                raw[base + expert] = 1
            for expert in self.files.super_set(layer):
                raw[base + expert] = 2
        self._tier = self.gpu.upload(bytes(raw))

    def _remember_ids(self, layer: int, values: Sequence[int]) -> None:
        seen: list[int] = []
        for expert in values:
            value = int(expert)
            if value not in seen:
                seen.append(value)
        self._predicted[layer] = seen

    def _note_routes(self) -> None:
        """The router ids of this span, read after the command buffer has finished."""
        rows = int(getattr(self, "_span_rows", 1))
        span_buf = getattr(self, "_span_choice", None)
        if rows > 1 and span_buf is not None:
            width = 4 * self.top_k
            raw = _buffer_bytes(span_buf, self.layers * width * 4)
            ids = np.frombuffer(raw, dtype=np.uint32).reshape(self.layers, 4, self.top_k)
            for layer in range(self.layers):
                self._remember_ids(layer, ids[layer, :rows].reshape(-1).tolist())
            return
        raw = _buffer_bytes(self._choice_ids, self.layers * self.top_k * 4)
        ids = np.frombuffer(raw, dtype=np.uint32).reshape(self.layers, self.top_k)
        for layer in range(self.layers):
            self._predicted[layer] = [int(expert) for expert in ids[layer]]

    def _super_gemv(
        self,
        activation: Any,
        image: Any,
        cols: int,
        stored: int,
        weight_off: int,
        scale_off: int,
        logical: int,
        *,
        half: bool,
    ) -> Dev:
        planes = self._super_planes
        if planes.bits == 8:
            return self.gpu.gemv_int8(
                activation,
                image,
                cols,
                stored,
                weight_off,
                scale_off,
                logical,
                half=half,
            )
        if planes.bits == 4:
            return self.gpu.gemv_fp4(
                activation,
                image,
                cols,
                stored,
                weight_off,
                scale_off,
                logical,
                half=half,
            )
        if planes.bits != 2:
            message = f"super width {planes.bits} is not 2, 4, or 8"
            raise RuntimeError(message)
        return self._super_int2(activation, image, cols, stored, weight_off, scale_off, half=half)

    def _super_int2(
        self,
        activation: Any,
        image: Any,
        cols: int,
        stored: int,
        weight_off: int,
        scale_off: int,
        *,
        half: bool,
    ) -> Dev:
        if self._int2_ids is None:
            self._int2_ids = self.gpu.dev_u32([0])
            self._int2_table = self.gpu.zeros(8)
            self._int2_miss = self.gpu.zeros(4)
            self._int2_one = self.gpu.dev_from(np.array([1.0], dtype=np.float32))
        if half:
            shared = self.gpu.zeros(cols * 4)
            return self.gpu.moe_decode_down_int2(
                activation,
                image,
                self._int2_table,
                self._int2_ids,
                self._int2_one,
                shared,
                1,
                0,
                cols,
                stored,
                weight_off,
                scale_off,
                self._int2_miss,
                0,
            )
        return self.gpu.moe_decode_gate_int2(
            activation,
            image,
            self._int2_table,
            self._int2_ids,
            1,
            0,
            cols,
            stored,
            weight_off,
            scale_off,
            self._int2_miss,
            0,
        )

    def _load_supers(self) -> None:
        if not isinstance(self.files, QuantRecord):
            return
        for layer, row in enumerate(self.files.super):
            for expert in row:
                image = self.files.expert_image(layer, expert)
                self._super_images[(layer, expert)] = self.gpu.upload_resident(image)

    def _decode_super(
        self, layer: int, row: Dev, running: Dev, ids: list[int], weight: Dev
    ) -> Dev:
        hits = [slot for slot, expert in enumerate(ids) if (layer, expert) in self._super_images]
        if not hits:
            return running
        running = self.gpu.copy_dev(running)
        planes = self._super_planes
        for slot in hits:
            image = self._super_images[(layer, ids[slot])]
            gate = self._super_gemv(
                row,
                image,
                planes.gate_rows,
                planes.gate_inner,
                0,
                planes.gate_scale,
                self.hidden,
                half=False,
            )
            activated = self.gpu.moe_activate(gate, 1, self.moe, planes.down_inner)
            down = self._super_gemv(
                activated,
                image,
                self.hidden,
                planes.down_inner,
                planes.down_weight,
                planes.down_scale,
                self.moe,
                half=True,
            )
            self.gpu.axpy(running, down, weight, slot)
        return running

    def _prefill_super(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        shared: Dev,
        host_ids: np.ndarray,
        count: int,
    ) -> Dev:
        if not isinstance(self.files, QuantRecord) or not self.files.super_set(layer):
            return shared
        experts, offsets, tokens, inverse = group_tier(
            host_ids, count, self.top_k, self.files.super_set(layer)
        )
        if int(experts.size) == 0:
            return shared
        planes = self._super_planes
        n_assign = int(tokens.size)
        partial = self.gpu.tmp(n_assign * self.hidden)
        for index, expert in enumerate(experts.tolist()):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            image = self._super_images[(layer, int(expert))]
            for assign in range(start, stop):
                token = int(tokens[assign])
                token_row = xs.view(token * self.hidden, self.hidden)
                gate = self._super_gemv(
                    token_row,
                    image,
                    planes.gate_rows,
                    planes.gate_inner,
                    0,
                    planes.gate_scale,
                    self.hidden,
                    half=False,
                )
                activated = self.gpu.moe_activate(gate, 1, self.moe, planes.down_inner)
                down = self._super_gemv(
                    activated,
                    image,
                    self.hidden,
                    planes.down_inner,
                    planes.down_weight,
                    planes.down_scale,
                    self.moe,
                    half=True,
                )
                self.gpu.copy_into(down, partial, assign * self.hidden)
        return self.gpu.moe_scatter(
            shared, partial, self._upload_u32(inverse), weight, count, self.top_k, self.hidden
        )

    def _upload_u32(self, values: np.ndarray) -> Any:
        raw = np.ascontiguousarray(values, dtype=np.uint32).tobytes()
        blob = self.gpu.span(len(raw))
        _buffer_view(blob.buf, blob.offset + len(raw))[blob.offset : blob.offset + len(raw)] = raw
        return blob

    def _router(self, layer: int) -> F16Matrix:
        return self.f16_weights[f"model.language_model.layers.{layer}.mlp.gate.weight"]

    def _load_experts(self, layer: int, experts: Sequence[int]) -> None:
        missing = [
            int(expert) for expert in experts if (layer, int(expert)) not in self.expert_cache
        ]
        if not missing:
            return
        if self._bf16:
            for expert in missing:
                self._ensure_bf16_expert(layer, expert)
            return
        if not isinstance(self.files, QuantRecord):
            message = "expert load has no record"
            raise RuntimeError(message)
        buffers: list[tuple[int, Any]] = []
        loads: list[tuple[Any, int, int, int]] = []
        for expert in sorted(set(missing)):
            tier, row = self.files.tier_row(layer, expert)
            if tier != 0:
                message = "the linear expert path reads int4 hot images"
                raise RuntimeError(message)
            buf = self.gpu.alloc(self.files.live)
            buffers.append((expert, buf))
            loads.append((buf, 0, self.files.live, row * self.files.slot))
        self.gpu.io_read(self.files.hot_io[layer], loads, gpu_wait=True)
        for expert, buf in buffers:
            self.expert_cache[(layer, expert)] = self._expert_views(buf)

    def _plane_views(self, buf: Any, planes: ExpertPlanes) -> tuple[Int4Matrix, Int4Matrix]:
        gate = Int4Matrix(
            planes.gate_rows,
            self.hidden,
            planes.gate_rows,
            buf,
            buf,
            planes.gate_inner,
            0,
            planes.gate_scale,
        )
        down = Int4Matrix(
            self.hidden,
            self.moe,
            self.hidden,
            buf,
            buf,
            planes.down_inner,
            planes.down_weight,
            planes.down_scale,
        )
        return gate, down

    def _expert_views(self, buf: Any) -> tuple[Int4Matrix, Int4Matrix]:
        stored_down = stored_k_int4(self.moe, align_k=True)
        data_n, scale_n = int4_plane_sizes(1280, self.hidden)
        split = data_n + scale_n
        down_data, _down_scale = int4_plane_sizes(self.hidden, stored_down)
        gate = Int4Matrix(1280, self.hidden, 1280, buf, buf, self.hidden, 0, data_n)
        down = Int4Matrix(
            self.hidden,
            self.moe,
            self.hidden,
            buf,
            buf,
            stored_down,
            split,
            split + down_data,
        )
        return gate, down

    def _ensure_bf16_expert(self, layer: int, expert: int) -> None:
        if self.store is None:
            message = "BF16 expert load has no shard store"
            raise RuntimeError(message)
        gate = self.gpu.load_f32(self.store.expert(layer, "gate_up_proj", expert))
        down = self.gpu.load_f32(self.store.expert(layer, "down_proj", expert))
        self.expert_cache[(layer, expert)] = (gate, down)

    def _routed_dev(self, layer: int, expert: int, vector: Dev) -> Dev:
        gate_up, down = self.expert_cache[(layer, expert)]
        hidden = self.gpu.linear(gate_up, vector, 1, self.hidden)
        activated = self.gpu.mul(
            self.gpu.silu(hidden.view(0, self.moe)), hidden.view(self.moe, self.moe)
        )
        return self.gpu.linear(down, activated, 1, self.moe)

    def _ple_dev(
        self,
        residual: Dev,
        tokens: list[int],
        count: int,
        prior: list[int] | None = None,
    ) -> None:
        """N-gram memory injected before this layer's residual read."""
        if count >= 256:
            self._ple_chunk(residual, tokens, count, prior)
            return
        ple = self._ple_bundle()
        span = self.branches * self.hidden
        conv_w, conv_off = ple["conv"]
        prefix = self.tokens if prior is None else self.tokens + prior
        for index in range(count):
            history = prefix + tokens[: index + 1]
            rows = [
                _ngram_row(
                    history,
                    head,
                    ple["mult"],
                    ple["offset"],
                    ple["vocab"],
                    heads_per_ngram=self.heads_per_ngram,
                    ngram_size=self.ngram_size,
                    eos=self.eos,
                )
                for head in range(self.ngram_heads)
            ]
            embedding = self._ngram_embedding(rows)
            key = self.gpu.linear(ple["key"], embedding, 1, self.hidden)
            value = self.gpu.linear(ple["value"], embedding, 1, self.hidden)
            slot = residual.view(index * span, span)
            query_w, query_off = ple["norm_query"]
            key_w, key_off = ple["norm_key"]
            conv_norm, conv_norm_off = ple["norm_conv"]
            query = self.gpu.rms(
                slot, query_w, query_off, self.hidden, self.branches, self.eps, True, self.branches
            )
            keyed = self.gpu.rms(
                key, key_w, key_off, self.hidden, self.branches, self.eps, True, self.branches
            )
            gate = self.gpu.ple_gate(query, keyed, self.hidden, self.branches)
            updated = self.gpu.gate_broadcast(gate, value, self.branches, self.hidden)
            normed = self.gpu.rms(
                updated,
                conv_norm,
                conv_norm_off,
                self.hidden,
                self.branches,
                self.eps,
                True,
                self.branches,
            )
            conv = self.gpu.dilated_conv(
                self.ple_hist, normed, conv_w, conv_off, span, 4, self.ngram_size
            )
            self.gpu.add_into(slot, self.gpu.add(updated, self.gpu.silu(conv)))
            self._mark_ple_token(index)

    def _ple_chunk(
        self,
        residual: Dev,
        tokens: list[int],
        count: int,
        prior: list[int] | None,
    ) -> None:
        """One row-id pass and one dequant for a long chunk. The conv stays ordered."""
        ple = self._ple_bundle()
        prefix = self.tokens if prior is None else self.tokens + prior
        rows = ngram_rows(
            prefix + tokens,
            ple["mult"],
            ple["offset"],
            ple["vocab"],
            heads_per_ngram=self.heads_per_ngram,
            ngram_size=self.ngram_size,
            eos=self.eos,
            start=len(prefix),
        )
        if len(rows) != count * self.ngram_heads:
            message = f"n-gram rows {len(rows)} are not {count} by {self.ngram_heads}"
            raise RuntimeError(message)
        embedding = self._ngram_embedding(rows)
        width = self.ngram_heads * NGRAM_WIDTH
        if width != self.hidden or embedding.n != count * width:
            message = f"n-gram width {width} embedding {embedding.n} does not match hidden"
            raise RuntimeError(message)
        key = self.gpu.linear(ple["key"], embedding, count, self.hidden)
        value = self.gpu.linear(ple["value"], embedding, count, self.hidden)
        if key.n != count * self.branches * self.hidden or value.n != count * self.hidden:
            message = "n-gram projection width does not match the residual"
            raise RuntimeError(message)
        query_w, query_off = ple["norm_query"]
        key_w, key_off = ple["norm_key"]
        query = self.gpu.rms(
            residual,
            query_w,
            query_off,
            self.hidden,
            count * self.branches,
            self.eps,
            True,
            self.branches,
        )
        keyed = self.gpu.rms(
            key,
            key_w,
            key_off,
            self.hidden,
            count * self.branches,
            self.eps,
            True,
            self.branches,
        )
        gate = self.gpu.ple_gate(query, keyed, self.hidden, count * self.branches)
        span = self.branches * self.hidden
        conv_w, conv_off = ple["conv"]
        conv_norm, conv_norm_off = ple["norm_conv"]
        for index in range(count):
            updated = self.gpu.gate_broadcast(
                gate.view(index * self.branches, self.branches),
                value.view(index * self.hidden, self.hidden),
                self.branches,
                self.hidden,
            )
            normed = self.gpu.rms(
                updated,
                conv_norm,
                conv_norm_off,
                self.hidden,
                self.branches,
                self.eps,
                True,
                self.branches,
            )
            conv = self.gpu.dilated_conv(
                self.ple_hist, normed, conv_w, conv_off, span, 4, self.ngram_size
            )
            slot = residual.view(index * span, span)
            self.gpu.add_into(slot, self.gpu.add(updated, self.gpu.silu(conv)))

    def _ngram_embedding(self, rows: list[int]) -> Dev:
        if self._bf16:
            stacked = np.concatenate([self.files.ngram_vector(int(row)) for row in rows])
            return self.gpu.dev_from(stacked)
        if not isinstance(self.files, QuantRecord):
            message = "n-gram load has no record"
            raise RuntimeError(message)
        return self.gpu.ngram_dequant(self.files.ngram_io, rows)

    def _ple_bundle(self) -> dict[str, Any]:
        if self._ple is not None:
            return self._ple
        base = "model.language_model.layers.1.ple"
        table = "model.language_model.layers.1.ple.ple_embedding"
        self._ple = {
            "key": self._int4(f"{base}.key_proj.weight"),
            "value": self._int4(f"{base}.value_proj.weight"),
            "conv": self._weight(f"{base}.conv1d.weight"),
            "norm_conv": self._weight(f"{base}.norm_conv.weight"),
            "norm_key": self._weight(f"{base}.norm_key.weight"),
            "norm_query": self._weight(f"{base}.norm_query.weight"),
            "mult": self.i64[f"{table}.layer_multipliers"],
            "offset": self.i64[f"{table}.ngram_heads_offsets"],
            "vocab": self.i64[f"{table}.ngram_heads_vocab_sizes"],
        }
        return self._ple

    def _final_bundle(self) -> dict[str, Any]:
        if self._final is None:
            base = "model.language_model.hyper_connection_mixer"
            self._final = {
                "norm": self._weight(f"{base}.hc_norm.weight"),
                "down": self._int4(f"{base}.input_mix_weight_down.weight"),
                "up": self._int4(f"{base}.input_mix_weight_up.weight"),
            }
        return self._final

    def _lm_head(self) -> Int4Matrix:
        if self._head is None:
            self._head = self._int4("lm_head.weight")
        return self._head

    def encode_chat(
        self,
        messages: list[dict[str, Any]],
        tools: Any,
        enable_thinking: bool,
        reasoning_effort: str | None = None,
        continue_final: bool = False,
    ) -> list[int]:
        from jinja2 import Environment

        def raise_exception(message: str) -> None:
            raise RuntimeError(message)

        template = Environment().from_string(self.chat_template)
        rendered = {
            "messages": normalize_messages(messages),
            "tools": tools,
            "add_generation_prompt": not continue_final,
            "enable_thinking": enable_thinking,
            "raise_exception": raise_exception,
        }
        if reasoning_effort:
            rendered["reasoning_effort"] = reasoning_effort
        text = template.render(**rendered)
        if continue_final:
            text = _drop_final_close(text)
        return list(self.tokenizer.encode(text).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


def _drop_final_close(text: str) -> str:
    """Leave the last assistant turn open so generation continues it."""
    close = "<|im_end|>\n"
    if text.endswith(close):
        return text[: -len(close)]
    return text
