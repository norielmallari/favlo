"""Qwen3.8-Flash-Next forward on Metal."""

from __future__ import annotations

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
    HEADROOM,
    INDEX_STRIDE,
    PREFILL_CHUNK,
    kv_bytes_per_token,
    machine_memory,
    prompt_token_limit,
    resident_candidates,
    residual_bytes_per_token,
    scratch_bytes,
)
from qwen38_flash_next.convert import ShardStore, _layer_f32_keys, _layer_int4_keys
from qwen38_flash_next.experts import ExpertAtlas, group_tier
from qwen38_flash_next.formats import (
    BLOCK,
    NGRAM_WIDTH,
    ExpertPlanes,
    expert_planes,
    int4_plane_sizes,
    stored_k_int4,
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
)
from qwen38_flash_next.tools import (
    canonical_continuation,
    continuing_ids,
    forced_call_options,
    function_name_options,
    normalize_messages,
)


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


class Qwen38Runtime:
    """One sequence of Qwen3.8-Flash-Next. Linear maps are Metal shader dots."""

    def __init__(
        self, root: Path, *, bf16: bool = False, resident_percent: float | None = None
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
        self._predicted: dict[int, list[int]] = {}
        self._prefetch_events: dict[int, int] = {}
        self._prefetch_protect: set[tuple[int, int]] = set()
        self._residual_buf: Any = None
        self._residual_n = 0
        self._decoding = False
        self._force_sync = False
        self.atlas: ExpertAtlas | None = None
        self._streaming_prefill = False
        self._stage: list[Any] | None = None
        self._stage_table: Any = None
        self._stage_miss: Any = None
        self._stage_event = [0, 0]
        self._stage_bytes = 0
        self._hold_xs: Any = None
        self._hold_out: Any = None
        self._hold_ids: Any = None
        self._hold_w: Any = None
        self._hold_cap = 0
        self._hot = expert_planes(4, self.hidden, self.moe)
        self._cold = expert_planes(2, self.hidden, self.moe)
        self._super_planes = expert_planes(8, self.hidden, self.moe)
        self._super_images: dict[tuple[int, int], Any] = {}
        self.tokens: list[int] = []
        self.last_logits: Dev | None = None
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self._decode_steps: list[tuple[float, float, int]] = []
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
        self.kv_k = {index: self.gpu.alloc(512 * 4) for index in attention}
        self.kv_v = {index: self.gpu.alloc(512 * 4) for index in attention}
        self.indexer_buf = {index: self.gpu.alloc(128 * 4) for index in attention}
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
            self._load_record()
            self._load_supers()
            state_bytes = 48 * 128 * 128 * 4
            self._gdn_shadow = {
                index: self.gpu.alloc(state_bytes) for index in self.gdn_state
            }
            self._conv_shadow = {
                index: self.gpu.alloc(self._conv_bytes) for index in self.conv_hist
            }
            self._ple_shadow = self.gpu.alloc(self._ple_bytes)
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
        self._set_prompt_limit(resident_percent)

    def _set_prompt_limit(self, resident_percent: float | None) -> None:
        self.resident_percent = resident_percent
        self.prefill_chunk = PREFILL_CHUNK
        attention = sum(kind == "full_attention" for kind in self.kinds)
        fixed = HEADROOM
        staging = 0
        if isinstance(self.files, QuantRecord):
            fixed += (self.files.root / "resident.bin").stat().st_size
            hot = sum(os.fstat(fd).st_size for fd in self.files.hot_fds)
            self.resident_candidates = resident_candidates(hot, resident_percent)
            self.resident_budget = self.resident_candidates[0]
            staging = self._stage_budget() * 2
        else:
            self.resident_candidates = ()
            self.resident_budget = 0
        index_row = PREFILL_CHUNK * INDEX_STRIDE * 4
        self.prompt_limit = prompt_token_limit(
            memory=machine_memory(),
            resident_hot=self.resident_budget,
            kv_per_token=kv_bytes_per_token(attention),
            residual_per_token=residual_bytes_per_token(self.branches, self.hidden),
            scratch=scratch_bytes(PREFILL_CHUNK) + index_row,
            staging=staging,
            fixed=fixed,
            score_per_token=PREFILL_CHUNK,
        )
        _log(
            f"prompt limit {self.prompt_limit} tokens, "
            f"resident budget {self.resident_budget >> 20} MiB"
        )

    def _require_prompt_room(self, total: int) -> None:
        if total <= self.prompt_limit:
            return
        message = (
            f"prompt of {total} tokens exceeds the limit of {self.prompt_limit} "
            f"(resident experts {self.resident_budget >> 20} MiB)"
        )
        raise RuntimeError(message)

    def _add_stop_token(self, piece: Any) -> None:
        """A single-id end marker stops generation. The n-gram boundary stays `eos`."""
        if not isinstance(piece, str) or piece == "":
            return
        encoded = self.tokenizer.encode(piece).ids
        if len(encoded) == 1:
            self.stop_ids.add(int(encoded[0]))

    def reset(self) -> None:
        self.tokens = []
        self.last_logits = None
        for buffer in self.gdn_state.values():
            _fill_buffer(buffer, 48 * 128 * 128 * 4, 0)
        for history in self.conv_hist.values():
            _fill_buffer(history, self._conv_bytes, 0)
        _fill_buffer(self.ple_hist, self._ple_bytes, 0)
        for index in self.kv_len:
            self.kv_len[index] = 0

    def align(self, prompt: list[int]) -> int:
        if prompt[: len(self.tokens)] != self.tokens:
            _log("prefix miss")
            self.reset()
            return 0
        _log(f"prefix hit {len(self.tokens)}")
        return len(self.tokens)

    def generate(
        self,
        prompt: list[int],
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[int]:
        start = self.align(prompt)
        if start == len(prompt):
            if self.last_logits is None:
                message = "the cached prefix has no logits"
                raise RuntimeError(message)
            logits = self.last_logits
            _log("prefill 0 tok")
        else:
            logits = self.forward(prompt[start:])
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
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self._decode_steps = []
        self._hit_limit = False
        produced_ids: list[int] = []
        produced = 0
        while True:
            if max_tokens is not None and produced >= max_tokens:
                self._hit_limit = True
                return
            if len(self.tokens) + 1 > self.prompt_limit:
                self._hit_limit = True
                return
            token = self._pick(logits, temperature, top_p, top_k, produced_ids)
            if token in self.stop_ids:
                return
            yield token
            produced += 1
            produced_ids.append(token)
            if self.required_function and "</tool_call>" in self.decode(produced_ids):
                return
            # The prefill logits keep that scratch alive. Release them before the
            # decode arenas are allocated.
            self.last_logits = None
            logits = None
            logits = self.forward([token], decode=True)

    def _log_decode(self) -> None:
        _log(_tok_s("decode", self._decode_tokens, self._decode_seconds))
        resident = [step[2] for step in self._decode_steps]
        start = after_atlas_load(resident)
        filled = self._decode_steps[:start]
        steady = self._decode_steps[start:]
        if filled:
            _log(
                "decode atlas load "
                f"{len(filled)} tok in {sum(step[0] for step in filled):.2f}s"
            )
        if not steady:
            _log("decode after atlas load: atlas still filling")
            return
        wall = sum(step[0] for step in steady)
        gpu = sum(step[1] for step in steady)
        _log(_tok_s("decode after atlas load", len(steady), wall))
        if gpu > 0:
            _log(
                "decode after atlas load "
                f"{len(steady) / gpu:.2f} tok/s ({len(steady)} tok in {gpu:.2f}s gpu)"
            )

    def _pick(
        self,
        logits: Any,
        temperature: float,
        top_p: float,
        top_k: int,
        produced: list[int],
    ) -> int:
        if self.tool_names and isinstance(logits, Dev):
            constrained = self._constrained_name(logits, temperature, produced)
            if constrained is not None:
                return constrained
        if isinstance(logits, Dev):
            return self.gpu.sample(logits, temperature, top_p, top_k, self.rng)
        return _sample(logits, temperature, top_p, top_k, self.rng)

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

    def forward(self, tokens: list[int], *, decode: bool = False) -> Dev:
        started = time.perf_counter()
        gpu_at = self.gpu.gpu_seconds
        self._decoding = decode
        stage_bytes = 0
        total = len(self.tokens) + len(tokens)
        self._require_prompt_room(total)
        chunk = len(tokens) if decode else min(len(tokens), self.prefill_chunk)
        scratch = scratch_bytes(max(chunk, 1))
        if not decode and total > DENSE_LIMIT:
            blocks = max(total // INDEX_BLOCK, 1)
            scratch += chunk * blocks * 4 + chunk * INDEX_STRIDE * 4
        self.gpu.begin_scratch(scratch, copies=2 if decode else 1)
        if not self._bf16:
            if decode:
                self._ensure_atlas()
            else:
                self._begin_prefill_stream()
        streamed = self.atlas.streamed if self.atlas is not None else 0
        if self.atlas is not None:
            _fill_buffer(self.atlas.miss, self.layers * 4, 0)
        if decode and self.atlas is not None:
            self._prefetch_decode()
            self._capture_recurrent()
        logits = self._forward_body(tokens)
        self.gpu.flush()
        if self._streaming_prefill:
            missed = self._stage_missed()
            stage_bytes = self._stage_bytes
            self._end_prefill_stream()
            if missed:
                message = "prefill referenced an expert that was not loaded"
                raise RuntimeError(message)
            self._ensure_atlas()
        elif self.atlas is not None and self._expert_miss():
            if not decode:
                message = "prefill referenced an expert that was not loaded"
                raise RuntimeError(message)
            logits = self._rerun_decode(tokens)
        self.tokens.extend(tokens)
        self.last_logits = logits
        elapsed = time.perf_counter() - started
        gpu_elapsed = self.gpu.gpu_seconds - gpu_at
        if decode:
            self._decode_tokens += len(tokens)
            self._decode_seconds += elapsed
            resident = self.atlas.resident_count if self.atlas is not None else 0
            self._decode_steps.append((elapsed, gpu_elapsed, resident))
        else:
            _log(_tok_s("prefill", len(tokens), elapsed))
            if gpu_elapsed > 0 and len(tokens) > 0:
                _log(
                    "prefill after load "
                    f"{len(tokens) / gpu_elapsed:.2f} tok/s "
                    f"({len(tokens)} tok in {gpu_elapsed:.2f}s gpu)"
                )
            if stage_bytes:
                _log(f"experts streamed {stage_bytes >> 20} MiB")
            elif self.atlas is not None:
                delta = self.atlas.streamed - streamed
                _log(
                    f"experts resident {self.atlas.resident_count}"
                    f" streamed {delta >> 20} MiB"
                )
        return logits

    def _forward_body(self, tokens: list[int]) -> Dev:
        count = len(tokens)
        self._reserve_caches(len(self.tokens) + count)
        residual = self._embed_dev(tokens)
        span = self.branches * self.hidden
        step = count if self._decoding else self.prefill_chunk
        wide = self._streaming_prefill and count > self.prefill_chunk
        if wide:
            self._ensure_hold(count)
        for layer in range(self.layers):
            layer_started = time.perf_counter()
            held = 0
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
                # open until prefill needs expert ids, or the host reads the token.
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
                            held = len(self.expert_cache)
                            self.expert_cache.clear()
                        self._inject_dev(view, hats, mixed, bundle["mlp_inject"], n)
                    if self._decoding and self.atlas is not None:
                        self.gpu.submit()
            if wide:
                if bundle is None:
                    message = "prefill layer had no tokens"
                    raise RuntimeError(message)
                self._moe_wide(layer, bundle, residual, count)
            if self._bf16:
                _log(f"layer {layer} stepped {count} experts {held}")
            elif count > DENSE_LIMIT:
                elapsed = time.perf_counter() - layer_started
                _log(f"layer {layer} {elapsed:.2f}s")
        self.gpu.rewind_scratch()
        final = self._final_bundle()
        last = residual.view(
            (count - 1) * self.branches * self.hidden, self.branches * self.hidden
        )
        hat = self._rms_branches(last, final["norm"], 1)
        hidden = self._gr_dev(final["down"], final["up"], hat, 1)
        return self.gpu.linear(self._lm_head(), hidden, 1, self.hidden)

    def _rerun_decode(self, tokens: list[int]) -> Dev:
        """Load whatever this token routed to, then run it again from the snapshot."""
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

    def _ensure_atlas(self) -> None:
        if self._bf16 or self.atlas is not None:
            return
        if not isinstance(self.files, QuantRecord):
            return
        self.atlas = ExpertAtlas(
            self.gpu, self.files, self.layers, self.experts, self.resident_candidates
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

    def _begin_prefill_stream(self) -> None:
        """Two layer buffers. The atlas stays unallocated until this call returns."""
        if not isinstance(self.files, QuantRecord):
            return
        self._end_prefill_stream()
        self.atlas = None
        budget = self._stage_budget()
        self._stage = [self.gpu.alloc(budget), self.gpu.alloc(budget)]
        if self._stage[0] is None or self._stage[1] is None:
            message = f"prefill stage of {budget} bytes was not allocated"
            raise RuntimeError(message)
        self._stage_table = self.gpu.alloc(self.experts * 8)
        self._stage_miss = self.gpu.zeros(self.layers * 4)
        self._stage_event = [0, 0]
        self._stage_bytes = 0
        self._streaming_prefill = True
        self._submit_prefill_layer(0)
        _log(f"prefill stage {budget >> 20} MiB x2")

    def _end_prefill_stream(self) -> None:
        self._streaming_prefill = False
        self._stage = None
        self._stage_table = None
        self._stage_miss = None
        self._stage_event = [0, 0]

    def _submit_prefill_layer(self, layer: int) -> None:
        if not isinstance(self.files, QuantRecord) or self._stage is None:
            message = "prefill stream has no tier files"
            raise RuntimeError(message)
        hot_bytes, cold_bytes, _table = prefill_image_table(
            self.files.hot[layer],
            self.files.super[layer],
            self.experts,
            self.files.slot,
            self.files.cold_slot,
        )
        buf = self._stage[layer % 2]
        hot_size = os.fstat(self.files.hot_fds[layer]).st_size
        cold_size = os.fstat(self.files.cold_fds[layer]).st_size
        if hot_size < hot_bytes or cold_size < cold_bytes:
            message = f"layer {layer} tier file is shorter than the image table"
            raise RuntimeError(message)
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
        _hot, _cold, table = prefill_image_table(
            self.files.hot[layer],
            self.files.super[layer],
            self.experts,
            self.files.slot,
            self.files.cold_slot,
        )
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
        """Queue the previous token's experts before this token's shaders start."""
        self._prefetch_events = {}
        if self.atlas is None or not self._predicted:
            self._prefetch_protect = set()
            return
        protect = {
            (layer, expert) for layer, ids in self._predicted.items() for expert in ids
        }
        self._prefetch_protect = protect
        for layer, ids in self._predicted.items():
            self._prefetch_events[layer] = self.atlas.stage(layer, ids, protect)

    def _capture_recurrent(self) -> None:
        self._kv_snap = {
            index: (
                self.kv_k[index],
                self.kv_v[index],
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

    def _restore_recurrent(self) -> None:
        for index, (key, value, indexer, length) in self._kv_snap.items():
            self.kv_k[index] = key
            self.kv_v[index] = value
            self.indexer_buf[index] = indexer
            self.kv_len[index] = length
        state_bytes = 48 * 128 * 128 * 4
        for layer, buf in self.gdn_state.items():
            self.gpu.copy_bytes(self._gdn_shadow[layer], 0, buf, 0, state_bytes)
        for layer, buf in self.conv_hist.items():
            self.gpu.copy_bytes(self._conv_shadow[layer], 0, buf, 0, self._conv_bytes)
        self.gpu.copy_bytes(self._ple_shadow, 0, self.ple_hist, 0, self._ple_bytes)

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
        _log("loading quantized record")
        if not isinstance(self.files, QuantRecord):
            message = "quantized load has no record"
            raise RuntimeError(message)
        self.files.open_io(self.gpu)
        manifest = self.files.manifest
        size = (self.files.root / "resident.bin").stat().st_size
        self._resident = self.gpu.alloc(size)
        self.gpu.io_read(self.files.resident_io, [(self._resident, 0, size, 0)])
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
        _log(f"loaded {len(self.weights)} int4 matrices")

    def _load_bf16(self) -> None:
        """Upload the checkpoint's own values. Experts and n-gram rows stay on disk."""
        store = self.store
        if store is None:
            message = "BF16 load has no shard store"
            raise RuntimeError(message)
        _check_f32_gemv(self.gpu)
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
        _log(f"loaded {len(self.weights)} bf16 matrices")

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
            self._reserve(self.kv_k, layer, length, extra, 512)
            self._reserve(self.kv_v, layer, length, extra, 512)
            self._reserve(self.indexer_buf, layer, length, extra, 128)

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
        self.gpu.gdn_conv_scan(self.conv_hist[layer], qkv, conv_w, conv_off, mixed, count)
        heads = self.gpu.tmp(count * 6144)
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
        self, layer: int, bundle: dict[str, Any], xs: Dev, count: int, *, base: int
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
            message = f"attention cache length {length} does not match position {base}"
            raise RuntimeError(message)
        k_buf = self._reserve(self.kv_k, layer, length, count, 512)
        v_buf = self._reserve(self.kv_v, layer, length, count, 512)
        i_buf = self._reserve(self.indexer_buf, layer, length, count, 128)
        self.kv_len[layer] = length + count
        self.gpu.copy_bytes(k.buf, k.offset * 4, k_buf, length * 512 * 4, count * 512 * 4)
        self.gpu.copy_bytes(value.buf, value.offset * 4, v_buf, length * 512 * 4, count * 512 * 4)
        self.gpu.indexer_write(index_qk, i_buf, count, length)
        if base + count <= DENSE_LIMIT:
            gated = self.gpu.qsa_attend(q, gate_pre, k_buf, v_buf, count, length)
        else:
            n_blocks = max((base + count - 1) // INDEX_BLOCK, 1)
            scores = self.gpu.tmp(count * n_blocks)
            indices = self.gpu.tmp(count * INDEX_STRIDE)
            selected = self.gpu.tmp(count)
            self.gpu.indexer_block_score(index_qk, i_buf, scores, count, length, n_blocks)
            self.gpu.indexer_select(
                scores, indices, selected, count, length, n_blocks, INDEX_STRIDE
            )
            gated = self.gpu.qsa_attend_selected(
                q, gate_pre, k_buf, v_buf, indices, selected, count, INDEX_STRIDE
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
        self._choice[layer] = choice
        # Stage before the expert kernels. A miss would zero that layer and
        # change every later route, so the token would have to be replayed.
        host_ids = self.gpu.read_u32(choice)
        rows = np.asarray(host_ids, dtype=np.uint32).reshape(count, self.top_k)
        self._predicted[layer] = [int(item) for item in rows[-1].tolist()]
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
        running = self._moe_int4_matmul(layer, xs, weight, running, host_ids, count)
        running = self._moe_tier_batched(
            layer,
            xs,
            weight,
            running,
            host_ids,
            count,
            self.files.cold_set(layer),
            self._cold,
            self._stage[layer % 2],
            self._stage_table,
            0,
            self._stage_miss,
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

    def _moe_int4_matmul(
        self,
        layer: int,
        xs: Dev,
        weight: Dev,
        running: Dev,
        host_ids: np.ndarray,
        count: int,
    ) -> Dev:
        if not isinstance(self.files, QuantRecord) or self._stage is None:
            message = "prefill stream has no staging buffer"
            raise RuntimeError(message)
        members = self.files.hot_set(layer)
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
        for expert, pairs in chosen.items():
            if len(pairs) >= 64:
                large.append((expert, pairs))
            else:
                small.add(expert)
        if large:
            _hot_bytes, _cold_bytes, image = prefill_image_table(
                self.files.hot[layer],
                self.files.super[layer],
                self.experts,
                self.files.slot,
                self.files.cold_slot,
            )
            stage = self._stage[layer % 2]
            for expert, pairs in large:
                self._room_for_expert()
                base = int(image[expert])
                if base < 0:
                    message = f"layer {layer} expert {expert} is not in the staging buffer"
                    raise RuntimeError(message)
                tokens = np.asarray([item[0] for item in pairs], dtype=np.uint32)
                slots = np.asarray([item[1] for item in pairs], dtype=np.uint32)
                n = int(tokens.size)
                token_buf = self._upload_u32(tokens)
                slot_buf = self._upload_u32(slots)
                gate, down = self._expert_views(stage)
                gate.weight_off += base
                gate.scale_off += base
                down.weight_off += base
                down.scale_off += base
                gathered = self.gpu.gather_half(xs, token_buf, n, self.hidden, gate.stored_inner)
                activated = self.gpu.moe_activate(
                    self.gpu.gemm_int4_half(gate, gathered, n),
                    n,
                    self.moe,
                    down.stored_inner,
                )
                mixed = self.gpu.gemm_int4_half(down, activated, n)
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
                self._hot,
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
        cap = 65536
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
        gemm = self.gpu.moe_grouped if planes.bits == 4 else self.gpu.moe_grouped_int2
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
            gate = self.gpu.gemv_int8(
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
            down = self.gpu.gemv_int8(
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
                gate = self.gpu.gemv_int8(
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
                down = self.gpu.gemv_int8(
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
        self, messages: list[dict[str, Any]], tools: Any, enable_thinking: bool
    ) -> list[int]:
        from jinja2 import Environment

        def raise_exception(message: str) -> None:
            raise RuntimeError(message)

        template = Environment().from_string(self.chat_template)
        text = template.render(
            messages=normalize_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            raise_exception=raise_exception,
        )
        return list(self.tokenizer.encode(text).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
