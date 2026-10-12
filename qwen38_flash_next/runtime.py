"""Qwen3.8-Flash-Next forward on Metal."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import objc

from qwen38_flash_next.checks import _ngram_row, _sample
from qwen38_flash_next.convert import ShardStore, _layer_f32_keys, _layer_int4_keys
from qwen38_flash_next.formats import BLOCK, int4_plane_sizes, stored_k_int4
from qwen38_flash_next.record import QuantRecord, _Bf16Ngram
from qwen38_flash_next.session import (
    Dev,
    F16Matrix,
    Int4Matrix,
    MetalSession,
    _buffer_slice,
    _check_f32_gemv,
    _device_name,
    _fill_buffer,
    _log,
    _tok_s,
)


class Qwen38Runtime:
    """One sequence of Qwen3.8-Flash-Next. Linear maps are Metal shader dots."""

    def __init__(self, root: Path, *, bf16: bool = False) -> None:
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
        self.tokens: list[int] = []
        self.last_logits: Dev | None = None
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        self.rng = np.random.default_rng()
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
        max_tokens: int,
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
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        self._decode_tokens = 0
        self._decode_seconds = 0.0
        for _ in range(max_tokens):
            token = self._pick(logits, temperature, top_p, top_k)
            if token in self.stop_ids:
                break
            yield token
            logits = self.forward([token], decode=True)

    def _log_decode(self) -> None:
        _log(_tok_s("decode", self._decode_tokens, self._decode_seconds))

    def _pick(self, logits: Any, temperature: float, top_p: float, top_k: int) -> int:
        if isinstance(logits, Dev):
            return self.gpu.sample(logits, temperature, top_p, top_k, self.rng)
        return _sample(logits, temperature, top_p, top_k, self.rng)

    def forward(self, tokens: list[int], *, decode: bool = False) -> Dev:
        started = time.perf_counter()
        count = len(tokens)
        residual = self._embed_dev(tokens)
        for layer in range(self.layers):
            # Objective-C objects from encoding this layer. The command buffer
            # stays open until the next expert-id read.
            with objc.autorelease_pool():
                if (layer + 1) in self.ple_layers:
                    self._ple_dev(residual, tokens, count)
                bundle = self._layer(layer)
                hats = self._rms_branches(residual, bundle["attn_norm"], count)
                xs = self._gr_dev(bundle["attn_down"], bundle["attn_up"], hats, count)
                if self.kinds[layer] == "linear_attention":
                    mixed = self._gdn_dev(layer, bundle, xs, count)
                elif self.kinds[layer] == "full_attention":
                    mixed = self._qsa_dev(layer, bundle, xs, count)
                else:
                    message = f"unknown layer type {self.kinds[layer]}"
                    raise RuntimeError(message)
                self._inject_dev(residual, hats, mixed, bundle["attn_inject"], count)
                hats = self._rms_branches(residual, bundle["mlp_norm"], count)
                xs = self._gr_dev(bundle["mlp_down"], bundle["mlp_up"], hats, count)
                mixed = self._moe_dev(layer, bundle, xs, count)
                if self._bf16:
                    held = len(self.expert_cache)
                    self.expert_cache.clear()
                self._inject_dev(residual, hats, mixed, bundle["mlp_inject"], count)
                if self._bf16:
                    _log(f"layer {layer} stepped {count} experts {held}")
        final = self._final_bundle()
        last = residual.view((count - 1) * self.branches * self.hidden, self.branches * self.hidden)
        hat = self._rms_branches(last, final["norm"], 1)
        hidden = self._gr_dev(final["down"], final["up"], hat, 1)
        logits = self.gpu.linear(self._lm_head(), hidden, 1, self.hidden)
        self.tokens.extend(tokens)
        self.last_logits = logits
        self.gpu.flush()
        elapsed = time.perf_counter() - started
        if decode:
            self._decode_tokens += len(tokens)
            self._decode_seconds += elapsed
        else:
            _log(_tok_s("prefill", len(tokens), elapsed))
        return logits

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
        dst = self.gpu.tmp(count * span)
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

    def _gr_dev(self, down: Any, up: Any, hats: Dev, count: int) -> Dev:
        # Tech report eq. (31): SiLU((1/n_r) W_d vec(R̂)). The branch mean is the second 1/n_r.
        flat = self.branches * self.hidden
        projected = self.gpu.linear(down, hats, count, flat)
        hidden = self.gpu.silu(self.gpu.scale(projected, 1.0 / self.branches))
        gate = self.gpu.sigmoid(self.gpu.linear(up, hidden, count, int(up.inner)))
        return self.gpu.branch_mean(gate, hats, count, self.branches, self.hidden)

    def _inject_dev(self, residual: Dev, hats: Dev, mixed: Dev, weight: Any, count: int) -> None:
        flat = self.branches * self.hidden
        inject = self.gpu.linear(weight, hats, count, flat)
        scales = self.gpu.scale(self.gpu.sigmoid(self.gpu.scale(inject, 0.25)), 2.0)
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
        heads = self.gpu.tmp(count * 6144)
        for index in range(count):
            current = qkv.view(index * 10240, 10240)
            mixed = self.gpu.silu(
                self.gpu.causal_conv(self.conv_hist[layer], current, conv_w, conv_off, 10240, 4)
            )
            query = self.gpu.l2(mixed.view(0, 2048), 16, 128, self.eps)
            key = self.gpu.l2(mixed.view(2048, 2048), 16, 128, self.eps)
            # L2 norm, then the query is scaled by 1/sqrt(head_dim). The key is not.
            query = self.gpu.scale(query, 128.0**-0.5)
            value = mixed.view(4096, 6144)
            query = self.gpu.repeat_rows(query, 16, 128, 3)
            key = self.gpu.repeat_rows(key, 16, 128, 3)
            beta = self.gpu.sigmoid(raw_b.view(index * 48, 48))
            alpha = self.gpu.gdn_alpha(raw_a.view(index * 48, 48), dt_w, dt_off, a_w, a_off)
            y = self.gpu.gdn_dev(self.gdn_state[layer], query, key, value, alpha, beta)
            y = self.gpu.rms(y, norm_w, norm_off, 128, 48, self.eps, False, 0)
            gated = self.gpu.mul(y, self.gpu.sigmoid(z.view(index * 6144, 6144)))
            self.gpu.copy_into(gated, heads, index * 6144)
        return self.gpu.linear(bundle["out"], heads, count, 6144)

    def _grow_slot(
        self, table: dict[int, Any], layer: int, length: int, width: int
    ) -> tuple[Any, int]:
        need = (length + 1) * width * 4
        buf = table[layer]
        if buf.length() < need:
            grown = self.gpu.alloc(max(need * 2, width * 4))
            if length:
                self.gpu._retain.append(buf)
                self.gpu.copy_span(Dev(buf, length * width), 0, grown, 0, length * width)
            table[layer] = grown
            buf = grown
        return buf, length * width

    def _qsa_dev(self, layer: int, bundle: dict[str, Any], xs: Dev, count: int) -> Dev:
        hidden = self.hidden
        query = self.gpu.linear(bundle["q_proj"], xs, count, hidden)
        key = self.gpu.linear(bundle["k_proj"], xs, count, hidden)
        value = self.gpu.linear(bundle["v_proj"], xs, count, hidden)
        index_qk = self.gpu.linear(bundle["index_qk"], xs, count, hidden)
        q_w, q_off = bundle["q_norm"]
        k_w, k_off = bundle["k_norm"]
        gated = self.gpu.tmp(count * 6144)
        base = len(self.tokens)
        for index in range(count):
            position = base + index
            # Fused q_proj is per head [query, sigmoid gate], 256 + 256.
            packed = query.view(index * 12288, 12288)
            q_part, gate_pre = self.gpu.split_q(packed)
            q = self.gpu.rms(q_part, q_w, q_off, 256, 24, self.eps, True, 0)
            k = self.gpu.rms(key.view(index * 512, 512), k_w, k_off, 256, 2, self.eps, True, 0)
            q = self.gpu.rope(q, 24, 256, self.rotary_dim, position, self.rope_theta)
            k = self.gpu.rope(k, 2, 256, self.rotary_dim, position, self.rope_theta)
            length = self.kv_len[layer]
            k_buf, k_at = self._grow_slot(self.kv_k, layer, length, 512)
            v_buf, v_at = self._grow_slot(self.kv_v, layer, length, 512)
            i_buf, i_at = self._grow_slot(self.indexer_buf, layer, length, 128)
            self.kv_len[layer] = length + 1
            self.gpu.copy_span(k, 0, k_buf, k_at, 512)
            self.gpu.copy_span(value, index * 512, v_buf, v_at, 512)
            self.gpu.copy_span(index_qk, index * 640 + 512, i_buf, i_at, 128)
            tokens = self.kv_len[layer]
            if tokens // 4 > 512:
                message = "sequence is past the 2048-token indexer budget"
                raise RuntimeError(message)
            context = self.gpu.attend_dev(q, k_buf, v_buf, tokens)
            merged = self.gpu.sigmoid_mul(context, gate_pre)
            self.gpu.copy_into(merged, gated, index * 6144)
        return self.gpu.linear(bundle["o_proj"], gated, count, 6144)

    def _moe_dev(self, layer: int, bundle: dict[str, Any], xs: Dev, count: int) -> Dev:
        hidden = self.hidden
        gate = self.gpu.silu(self.gpu.linear(bundle["shared_gate"], xs, count, hidden))
        up = self.gpu.linear(bundle["shared_up"], xs, count, hidden)
        shared = self.gpu.linear(bundle["shared_down"], self.gpu.mul(gate, up), count, self.moe)
        coeff = self.gpu.sigmoid(self.gpu.linear(bundle["shared_coeff"], xs, count, hidden))
        shared = self.gpu.scale_rows(shared, coeff, count, hidden)
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
            buf = self.gpu.alloc(self.files.live)
            buffers.append((expert, buf))
            loads.append((buf, 0, self.files.live, expert * self.files.slot))
        self.gpu.io_read(self.files.expert_io[layer], loads, gpu_wait=True)
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

    def _ple_dev(self, residual: Dev, tokens: list[int], count: int) -> None:
        """N-gram memory injected before this layer's residual read."""
        ple = self._ple_bundle()
        span = self.branches * self.hidden
        conv_w, conv_off = ple["conv"]
        for index in range(count):
            history = self.tokens + tokens[: index + 1]
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
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            raise_exception=raise_exception,
        )
        return list(self.tokenizer.encode(text).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
