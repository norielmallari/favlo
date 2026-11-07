"""Qwen3-4B caption prefill.

The prompt is the checkpoint chat template with a user turn and a generation
prompt. ``enable_thinking`` is left unset, which matches the published encoder
(only ``false`` inserts an empty think block). The vector that enters
``cap_embedder`` is the penultimate layer, before ``model.norm``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment
from tokenizers import Tokenizer

from z_image_turbo.formats import TILE_M, pad_to
from z_image_turbo.metal import Gpu
from z_image_turbo.ops import project
from z_image_turbo.record import Plane
from z_image_turbo.scratch import TEXT_FFN, TEXT_HIDDEN, TEXT_Q, Scratch

LAYERS = 36
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
EPS = 1e-6
THETA = 1_000_000.0
MAX_TOKENS = 1536


def prompt_ids(prompt: str, source: Path) -> np.ndarray:
    config = json.loads((source / "tokenizer" / "tokenizer_config.json").read_text())
    env = Environment(autoescape=False)
    text = env.from_string(config["chat_template"]).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    tokenizer = Tokenizer.from_file(str(source / "tokenizer" / "tokenizer.json"))
    ids = np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.uint32)
    if ids.size == 0 or ids.size > MAX_TOKENS:
        message = f"caption length {ids.size} does not fit in {MAX_TOKENS}"
        raise ValueError(message)
    return ids


def embed_caption(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    ids: np.ndarray,
    *,
    on_layer: Callable[[], None] | None = None,
) -> tuple[Any, int]:
    """Return the caption buffer (stride 3840) and the token count."""
    limit = int(ids.size)
    rows = pad_to(limit, TILE_M)
    if rows > scratch.cap_rows:
        message = f"caption pad {rows} exceeds {scratch.cap_rows}"
        raise ValueError(message)
    hidden = _prefill(gpu, scratch, weights, ids, rows, limit, on_layer)
    project(
        gpu,
        hidden,
        weights["cap_embedder.1.weight"],
        weights["cap_embedder.1.bias"],
        scratch.text_out,
        scratch.cap,
        rows,
        TEXT_HIDDEN,
    )
    # cap_embedder.0 is the RMSNorm gain on the text hidden, applied above.
    if on_layer is not None:
        on_layer()
    return scratch.cap, limit


def _prefill(
    gpu: Gpu,
    scratch: Scratch,
    weights: dict[str, Plane],
    ids: np.ndarray,
    rows: int,
    limit: int,
    on_layer: Callable[[], None] | None = None,
) -> Any:
    table = weights["model.embed_tokens.weight"]
    gather = gpu.gather_i8 if table.plane == "int8" else gpu.gather_fp4
    gather(
        gpu.upload(np.ascontiguousarray(ids)),
        table.data,
        table.scale,
        scratch.text_h,
        rows,
        limit,
        TEXT_HIDDEN,
        table.stored_k,
    )
    # The gather wrote `limit` rows. Norm the text hidden before the projection
    # with the DiT caption norm. The stack below uses the text eps.
    gpu.text_positions(scratch.pos_text, rows, limit, 0)
    positions = scratch.pos_text
    for layer in range(LAYERS - 1):
        prefix = f"model.layers.{layer}"
        gpu.rmsnorm(
            scratch.text_h,
            weights[f"{prefix}.input_layernorm.weight"].data,
            scratch.text_h2,
            rows,
            TEXT_HIDDEN,
            EPS,
        )
        project(
            gpu,
            scratch.text_h2,
            weights[f"{prefix}.self_attn.q_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_q,
            rows,
            TEXT_HIDDEN,
        )
        project(
            gpu,
            scratch.text_h2,
            weights[f"{prefix}.self_attn.k_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_k,
            rows,
            TEXT_HIDDEN,
        )
        project(
            gpu,
            scratch.text_h2,
            weights[f"{prefix}.self_attn.v_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_v,
            rows,
            TEXT_HIDDEN,
        )
        q_rows = rows * HEADS
        kv_rows = rows * KV_HEADS
        gpu.rmsnorm(
            scratch.text_q,
            weights[f"{prefix}.self_attn.q_norm.weight"].data,
            scratch.text_a,
            q_rows,
            HEAD_DIM,
            EPS,
        )
        gpu.rmsnorm(
            scratch.text_k,
            weights[f"{prefix}.self_attn.k_norm.weight"].data,
            scratch.text_q,
            kv_rows,
            HEAD_DIM,
            EPS,
        )
        # q norm landed in text_a. k norm landed in text_q. Move q back.
        # text_q now holds normalized K, overwriting Q. Keep Q in text_a.
        gpu.rope(scratch.text_a, positions, rows, HEADS, HEAD_DIM, THETA, (HEAD_DIM, 0, 0))
        gpu.rope(scratch.text_q, positions, rows, KV_HEADS, HEAD_DIM, THETA, (HEAD_DIM, 0, 0))
        gpu.attend(
            scratch.text_a,
            scratch.text_q,
            scratch.text_v,
            scratch.text_k,
            rows,
            limit,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            True,
        )
        project(
            gpu,
            scratch.text_k,
            weights[f"{prefix}.self_attn.o_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_h2,
            rows,
            TEXT_Q,
        )
        count = rows * TEXT_HIDDEN
        gpu.add_rows(scratch.text_h, scratch.text_h2, scratch.text_h, count)
        gpu.rmsnorm(
            scratch.text_h,
            weights[f"{prefix}.post_attention_layernorm.weight"].data,
            scratch.text_h2,
            rows,
            TEXT_HIDDEN,
            EPS,
        )
        project(
            gpu,
            scratch.text_h2,
            weights[f"{prefix}.mlp.gate_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_ffn,
            rows,
            TEXT_HIDDEN,
        )
        project(
            gpu,
            scratch.text_h2,
            weights[f"{prefix}.mlp.up_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_ffn2,
            rows,
            TEXT_HIDDEN,
        )
        gpu.silu_mul(scratch.text_ffn, scratch.text_ffn2, scratch.text_ffn, rows * TEXT_FFN)
        project(
            gpu,
            scratch.text_ffn,
            weights[f"{prefix}.mlp.down_proj.weight"],
            None,
            scratch.text_out,
            scratch.text_h2,
            rows,
            TEXT_FFN,
        )
        gpu.add_rows(scratch.text_h, scratch.text_h2, scratch.text_h, count)
        if on_layer is not None:
            on_layer()
    gpu.rmsnorm(
        scratch.text_h,
        weights["cap_embedder.0.weight"].data,
        scratch.text_h2,
        rows,
        TEXT_HIDDEN,
        1e-5,
    )
    return scratch.text_h2
