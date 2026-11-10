"""Basic speech. An empty voice is text only; a named voice is a reference clip."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from voxcpm2.metal41.block import rope_factors, rope_scale
from voxcpm2.metal41.config import ModelConfig
from voxcpm2.metal41.dit import condition, fuse, project_patch, quantize, sample_patch, stop_token
from voxcpm2.metal41.metal import Session
from voxcpm2.metal41.progress import VoiceProgress, patch_limit, sample_total
from voxcpm2.metal41.stack import Cache, Stack
from voxcpm2.metal41.store import Store
from voxcpm2.metal41.tokenize import LineTokenizer
from voxcpm2.metal41.vae import Decoder


def generate(
    session: Session,
    store: Store,
    decoder: Decoder,
    tokenizer: LineTokenizer,
    config: ModelConfig,
    text: str,
    *,
    max_len: int,
    min_len: int,
    steps: int,
    seed: int | None,
    progress: VoiceProgress,
    stop: Any = None,
) -> np.ndarray:
    """Speak ``text`` and return one float32 waveform at the decoder rate."""
    tokens = tokenizer.encode(text)
    limit = patch_limit(int(tokens.shape[0]) - 1, max_len, config.max_length)
    progress.open(sample_total(limit, steps))
    progress.stage("text")
    hidden = store.embed(tokens)
    base = _tower(
        session,
        store,
        config,
        "base_lm.layers",
        "base_lm.norm.weight",
        layers=config.lm.num_hidden_layers,
        hidden=config.lm.hidden_size,
        heads=config.lm.num_attention_heads,
        kv_heads=config.lm.num_key_value_heads,
        head_dim=config.lm.kv_channels,
        rope=True,
    )
    residual = _tower(
        session,
        store,
        config,
        "residual_lm.layers",
        "residual_lm.norm.weight",
        layers=config.residual_layers,
        hidden=config.lm.hidden_size,
        heads=config.lm.num_attention_heads,
        kv_heads=config.lm.num_key_value_heads,
        head_dim=config.lm.kv_channels,
        rope=not config.residual_no_rope,
    )
    encoder = _tower(
        session,
        store,
        config,
        "feat_encoder.encoder.layers",
        "feat_encoder.encoder.norm.weight",
        layers=config.encoder.layers,
        hidden=config.encoder.hidden,
        heads=config.encoder.heads,
        kv_heads=config.lm.num_key_value_heads,
        head_dim=config.encoder.kv_channels,
        rope=True,
    )
    dit = _tower(
        session,
        store,
        config,
        "feat_decoder.estimator.decoder.layers",
        "feat_decoder.estimator.decoder.norm.weight",
        layers=config.dit.layers,
        hidden=config.dit.hidden,
        heads=config.dit.heads,
        kv_heads=config.lm.num_key_value_heads,
        head_dim=config.dit.kv_channels,
        rope=True,
    )
    positions = np.arange(hidden.shape[0], dtype=np.int32)
    room = int(hidden.shape[0]) + limit
    base_cache = Cache(base.layers, base.kv_heads, base.head_dim, capacity=room)
    res_cache = Cache(residual.layers, residual.kv_heads, residual.head_dim, capacity=room)
    try:
        lm = base.forward(hidden, positions, True, base_cache)
        fused = fuse(session, store, lm, np.zeros_like(lm))
        res = residual.forward(fused, positions, True, res_cache)
        lm_hidden = lm[-1]
        res_hidden = res[-1]
        cond = np.zeros((config.feat_dim, config.patch_size), dtype=np.float32)
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        patches: list[np.ndarray] = []
        for index in range(limit):
            if stop is not None:
                stop()
            patch = index + 1
            progress.stage(f"patch {patch}/{limit}")
            started = time.perf_counter()
            mu = condition(session, store, lm_hidden, res_hidden)

            def begin(step: int, total: int, patch: int = patch) -> None:
                progress.stage(f"patch {patch}/{limit} step {step}/{total}")

            pred = sample_patch(
                session,
                store,
                dit,
                mu,
                cond,
                steps=steps,
                cfg=config.cfg_rate,
                mean_mode=config.mean_mode,
                rng=rng,
                on_step=progress.advance,
                before_step=begin,
            )
            progress.stage(f"patch {patch}/{limit}")
            progress.note(f"patch {patch}/{limit} {time.perf_counter() - started:.2f}s")
            patches.append(pred)
            cond = pred
            if index > min_len and stop_token(session, store, lm_hidden) == 1:
                break
            feat = project_patch(session, store, encoder, pred)
            lm_hidden = quantize(
                session,
                store,
                base.forward(feat, np.array([base_cache.length]), True, base_cache),
                config.fsq_scale,
            )
            res_hidden = residual.forward(
                fuse(session, store, lm_hidden, feat),
                np.array([res_cache.length]),
                True,
                res_cache,
            )
        progress.patches = len(patches)
        progress.stage("decode")
        started = time.perf_counter()
        latent = np.concatenate(patches, axis=1)
        wave = decoder.decode(latent)
        progress.decode_s = time.perf_counter() - started
        progress.advance()
        return wave
    finally:
        base_cache.release()
        res_cache.release()


def _tower(
    session: Session,
    store: Store,
    config: ModelConfig,
    prefix: str,
    norm: str,
    *,
    layers: int,
    hidden: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    rope: bool,
) -> Stack:
    rope_cfg = config.lm.rope_scaling
    factors = rope_factors(1, rope_cfg, config.lm.max_position_embeddings)
    scale = rope_scale(config.lm.max_position_embeddings, rope_cfg.original_max_position_embeddings)
    return Stack(
        session,
        store,
        prefix,
        norm,
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        eps=config.lm.rms_norm_eps,
        rope=rope,
        theta=config.lm.rope_theta,
        factors=factors,
        scale=scale,
    )
