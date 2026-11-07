"""Local encoder, local diffusion transformer, and the Euler solver."""

from __future__ import annotations

import math

import numpy as np

from voxcpm2.block import silu
from voxcpm2.fsq import quantize_bound
from voxcpm2.metal import Session
from voxcpm2.stack import Cache, Stack
from voxcpm2.store import Store


def diffusion_schedule(steps: int, sway: float = 1.0) -> np.ndarray:
    """Swayed timesteps from 1 down to 0. The endpoints stay put."""
    span = np.linspace(1.0, 0.0, int(steps) + 1, dtype=np.float64)
    return span + float(sway) * (np.cos(np.pi / 2.0 * span) - 1.0 + span)


def zero_init_steps(span: np.ndarray) -> int:
    """The first Euler steps whose velocity is zero."""
    return max(1, int(len(span) * 0.04))


def project_patch(session: Session, store: Store, stack: Stack, patch: np.ndarray) -> np.ndarray:
    """Encode one latent patch and map it into the language-model width."""
    frames = store.linear(
        session,
        "feat_encoder.in_proj.weight",
        np.asarray(patch, dtype=np.float32).T,
        "feat_encoder.in_proj.bias",
    )
    special = store.vector("feat_encoder.special_token").reshape(1, -1)
    sequence = np.concatenate((special, frames), axis=0)
    cache = Cache(stack.layers, stack.kv_heads, stack.head_dim)
    hidden = stack.forward(sequence, np.arange(sequence.shape[0]), False, cache)
    return store.linear(session, "enc_to_lm_proj.weight", hidden[0], "enc_to_lm_proj.bias")


def quantize(session: Session, store: Store, hidden: np.ndarray, scale: int) -> np.ndarray:
    """Finite scalar quantizer used on audio positions."""
    projected = store.linear(session, "fsq_layer.in_proj.weight", hidden, "fsq_layer.in_proj.bias")
    bounded = quantize_bound(projected, scale).astype(np.float32)
    return store.linear(session, "fsq_layer.out_proj.weight", bounded, "fsq_layer.out_proj.bias")


def fuse(session: Session, store: Store, hidden: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Concat the language-model state with the masked patch embedding."""
    joined = np.concatenate((hidden, feat), axis=-1)
    return store.linear(
        session,
        "fusion_concat_proj.weight",
        joined,
        "fusion_concat_proj.bias",
    )


def stop_token(session: Session, store: Store, hidden: np.ndarray) -> int:
    """Argmax of the two-way stop head. ``1`` ends the clip."""
    mid = silu(store.linear(session, "stop_proj.weight", hidden, "stop_proj.bias"))
    logits = store.linear(session, "stop_head.weight", mid)
    return int(np.argmax(logits))


def sample_patch(
    session: Session,
    store: Store,
    stack: Stack,
    mu: np.ndarray,
    cond: np.ndarray,
    *,
    steps: int,
    cfg: float,
    mean_mode: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """One latent patch, ``[channels, frames]``, from the swayed Euler solver."""
    span = diffusion_schedule(steps)
    time = float(span[0])
    dt = float(span[0] - span[1])
    skip = zero_init_steps(span)
    latent = rng.standard_normal((int(cond.shape[0]), int(cond.shape[1]))).astype(np.float32)
    blank = np.zeros_like(mu)
    for step in range(1, len(span)):
        if step <= skip:
            velocity = np.zeros_like(latent)
        else:
            positive = _estimate(session, store, stack, latent, mu, time, cond, dt, mean_mode)
            negative = _estimate(session, store, stack, latent, blank, time, cond, dt, mean_mode)
            star = float(np.sum(positive * negative) / (np.sum(negative * negative) + 1e-8))
            velocity = negative * star + float(cfg) * (positive - negative * star)
        latent = latent - np.float32(dt) * velocity
        time = time - dt
        if step < len(span) - 1:
            dt = time - float(span[step + 1])
    return latent


def condition(
    session: Session,
    store: Store,
    lm_hidden: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """The two tokens the diffusion transformer reads as ``mu``."""
    left = store.linear(session, "lm_to_dit_proj.weight", lm_hidden, "lm_to_dit_proj.bias")
    right = store.linear(session, "res_to_dit_proj.weight", residual, "res_to_dit_proj.bias")
    return np.concatenate((left, right), axis=-1)


def _estimate(
    session: Session,
    store: Store,
    stack: Stack,
    latent: np.ndarray,
    mu: np.ndarray,
    time: float,
    cond: np.ndarray,
    dt: float,
    mean_mode: bool,
) -> np.ndarray:
    frames = store.linear(
        session,
        "feat_decoder.estimator.in_proj.weight",
        latent.T,
        "feat_decoder.estimator.in_proj.bias",
    )
    context = store.linear(
        session,
        "feat_decoder.estimator.cond_proj.weight",
        cond.T,
        "feat_decoder.estimator.cond_proj.bias",
    )
    delta = dt if mean_mode else 0.0
    token = _mlp(session, store, "time_mlp", time) + _mlp(session, store, "delta_time_mlp", delta)
    token = token.reshape(1, -1)
    tokens = np.concatenate((mu.reshape(-1, frames.shape[-1]), token, context, frames), axis=0)
    cache = Cache(stack.layers, stack.kv_heads, stack.head_dim)
    hidden = stack.forward(tokens, np.arange(tokens.shape[0]), False, cache)
    kept = hidden[context.shape[0] + mu.reshape(-1, frames.shape[-1]).shape[0] + 1 :]
    out = store.linear(
        session,
        "feat_decoder.estimator.out_proj.weight",
        kept,
        "feat_decoder.estimator.out_proj.bias",
    )
    return out.T


def _mlp(session: Session, store: Store, name: str, value: float) -> np.ndarray:
    stem = f"feat_decoder.estimator.{name}"
    hidden = silu(
        store.linear(session, f"{stem}.linear_1.weight", _sinusoid(value), f"{stem}.linear_1.bias")
    )
    return store.linear(session, f"{stem}.linear_2.weight", hidden, f"{stem}.linear_2.bias")


def _sinusoid(value: float, dim: int = 1024, scale: float = 1000.0) -> np.ndarray:
    half = dim // 2
    slope = math.log(10000.0) / (half - 1)
    freq = np.exp(np.arange(half, dtype=np.float64) * -slope)
    arg = scale * float(value) * freq
    return np.concatenate((np.sin(arg), np.cos(arg))).astype(np.float32)
