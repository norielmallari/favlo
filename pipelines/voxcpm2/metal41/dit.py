"""Local encoder, local diffusion transformer, and the Euler solver."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from voxcpm2.metal41.block import silu
from voxcpm2.metal41.fsq import quantize_bound
from voxcpm2.metal41.metal import Plane, Session
from voxcpm2.metal41.stack import Cache, Stack
from voxcpm2.metal41.store import Store


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
    try:
        hidden = stack.forward(sequence, np.arange(sequence.shape[0]), False, cache)
    finally:
        cache.release()
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
    on_step: Any = None,
    before_step: Any = None,
) -> np.ndarray:
    """One latent patch, ``[channels, frames]``, from the swayed Euler solver.

    ``before_step`` runs before the estimates so the sample row can name the
    step. ``on_step`` runs after that step's command buffers have finished.
    """
    span = diffusion_schedule(steps)
    counted = len(span) - 1
    time = float(span[0])
    dt = float(span[0] - span[1])
    skip = zero_init_steps(span)
    latent = rng.standard_normal((int(cond.shape[0]), int(cond.shape[1]))).astype(np.float32)
    blank = np.zeros_like(mu)
    for step in range(1, len(span)):
        if before_step is not None:
            before_step(step, counted)
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
        if on_step is not None:
            on_step()
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
    """One diffusion estimate. The projections and the stack share one command buffer."""
    frames_in = np.ascontiguousarray(latent.T, dtype=np.float32)
    cond_in = np.ascontiguousarray(cond.T, dtype=np.float32)
    mu_in = np.ascontiguousarray(mu, dtype=np.float32).reshape(-1, stack.hidden)
    n_mu = int(mu_in.shape[0])
    n_cond = int(cond_in.shape[0])
    n_frames = int(frames_in.shape[0])
    seq = n_mu + 1 + n_cond + n_frames
    delta = float(dt) if mean_mode else 0.0
    cache = Cache(stack.layers, stack.kv_heads, stack.head_dim, capacity=seq)
    stack.prepare(seq, seq)
    lat = session.load(frames_in)
    cond_plane = session.load(cond_in)
    time_plane = session.load(_sinusoid(float(time)))
    delta_plane = session.load(_sinusoid(delta))
    mu_plane = session.load(mu_in)
    frames = session.plane(n_frames, stack.hidden)
    context = session.plane(n_cond, stack.hidden)
    token = session.plane(1, stack.hidden)
    hid = session.plane(1, stack.hidden)
    act = session.plane(1, stack.hidden)
    tokens = session.plane(seq, stack.hidden)
    kept = session.plane(n_frames, stack.hidden)
    out = session.plane(n_frames, int(frames_in.shape[1]))
    owned = [
        lat,
        cond_plane,
        time_plane,
        delta_plane,
        mu_plane,
        frames,
        context,
        token,
        hid,
        act,
        tokens,
        kept,
        out,
    ]
    handles: list[object] = []
    try:
        _project(session, store, "feat_decoder.estimator.in_proj", lat, frames)
        _project(session, store, "feat_decoder.estimator.cond_proj", cond_plane, context)
        _time_mlp(session, store, "time_mlp", time_plane, token, hid, act, add=False)
        _time_mlp(session, store, "delta_time_mlp", delta_plane, token, hid, act, add=True)
        session.copy(mu_plane, tokens, n_mu, 0, 0)
        session.copy(token, tokens, 1, 0, n_mu)
        session.copy(context, tokens, n_cond, 0, n_mu + 1)
        session.copy(frames, tokens, n_frames, 0, n_mu + 1 + n_cond)
        hidden = stack.encode(tokens, np.arange(seq), False, cache, owned, handles)
        session.copy(hidden, kept, n_frames, n_mu + 1 + n_cond, 0)
        _project(session, store, "feat_decoder.estimator.out_proj", kept, out)
        result = session.read(out)
    finally:
        session.release_all(owned, handles)
        cache.release()
    return result.T


def _project(session: Session, store: Store, name: str, src: Plane, dst: Plane) -> None:
    session.project(
        src,
        store.weight(session, f"{name}.weight"),
        store.bias_handle(session, f"{name}.bias"),
        dst,
    )


def _time_mlp(
    session: Session,
    store: Store,
    name: str,
    src: Plane,
    dst: Plane,
    hid: Plane,
    act: Plane,
    *,
    add: bool,
) -> None:
    stem = f"feat_decoder.estimator.{name}"
    session.project(
        src,
        store.weight(session, f"{stem}.linear_1.weight"),
        store.bias_handle(session, f"{stem}.linear_1.bias"),
        hid,
    )
    session.silu(hid, act)
    session.project(
        act,
        store.weight(session, f"{stem}.linear_2.weight"),
        store.bias_handle(session, f"{stem}.linear_2.bias"),
        hid if add else dst,
    )
    if add:
        session.add(dst, hid, dst)


def _sinusoid(value: float, dim: int = 1024, scale: float = 1000.0) -> np.ndarray:
    half = dim // 2
    slope = math.log(10000.0) / (half - 1)
    freq = np.exp(np.arange(half, dtype=np.float64) * -slope)
    arg = scale * float(value) * freq
    return np.concatenate((np.sin(arg), np.cos(arg))).astype(np.float32)
