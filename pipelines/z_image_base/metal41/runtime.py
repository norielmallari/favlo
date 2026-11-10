"""Load the packed record and run the requested Euler steps."""

from __future__ import annotations

import gc
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import objc

from favlo.media_memory import begin_release, finish_release, open_gate, stamp_owner
from favlo.media_pipeline import MediaPaused, checkpoint
from favlo.memory import memory
from z_image.metal41.dit import velocity
from z_image.metal41.formats import TILE_M, pad_to
from z_image.metal41.hold import ImageHold
from z_image.metal41.metal import Gpu
from z_image.metal41.progress import ImageProgress, LoadProgress, sample_total
from z_image.metal41.record import Plane, load, weight_bytes
from z_image.metal41.scratch import DIT_HIDDEN, MAX_CAPTION, MAX_EDGE, Scratch, scratch_bytes
from z_image.metal41.text import embed_caption, prompt_ids
from z_image_base.metal41.guide import applies, normalization_limit
from z_image_base.metal41.schedule import MAX_STEPS, sigmas

_ONES = 1024 * 4


class Runtime:
    def __init__(
        self,
        path: Path,
        console: Any = None,
        *,
        owner: str | None = None,
    ) -> None:
        self.path = path
        self.console = console
        self._closed = True
        self._owner = stamp_owner("z_image_base", owner, id(self))
        self.gpu = None
        self.weights = {}
        self.scratch = None
        self._pos = None
        self._neg = None
        self._pos_bytes = 0
        self.ones = None
        self.source = None
        open_gate(weight_bytes(self.path) + _ONES, detail="packed record")
        self._open()

    def ensure_open(self, request: Any) -> None:
        """Wait for prior releases, then load when closed and reserve this image."""
        width = int(getattr(request, "width", MAX_EDGE))
        height = int(getattr(request, "height", MAX_EDGE))
        need = self.working_set_bytes(request)
        if self._closed:
            open_gate(need, detail=f"{width}x{height}")
            self._open()
        self._reserve_scratch(width, height)

    def working_set_bytes(self, request: Any) -> int:
        """Weights plus scratch for the named size."""
        width = int(getattr(request, "width", MAX_EDGE))
        height = int(getattr(request, "height", MAX_EDGE))
        guides = _guide_bytes(width, height)
        return weight_bytes(self.path) + scratch_bytes(width, height) + _ONES + guides

    def _open(self) -> None:
        """Load the record after ``close`` has dropped the previous device."""
        if not self._closed:
            return
        record_bytes = weight_bytes(self.path)
        if record_bytes > self._room():
            message = "the packed record does not fit"
            raise RuntimeError(message)
        self.gpu = Gpu(self._owner)
        try:
            # The process has no run loop. Drain the pool so only this object keeps the buffers.
            with objc.autorelease_pool():
                self.weights = self._load(self.path)
                if self._room() < _ONES:
                    message = "the packed record does not fit"
                    raise RuntimeError(message)
                self.ones = self.gpu.keep(self.gpu.upload(np.ones(1024, dtype=np.float32)))
                if self._room() >= scratch_bytes(MAX_EDGE, MAX_EDGE):
                    self.scratch = Scratch(self.gpu)
                if self.source is None:
                    self.source = _source_of(self.path)
                self.gpu.end()
            self._closed = False
        except Exception:
            self._closed = False
            self.close()
            raise

    def close(self) -> None:
        """Finish device work, sweep the owner, and wait until those pages return."""
        if self._closed:
            return
        before, held = begin_release(self._owner)
        self._closed = True
        try:
            with objc.autorelease_pool():
                gpu = self.gpu
                pos = self._pos
                neg = self._neg
                self._pos = None
                self._neg = None
                self._pos_bytes = 0
                if gpu is not None and pos is not None:
                    gpu.forget(pos)
                if gpu is not None and neg is not None:
                    gpu.forget(neg)
                if gpu is not None:
                    gpu.close()
                for plane in self.weights.values():
                    plane.release()
                self.weights.clear()
                scratch = self.scratch
                self.scratch = None
                if scratch is not None:
                    scratch.release()
                self.ones = None
                self.gpu = None
                gc.collect()
        finally:
            finish_release(self._owner, before, held)

    def _room(self) -> int:
        """Bytes a new span may take. The serve ledger applies ``--max-memory``."""
        return memory.room()

    def _reserve_scratch(self, width: int, height: int) -> None:
        """Hold scratch for this image. A larger ledger grows it; a smaller one keeps it."""
        scratch = self.scratch
        if scratch is not None and scratch.covers(MAX_EDGE, MAX_EDGE):
            return
        needed = scratch_bytes(width, height)
        preferred = scratch_bytes(MAX_EDGE, MAX_EDGE)
        held = 0 if scratch is None else scratch_bytes(scratch.width, scratch.height)
        room = self._room() + held
        if room >= preferred:
            chosen = (MAX_EDGE, MAX_EDGE)
        elif scratch is not None and scratch.covers(width, height):
            return
        elif room >= needed:
            chosen = (width, height)
        else:
            message = f"scratch for {width}x{height} does not fit"
            raise RuntimeError(message)
        if scratch is not None:
            scratch.release()
            self.scratch = None
        self.scratch = Scratch(self.gpu, chosen[0], chosen[1])

    def _ensure_guides(self, count: int) -> None:
        """Hold the positive velocity and the negative caption. Scratch stays shared."""
        vel_bytes = count * 4
        cap_bytes = self.scratch.cap_rows * DIT_HIDDEN * 4
        if self._pos is None or self._pos_bytes < vel_bytes:
            if self._pos is not None:
                self.gpu.forget(self._pos)
            self._pos = self.gpu.keep(self.gpu.empty(vel_bytes))
            self._pos_bytes = vel_bytes
        if self._neg is None:
            self._neg = self.gpu.keep(self.gpu.empty(cap_bytes))

    def _load(self, path: Path) -> dict[str, Plane]:
        loading = LoadProgress(self.console, path.name)
        try:
            counted = loading.open(weight_bytes(path))
            weights = load(path, self.gpu, on_bytes=loading.advance if counted else None)
        except Exception:
            loading.close("load failed")
            raise
        if counted:
            loading.finish(f"load {len(weights)} tensors")
        return weights

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        *,
        stop: Callable[[], str | None] | None = None,
        latent: np.ndarray | None = None,
        done_steps: int = 0,
        step_times: list[float] | None = None,
        spent: float = 0.0,
        guidance: float = 4.0,
        negative: str = "",
        shift: float = 6.0,
        cfg_truncation: float = 1.0,
        cfg_normalization: float = 0.0,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        if getattr(self, "_closed", False):
            self._open()
        _check_size(width, height, steps)
        _check_guidance(guidance, cfg_normalization)
        self._reserve_scratch(width, height)
        if done_steps and latent is None:
            message = "a resumed step needs its latent"
            raise ValueError(message)
        knots = sigmas(steps, shift)
        captions, guided = _plan(knots, done_steps, steps, guidance, cfg_truncation)
        progress = ImageProgress(self.console, f"{width}x{height}")
        progress.open(
            sample_total(steps=steps, done_steps=done_steps, captions=captions, guided=guided)
        )
        try:
            pixels, digest, timing = self._run(
                prompt,
                width,
                height,
                steps,
                seed,
                stop=stop,
                latent=latent,
                done_steps=done_steps,
                step_times=step_times,
                spent=spent,
                progress=progress,
                knots=knots,
                guidance=guidance,
                negative=negative,
                cfg_truncation=cfg_truncation,
                cfg_normalization=cfg_normalization,
            )
        except MediaPaused as paused:
            label = "sample paused" if paused.reason == "preempt" else "sample cancelled"
            progress.close(label)
            raise
        except Exception:
            progress.close(progress.summary(failed=True))
            raise
        summary = progress.summary()
        try:
            progress.note(
                "generate "
                f"{timing['total']:.2f}s caption {timing['caption']:.2f}s vae {timing['vae']:.2f}s"
            )
        finally:
            progress.close(summary)
        return pixels, digest, timing

    def _run(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        *,
        stop: Callable[[], str | None] | None,
        latent: np.ndarray | None,
        done_steps: int,
        step_times: list[float] | None,
        spent: float,
        progress: ImageProgress,
        knots: np.ndarray,
        guidance: float,
        negative: str,
        cfg_truncation: float,
        cfg_normalization: float,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        wall = time.perf_counter()
        times = list(step_times or [])
        latent_h = height // 8
        latent_w = width // 8
        patch_h = latent_h // 2
        patch_w = latent_w // 2
        count = 16 * latent_h * latent_w
        caption_s = 0.0
        length = 0
        neg_length = 0
        limit = normalization_limit(cfg_normalization)
        if done_steps < steps:
            progress.stage("caption")
            caption_started = time.perf_counter()
            with objc.autorelease_pool():
                ids = prompt_ids(prompt, self.source)
                _, length = embed_caption(
                    self.gpu,
                    self.scratch,
                    self.weights,
                    ids,
                    on_layer=progress.advance,
                )
                if guidance > 0:
                    self._ensure_guides(count)
                    nids = prompt_ids(negative, self.source)
                    _, neg_length = embed_caption(
                        self.gpu,
                        self.scratch,
                        self.weights,
                        nids,
                        on_layer=progress.advance,
                        destination=self._neg,
                    )
            self.gpu.reclaim()
            caption_s = time.perf_counter() - caption_started
            if callable(stop) and stop() == "cancel":
                self._stop(
                    stop,
                    stage="step" if done_steps else "caption",
                    step=done_steps,
                    latent=latent,
                    times=times,
                    caption_s=caption_s,
                    spent=spent,
                    wall=wall,
                )
        image_rows = pad_to(patch_h * patch_w, TILE_M)
        if latent is None:
            noise = np.random.default_rng(seed).standard_normal((16, latent_h, latent_w))
            self.gpu.store_f32(self.scratch.latent, noise.astype(np.float32))
        else:
            self.gpu.store_f32(self.scratch.latent, np.ascontiguousarray(latent, dtype=np.float32))
        for index in range(done_steps, steps):
            sigma = float(knots[index])
            progress.stage(f"step {index + 1}/{steps}")
            started = time.perf_counter()
            with objc.autorelease_pool():
                self.gpu.zero(self.scratch.tokens, image_rows * 256 * 4)
                self.gpu.patchify(
                    self.scratch.latent, self.scratch.tokens, 16, patch_h, patch_w, 256
                )
                velocity(
                    self.gpu,
                    self.scratch,
                    self.weights,
                    length,
                    sigma,
                    self.ones,
                    patch_h,
                    patch_w,
                    on_layer=progress.advance,
                )
                if applies(sigma, guidance, cfg_truncation):
                    self._ensure_guides(count)
                    self.gpu.copy_rows(
                        self.scratch.vel, self._pos, count, count, count, 0, 0, 1
                    )
                    velocity(
                        self.gpu,
                        self.scratch,
                        self.weights,
                        neg_length,
                        sigma,
                        self.ones,
                        patch_h,
                        patch_w,
                        on_layer=progress.advance,
                        caption=self._neg,
                    )
                    self.gpu.cfg_apply(self._pos, self.scratch.vel, guidance, limit, count)
                self.gpu.euler(
                    self.scratch.latent,
                    self.scratch.vel,
                    float(knots[index] - knots[index + 1]),
                    count,
                )
            self.gpu.reclaim()
            times.append(time.perf_counter() - started)
            progress.note(f"step {index + 1}/{steps} {times[-1]:.2f}s")
            reason = stop() if callable(stop) else None
            if reason not in {"cancel", "preempt"}:
                reason = None
            if reason == "cancel":
                self._stop(
                    stop,
                    stage="step",
                    step=index + 1,
                    times=times,
                    caption_s=caption_s,
                    spent=spent,
                    wall=wall,
                )
            elif reason == "preempt":
                self._stop(
                    stop,
                    stage="step",
                    step=index + 1,
                    latent=self._read_latent(count, latent_h, latent_w),
                    times=times,
                    caption_s=caption_s,
                    spent=spent,
                    wall=wall,
                )
        drawn = self._read_latent(count, latent_h, latent_w)
        digest = hashlib.sha256(np.ascontiguousarray(drawn).tobytes()).hexdigest()
        self._stop(
            stop,
            stage="step",
            step=steps,
            latent=drawn,
            times=times,
            caption_s=caption_s,
            spent=spent,
            wall=wall,
        )
        progress.stage("decode")
        vae_started = time.perf_counter()
        with objc.autorelease_pool():
            pixels = _decode_image(self, latent_h, latent_w, progress.advance)
        self.gpu.reclaim()
        vae_s = time.perf_counter() - vae_started
        if callable(stop) and stop() in {"cancel", "preempt"}:
            self._stop(
                stop,
                stage="decode",
                step=steps,
                pixels=np.array(pixels, copy=True),
                times=times,
                caption_s=caption_s,
                vae_s=vae_s,
                spent=spent,
                wall=wall,
            )
        total_s = spent + (time.perf_counter() - wall)
        timing = {
            "total": total_s,
            "caption": caption_s,
            "vae": vae_s,
            "steps": times,
        }
        return pixels, digest, timing

    def _read_latent(self, count: int, height: int, width: int) -> np.ndarray:
        drawn = self.gpu.read_f32(self.scratch.latent, count).reshape(16, height, width)
        if not np.isfinite(drawn).all():
            raise RuntimeError("latent is not finite")
        return np.array(drawn, copy=True, dtype=np.float32)

    def _stop(
        self,
        stop: Callable[[], str | None] | None,
        *,
        stage: str,
        step: int,
        times: list[float],
        caption_s: float,
        spent: float,
        wall: float,
        latent: np.ndarray | None = None,
        pixels: np.ndarray | None = None,
        vae_s: float = 0.0,
    ) -> None:
        kept = None if latent is None else np.array(latent, copy=True, dtype=np.float32)
        checkpoint(
            stop,
            ImageHold(
                stage=stage,
                step=step,
                latent=kept,
                pixels=pixels,
                times=list(times),
                caption_s=caption_s,
                vae_s=vae_s,
                spent=spent + (time.perf_counter() - wall),
            ),
        )


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
    owner: str | None = None,
) -> Runtime:
    del kv_cache, resident_percent
    memory.set_max_memory(max_memory)
    return Runtime(Path(path), console, owner=owner)


def _decode_image(
    runtime: Runtime,
    height: int,
    width: int,
    on_layer: Callable[[], None] | None = None,
) -> np.ndarray:
    from z_image.metal41.vae import decode

    return decode(
        runtime.gpu,
        runtime.weights,
        runtime.scratch.latent,
        height,
        width,
        on_layer=on_layer,
    )


def _patchify(latent: np.ndarray) -> np.ndarray:
    channels, height, width = latent.shape
    patch_h, patch_w = height // 2, width // 2
    # (C, patch_h, 2, patch_w, 2) → (patch_h, patch_w, 2, 2, C), channel fastest.
    ordered = np.transpose(latent.reshape(channels, patch_h, 2, patch_w, 2), (1, 3, 2, 4, 0))
    return np.ascontiguousarray(ordered.reshape(patch_h * patch_w, 2 * 2 * channels))


def _unpatchify(tokens: np.ndarray, height: int, width: int) -> np.ndarray:
    patch_h, patch_w = height // 2, width // 2
    ordered = tokens.reshape(patch_h, patch_w, 2, 2, 16)
    latent = np.transpose(ordered, (4, 0, 2, 1, 3))
    return np.ascontiguousarray(latent.reshape(16, height, width))


def _guide_bytes(width: int, height: int) -> int:
    """Positive velocity plus one caption buffer. Both stay off the shared scratch."""
    cap = pad_to(MAX_CAPTION, TILE_M) * DIT_HIDDEN * 4
    latent = 16 * (height // 8) * (width // 8) * 4
    return cap + latent


def _check_guidance(guidance: float, cfg_normalization: float) -> None:
    if guidance < 0:
        message = f"guidance must be non-negative, got {guidance}"
        raise ValueError(message)
    normalization_limit(cfg_normalization)


def _plan(
    knots: np.ndarray,
    done_steps: int,
    steps: int,
    guidance: float,
    truncation: float,
) -> tuple[int, int]:
    """Caption stacks, and how many remaining steps run a second velocity."""
    captions = 2 if guidance > 0 and done_steps < steps else 1
    guided = 0
    if guidance > 0:
        guided = sum(
            1
            for index in range(done_steps, steps)
            if applies(float(knots[index]), guidance, truncation)
        )
    return captions, guided


def _check_size(width: int, height: int, steps: int) -> None:
    if steps < 1 or steps > MAX_STEPS:
        message = f"steps must be from 1 to {MAX_STEPS}, got {steps}"
        raise ValueError(message)
    if width % 16 or height % 16:
        message = f"width and height must be multiples of 16, got {width}x{height}"
        raise ValueError(message)
    if width > MAX_EDGE or height > MAX_EDGE or width < 16 or height < 16:
        message = f"edge must be from 16 to {MAX_EDGE}, got {width}x{height}"
        raise ValueError(message)


def _source_of(record: Path) -> Path:
    import json

    packed = record / "tokenizer"
    if (packed / "tokenizer.json").is_file() and (packed / "tokenizer_config.json").is_file():
        return record
    body = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    source = body.get("source")
    if isinstance(source, str) and Path(source).is_dir():
        return Path(source)
    sibling = record.parent / "Z-Image"
    if sibling.is_dir():
        return sibling
    message = "tokenizer source is not next to the packed record"
    raise FileNotFoundError(message)
