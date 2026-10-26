"""Load the packed record and run eight Euler steps."""

from __future__ import annotations

import gc
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import objc

from favlo.memory import memory
from z_image_turbo.dit import velocity
from z_image_turbo.formats import TILE_M, pad_to
from z_image_turbo.hold import ImageHold, ImagePaused, ask
from z_image_turbo.metal import Gpu, reclaimable_bytes, wait_until_reclaimed
from z_image_turbo.progress import ImageProgress, LoadProgress, sample_total
from z_image_turbo.record import Plane, load, weight_bytes
from z_image_turbo.schedule import STEPS, sigmas
from z_image_turbo.scratch import MAX_EDGE, Scratch, scratch_bytes
from z_image_turbo.text import embed_caption, prompt_ids

_ONES = 1024 * 4


class Runtime:
    def __init__(
        self,
        path: Path,
        console: Any = None,
        *,
        owner: str | None = None,
        max_memory: str | int | None = None,
    ) -> None:
        self.path = path
        self.console = console
        self._closed = True
        self._owner = owner or f"z-image-turbo:{id(self)}"
        self._memory_cap = _parse_memory(max_memory)
        self.gpu = None
        self.weights = {}
        self.scratch = None
        self.ones = None
        self.source = None
        self._open()

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
        """Finish device work and drop every buffer before another pipeline allocates."""
        if self._closed:
            return
        gpu = self.gpu
        freed = gpu.held_bytes() if gpu is not None and hasattr(gpu, "held_bytes") else 0
        before = reclaimable_bytes() if freed else 0
        self._closed = True
        try:
            with objc.autorelease_pool():
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
            if freed:
                wait_until_reclaimed(before, freed)
        finally:
            self._sweep_owner()

    def _sweep_owner(self) -> None:
        """Return this opening's spans. ``Gpu.close`` sweeps the same string."""
        owner = getattr(self, "_owner", None)
        if isinstance(owner, str) and owner:
            memory.sweep(owner)

    def _room(self) -> int:
        """Bytes a new span may take. ``max_memory`` caps the ledger."""
        ceiling = memory.budget()
        cap = getattr(self, "_memory_cap", None)
        if cap is not None:
            ceiling = min(ceiling, cap)
        return max(0, ceiling - memory.live - memory.pending)

    def _reserve_scratch(self, width: int, height: int) -> None:
        """Hold scratch for this image. A short ledger keeps the requested size."""
        scratch = self.scratch
        if scratch is not None and scratch.covers(width, height):
            return
        needed = scratch_bytes(width, height)
        preferred = scratch_bytes(MAX_EDGE, MAX_EDGE)
        held = 0 if scratch is None else scratch_bytes(scratch.width, scratch.height)
        room = self._room() + held
        if room >= preferred:
            chosen = (MAX_EDGE, MAX_EDGE)
        elif room >= needed:
            chosen = (width, height)
        else:
            message = f"scratch for {width}x{height} does not fit"
            raise RuntimeError(message)
        if scratch is not None:
            scratch.release()
            self.scratch = None
        self.scratch = Scratch(self.gpu, chosen[0], chosen[1])

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
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        if getattr(self, "_closed", False):
            self._open()
        _check_size(width, height, steps)
        self._reserve_scratch(width, height)
        if done_steps and latent is None:
            message = "a resumed step needs its latent"
            raise ValueError(message)
        progress = ImageProgress(self.console, f"{width}x{height}")
        progress.open(sample_total(steps=steps, done_steps=done_steps))
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
            )
        except ImagePaused as paused:
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
            self.gpu.reclaim()
            caption_s = time.perf_counter() - caption_started
            if ask(stop) == "cancel":
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
        knots = sigmas(steps)
        for index in range(done_steps, steps):
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
                    float(knots[index]),
                    self.ones,
                    patch_h,
                    patch_w,
                    on_layer=progress.advance,
                )
                self.gpu.euler(
                    self.scratch.latent,
                    self.scratch.vel,
                    float(knots[index] - knots[index + 1]),
                    count,
                )
            self.gpu.reclaim()
            times.append(time.perf_counter() - started)
            progress.note(f"step {index + 1}/{steps} {times[-1]:.2f}s")
            reason = ask(stop)
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
        if ask(stop):
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
        reason = ask(stop)
        if reason is None:
            return
        kept = None if latent is None else np.array(latent, copy=True, dtype=np.float32)
        raise ImagePaused(
            reason,
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
    return Runtime(Path(path), console, owner=owner, max_memory=max_memory)


def _parse_memory(text: str | int | None) -> int | None:
    """``32GiB``, ``512MiB``, or a byte count. Empty leaves the ledger uncapped."""
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


def _decode_image(
    runtime: Runtime,
    height: int,
    width: int,
    on_layer: Callable[[], None] | None = None,
) -> np.ndarray:
    from z_image_turbo.vae import decode

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


def _check_size(width: int, height: int, steps: int) -> None:
    if steps != STEPS:
        message = f"this pipeline evaluates {STEPS} velocities, got {steps}"
        raise ValueError(message)
    if width % 16 or height % 16:
        message = f"width and height must be multiples of 16, got {width}x{height}"
        raise ValueError(message)
    if width > MAX_EDGE or height > MAX_EDGE or width < 16 or height < 16:
        message = f"edge must be from 16 to {MAX_EDGE}, got {width}x{height}"
        raise ValueError(message)


def _source_of(record: Path) -> Path:
    import json

    body = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    source = body.get("source")
    if isinstance(source, str) and Path(source).is_dir():
        return Path(source)
    sibling = record.parent / "Z-Image-Turbo"
    if sibling.is_dir():
        return sibling
    message = "tokenizer source is not next to the packed record"
    raise FileNotFoundError(message)
