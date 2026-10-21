"""Load the packed record and run eight Euler steps."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import objc

from z_image_turbo.dit import velocity
from z_image_turbo.formats import TILE_M, pad_to
from z_image_turbo.metal import Gpu
from z_image_turbo.png import encode_png
from z_image_turbo.record import Plane, load
from z_image_turbo.schedule import STEPS, sigmas
from z_image_turbo.scratch import MAX_EDGE, Scratch
from z_image_turbo.text import embed_caption, prompt_ids


class Runtime:
    def __init__(self, path: Path, console: Any = None) -> None:
        self.path = path
        self.console = console
        self._closed = False
        self.gpu = Gpu()
        # The process has no run loop. Drain the pool so only this object keeps the buffers.
        with objc.autorelease_pool():
            self.weights: dict[str, Plane] = load(path, self.gpu)
            self.scratch = Scratch(self.gpu)
            self.ones = self.gpu.keep(self.gpu.upload(np.ones(1024, dtype=np.float32)))
            self.source = _source_of(path)
            self.gpu.end()

    def close(self) -> None:
        """Finish device work and drop every buffer before another pipeline allocates."""
        if self._closed:
            return
        self._closed = True
        with objc.autorelease_pool():
            self.gpu.close()
            for plane in self.weights.values():
                plane.release()
            self.weights.clear()
            scratch = self.scratch
            self.scratch = None
            if scratch is not None:
                scratch.release()
            self.ones = None

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
    ) -> tuple[bytes, str, dict[str, Any]]:
        _check_size(width, height, steps)
        wall = time.perf_counter()
        caption_started = time.perf_counter()
        with objc.autorelease_pool():
            ids = prompt_ids(prompt, self.source)
            _caption, length = embed_caption(self.gpu, self.scratch, self.weights, ids)
            rows = pad_to(length, TILE_M)
            caption = self.gpu.read_f32(_caption, rows * 3840).reshape(rows, 3840)[:length].copy()
        self.gpu.reclaim()
        caption_s = time.perf_counter() - caption_started
        latent_h = height // 8
        latent_w = width // 8
        patch_h = latent_h // 2
        patch_w = latent_w // 2
        noise = np.random.default_rng(seed).standard_normal((16, latent_h, latent_w))
        latent = noise.astype(np.float32)
        knots = sigmas(steps)
        times: list[float] = []
        for index in range(steps):
            started = time.perf_counter()
            with objc.autorelease_pool():
                tokens = _patchify(latent)
                pred = velocity(
                    self.gpu,
                    self.scratch,
                    self.weights,
                    tokens,
                    caption,
                    float(knots[index]),
                    self.ones,
                    patch_h,
                    patch_w,
                )
                latent = latent + float(knots[index] - knots[index + 1]) * _unpatchify(
                    pred, latent_h, latent_w
                )
            self.gpu.reclaim()
            times.append(time.perf_counter() - started)
            print(f"step {index + 1}/{steps} {times[-1]:.2f}s", flush=True)
        if not np.isfinite(latent).all():
            raise RuntimeError("latent is not finite")
        digest = hashlib.sha256(np.ascontiguousarray(latent).tobytes()).hexdigest()
        vae_started = time.perf_counter()
        with objc.autorelease_pool():
            pixels = _decode_image(self, latent)
        self.gpu.reclaim()
        vae_s = time.perf_counter() - vae_started
        total_s = time.perf_counter() - wall
        print(
            f"generate {total_s:.2f}s caption {caption_s:.2f}s vae {vae_s:.2f}s",
            flush=True,
        )
        timing = {
            "total": total_s,
            "caption": caption_s,
            "vae": vae_s,
            "steps": times,
        }
        return encode_png(pixels), digest, timing


def open_runtime(
    path: Path,
    *,
    kv_cache: str = "e4m3",
    console: Any = None,
    resident_percent: float | None = None,
    max_memory: str | int | None = None,
) -> Runtime:
    del kv_cache, resident_percent, max_memory
    return Runtime(Path(path), console)


def _decode_image(runtime: Runtime, latent: np.ndarray) -> np.ndarray:
    from z_image_turbo.vae import decode

    image = decode(runtime.gpu, runtime.weights, latent)
    if not np.isfinite(image).all():
        raise RuntimeError("decoded image is not finite")
    clipped = np.clip(image, -1.0, 1.0)
    bytes_ = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    return np.transpose(bytes_, (1, 2, 0))


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
