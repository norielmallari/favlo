"""Edit one still. References are encoded again after a resume."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from favlo.media_pipeline import MediaRequestError, checkpoint

from .constants import PROMPT_LENGTH, ROPE_SCALE
from .dit import velocity
from .metal import View
from .pixels import fit_input, read_image, to_model, to_u8
from .rope import dit_table, latent_ids, text_ids
from .schedule import sigmas
from .text import encode_prompt
from .vae import decode, encode


def edit(runtime: Any, request: Any, hold: Any, stop: Any, job: dict[str, Any] | None) -> np.ndarray:
    """uint8 HWC RGB. ``hold`` is the target latent and the next step, or a decode."""
    started = time.perf_counter()
    gpu = runtime.gpu()
    store = runtime.store
    gpu.bind_store(store)
    if isinstance(hold, dict) and hold.get("phase") == "decode":
        _mark(job, "decode", 95)
        pixels = _pixels(gpu, store, hold, None)
        _timed(started)
        return pixels
    images, width, height = _stills(request)
    if isinstance(hold, dict) and hold.get("phase") == "denoise":
        grid_h = int(hold["grid_h"])
        grid_w = int(hold["grid_w"])
        host_target = np.ascontiguousarray(hold["latent"], dtype=np.float16)
        start = int(hold["step"])
    else:
        grid_h = height // 16
        grid_w = width // 16
        host_target = None
        start = 0
    _mark(job, "caption", 5)
    print("klein_4b caption", flush=True)
    text = encode_prompt(gpu, store, request.prompt)
    _release(gpu, store, "text/")
    ref_views: list[View] = []
    ref_ids: list[np.ndarray] = []
    for index, image in enumerate(images):
        tokens, found_h, found_w = encode(gpu, store, to_model(image))
        ref_views.append(tokens)
        ref_ids.append(latent_ids(found_h, found_w, ROPE_SCALE + ROPE_SCALE * index))
    if images:
        _release(gpu, store, "vae/encoder", "vae/quant_conv")
    count = grid_h * grid_w
    if host_target is None:
        seed = 0 if request.seed is None else int(request.seed)
        host_target = np.random.default_rng(seed).standard_normal((count, 128)).astype(np.float16)
    target = gpu.copy_in(host_target)
    refs = None
    if ref_views:
        refs = ref_views[0] if len(ref_views) == 1 else gpu.concat_rows(ref_views)
        if len(ref_views) > 1:
            gpu.drop(*ref_views)
    schedule = sigmas(count, int(request.steps))
    target_ids = latent_ids(grid_h, grid_w, 0)
    image_ids = target_ids if refs is None else np.concatenate([*ref_ids, target_ids], axis=0)
    cos_host, sin_host = dit_table(np.concatenate([text_ids(PROMPT_LENGTH), image_ids], axis=0))
    cos = gpu.copy_in(cos_host)
    sin = gpu.copy_in(sin_host)
    print(f"klein_4b grid {grid_h}x{grid_w} steps {request.steps}", flush=True)
    _pause(stop, _denoise(start, host_target, grid_h, grid_w, width, height))
    for step in range(start, int(request.steps)):
        predicted = velocity(
            gpu,
            gpu.concat_rows([target] if refs is None else [refs, target]),
            text,
            cos,
            sin,
            float(schedule[step]),
            count,
        )
        updated = gpu.euler(target, predicted, float(schedule[step + 1]) - float(schedule[step]))
        gpu.drop(target, predicted)
        target = updated
        host_target = gpu.copy_out(target)
        done = step + 1
        _mark(job, "denoise", 10 + int(80 * done / request.steps))
        print(f"klein_4b step {done}/{request.steps}", flush=True)
        _pause(stop, _denoise(done, host_target, grid_h, grid_w, width, height))
    _release(gpu, store, "dit/")
    held = _denoise(int(request.steps), host_target, grid_h, grid_w, width, height)
    held["phase"] = "decode"
    _mark(job, "decode", 95)
    _pause(stop, held)
    pixels = _pixels(gpu, store, held, target)
    gpu.drop(text, cos, sin, target) if refs is None else gpu.drop(text, refs, cos, sin, target)
    _timed(started)
    return pixels


def _stills(request: Any) -> tuple[list[np.ndarray], int, int]:
    loaded: list[np.ndarray] = []
    for value in request.images:
        try:
            loaded.append(read_image(value))
        except ValueError as exc:
            raise MediaRequestError(str(exc), "image") from exc
    if not loaded:
        return [], int(request.width), int(request.height)
    images = [fit_input(image) for image in loaded]
    if request.width is None or request.height is None:
        width = int(images[0].shape[1])
        height = int(images[0].shape[0])
    else:
        width = int(request.width)
        height = int(request.height)
    return images, width, height


def _pixels(gpu: Any, store: Any, hold: dict[str, Any], latent: View | None) -> np.ndarray:
    owned = latent is None
    if latent is None:
        latent = gpu.copy_in(np.ascontiguousarray(hold["latent"], dtype=np.float16))
    plane = decode(gpu, store, latent, int(hold["grid_h"]), int(hold["grid_w"]))
    if owned:
        gpu.drop(latent)
    return to_u8(plane)


def _timed(started: float) -> None:
    print(f"klein_4b {time.perf_counter() - started:.2f}s", flush=True)


def _denoise(
    step: int,
    latent: np.ndarray,
    grid_h: int,
    grid_w: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "phase": "denoise",
        "step": int(step),
        "latent": np.ascontiguousarray(latent, dtype=np.float16).copy(),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "width": int(width),
        "height": int(height),
    }


def _release(gpu: Any, store: Any, *prefixes: str) -> None:
    """Wait, then drop this stage's weights and workspace."""
    gpu.sync()
    for prefix in prefixes:
        gpu.drop_prefix(prefix)
        store.close_prefix(prefix)
    gpu.release_workspace()


def _pause(stop: Any, hold: dict[str, Any]) -> None:
    checkpoint(stop, hold)


def _mark(job: dict[str, Any] | None, stage: str, progress: int) -> None:
    if isinstance(job, dict):
        job["stage"] = stage
        job["progress"] = progress
