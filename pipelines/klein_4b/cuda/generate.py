"""Caption, denoise, decode. Device buffers are rebuilt after a pause."""

from __future__ import annotations

from typing import Any

import numpy as np

from favlo.media_pipeline import checkpoint
from klein_4b.cuda import dit, text, vae
from klein_4b.cuda.constants import ALIGN, LATENT_CHANNELS
from klein_4b.cuda.ops import Ops
from klein_4b.cuda.pixels import to_model, to_u8
from klein_4b.cuda.prompt import encode
from klein_4b.cuda.request import Request
from klein_4b.cuda.rope import dit_table, image_ids, text_ids
from klein_4b.cuda.schedule import sigmas
from klein_4b.cuda.store import Store


def generate(
    ops: Ops,
    store: Store,
    request: Request,
    hold: Any,
    job: dict[str, Any],
    stop: Any,
) -> np.ndarray:
    grid_h = request.height // ALIGN
    grid_w = request.width // ALIGN
    if isinstance(hold, dict) and hold.get("phase") == "decode":
        _mark(job, "decode", 95)
        checkpoint(stop, _hold("decode", request.steps, hold["latent"], grid_h, grid_w, request))
        return _decode(ops, store, hold["latent"], grid_h, grid_w, request.height, request.width)

    job["stage"] = "caption"
    job["progress"] = 0
    ids, valid = encode(request.prompt, store.folder / "tokenizer" / "tokenizer.json")
    context = text.caption(ops, store, np.asarray(ids, dtype=np.int32), valid)
    _mark(job, "caption", 5)
    if isinstance(hold, dict) and hold.get("phase") == "denoise":
        latent = np.ascontiguousarray(hold["latent"], dtype=np.float16)
        start = int(hold["step"])
    else:
        latent = np.random.default_rng(request.seed).standard_normal((grid_h * grid_w, LATENT_CHANNELS)).astype(np.float16)
        start = 0
    checkpoint(stop, _hold("denoise", start, latent, grid_h, grid_w, request))

    references = [_encode_ref(ops, store, image) for image in request.images]
    store.free_prefix("vae/")
    ref_rows = sum(plane.rows for plane in references)
    text_cos, text_sin, image_cos, image_sin, joint_cos, joint_sin = _ropes(ops, request, references, grid_h, grid_w)
    table = sigmas(grid_h * grid_w, request.steps)
    _mark(job, "denoise", 10)
    for index in range(start, request.steps):
        device = ops.upload_f16(latent, latent.shape[0], LATENT_CHANNELS)
        pieces = references + [device]
        image = pieces[0] if len(pieces) == 1 else ops.concat_rows(pieces)
        if len(pieces) != 1:
            ops.free(device)
        velocity = dit.step(
            ops,
            store,
            image,
            context,
            float(table[index]),
            text_cos,
            text_sin,
            image_cos,
            image_sin,
            joint_cos,
            joint_sin,
            context.rows,
        )
        ops.free(image)
        tail = velocity if ref_rows == 0 else ops.slice_rows(velocity, ref_rows, latent.shape[0])
        if ref_rows:
            ops.free(velocity)
        current = ops.upload_f16(latent, latent.shape[0], LATENT_CHANNELS)
        updated = ops.euler(current, tail, float(table[index + 1] - table[index]))
        ops.free(current, tail)
        latent = np.ascontiguousarray(ops.download_f16(updated))
        ops.free(updated)
        done = index + 1
        _mark(job, "denoise", 10 + int(80 * done / request.steps))
        checkpoint(stop, _hold("denoise", done, latent, grid_h, grid_w, request))
    ops.free(context, text_cos, text_sin, image_cos, image_sin, joint_cos, joint_sin, *references)
    store.free_prefix("dit/")
    _mark(job, "decode", 95)
    checkpoint(stop, _hold("decode", request.steps, latent, grid_h, grid_w, request))
    return _decode(ops, store, latent, grid_h, grid_w, request.height, request.width)


def _encode_ref(ops: Ops, store: Store, image: np.ndarray):
    tokens = vae.encode(ops, store, to_model(image))
    return tokens


def _decode(ops: Ops, store: Store, latent: np.ndarray, grid_h: int, grid_w: int, height: int, width: int) -> np.ndarray:
    device = ops.upload_f16(latent, latent.shape[0], LATENT_CHANNELS)
    pixels = vae.decode(ops, store, device, grid_h, grid_w)
    ops.free(device)
    rgb = to_u8(pixels)
    if rgb.shape[0] != height or rgb.shape[1] != width:
        message = f"decoder returned {rgb.shape[0]}x{rgb.shape[1]}"
        raise RuntimeError(message)
    return rgb


def _ropes(ops: Ops, request: Request, references, grid_h: int, grid_w: int):
    del references
    ref_grids = [(image.shape[0] // ALIGN, image.shape[1] // ALIGN) for image in request.images]
    ref_grids.append((grid_h, grid_w))
    text_cos, text_sin = dit_table(text_ids(512))
    image_cos, image_sin = dit_table(image_ids(ref_grids))
    joint_cos = np.concatenate([text_cos, image_cos], axis=0)
    joint_sin = np.concatenate([text_sin, image_sin], axis=0)
    return (
        ops.upload_f32(text_cos),
        ops.upload_f32(text_sin),
        ops.upload_f32(image_cos),
        ops.upload_f32(image_sin),
        ops.upload_f32(joint_cos),
        ops.upload_f32(joint_sin),
    )


def _hold(phase: str, step: int, latent: np.ndarray, grid_h: int, grid_w: int, request: Request) -> dict[str, Any]:
    return {
        "phase": phase,
        "step": step,
        "latent": np.ascontiguousarray(latent, dtype=np.float16),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "width": request.width,
        "height": request.height,
    }


def _mark(job: dict[str, Any], stage: str, progress: int) -> None:
    job["stage"] = stage
    job["progress"] = progress
