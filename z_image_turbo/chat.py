"""Chat is refused. Images use the OpenAI generations body.

``POST /v1/images/edits`` and ``POST /v1/images/variations`` are not routes.
This pipeline has no image input. A transparent background, a hosted URL, and a
streaming partial are not produced. JPEG and WebP are not encoded.
"""

from __future__ import annotations

import base64
import secrets
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from favlo.images import load_image, store_source, sweep_images
from z_image_turbo.hold import ImageHold, ImageJob, ImagePaused
from z_image_turbo.png import REPORTED_QUALITY, pack_rgb, write_png
from z_image_turbo.schedule import STEPS
from z_image_turbo.scratch import MAX_EDGE


class ImageRequestError(ValueError):
    def __init__(self, message: str, param: str | None) -> None:
        super().__init__(message)
        self.param = param


class ImageFields(NamedTuple):
    prompt: str
    width: int
    height: int
    seed: int
    count: int
    quality: str
    background: str
    output_format: str
    size: str
    safety_identifier: str | None


def handle_chat(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    _reject(
        reply,
        "z-image-turbo generates images at POST /v1/images/generations",
        None,
    )


def handle_image(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del selection, args
    home = state.get("image_home")
    root = home if isinstance(home, Path) else None
    try:
        fields = _fields(body)
    except ImageRequestError as exc:
        _reject(reply, str(exc), exc.param)
        return
    job = _resume(state, fields)
    if job is None:
        job = ImageJob(
            id="image_" + secrets.token_hex(12),
            created=int(time.time()),
            prompt=fields.prompt,
            seed=fields.seed,
            width=fields.width,
            height=fields.height,
            count=fields.count,
            quality=fields.quality,
        )
    try:
        while job.index < fields.count:
            if _paused(state, job):
                _keep(reply, state, job, fields, _reason(state) == "cancel")
                return
            pixels, timing = _draw(runtime, fields, job, state)
            if _reason(state) == "cancel":
                _keep(reply, state, job, fields, True)
                return
            if _reason(state) == "preempt":
                job.hold = _decoded(pixels, timing)
                _keep(reply, state, job, fields, False)
                return
            job.images.append(_store(pixels, fields.quality, root))
            job.timings.append(timing)
            job.index += 1
            job.hold = None
    except ImagePaused as paused:
        job.hold = paused.hold
        _keep(reply, state, job, fields, paused.reason == "cancel")
        return
    except (TypeError, ValueError) as exc:
        state.pop("image_hold", None)
        text = str(exc)
        param = "prompt" if text.startswith("caption length") else None
        _reject(reply, text, param)
        return
    state.pop("image_hold", None)
    reply.json(200, _finished(job, fields))


def _draw(
    runtime: Any,
    fields: ImageFields,
    job: ImageJob,
    state: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    hold = job.hold
    if hold is not None and hold.pixels is not None:
        return hold.pixels, _timing(hold)
    latent = None
    done = 0
    times = None
    spent = 0.0
    if hold is not None and hold.stage == "step" and hold.latent is not None:
        latent = hold.latent
        done = hold.step
        times = hold.times
        spent = hold.spent
    elif hold is not None:
        spent = hold.spent
    pixels, _digest, timing = runtime.generate(
        fields.prompt,
        fields.width,
        fields.height,
        STEPS,
        fields.seed + job.index,
        stop=lambda: _reason(state),
        latent=latent,
        done_steps=done,
        step_times=times,
        spent=spent,
    )
    return pixels, timing


def _store(pixels: np.ndarray, quality: str, root: Path | None) -> dict[str, str]:
    image_id = "image_" + secrets.token_hex(12)
    write_png(store_source(image_id, pack_rgb(pixels), home=root), quality)
    sweep_images(home=root)
    stored = load_image(image_id, home=root)
    if stored is None:
        raise RuntimeError("stored image is missing")
    return {"id": image_id, "b64_json": base64.b64encode(stored).decode("ascii")}


def _decoded(pixels: np.ndarray, timing: dict[str, Any]) -> ImageHold:
    steps = timing.get("steps")
    return ImageHold(
        stage="decode",
        step=STEPS,
        pixels=np.array(pixels, copy=True),
        times=list(steps) if isinstance(steps, list) else [],
        caption_s=float(timing.get("caption") or 0.0),
        vae_s=float(timing.get("vae") or 0.0),
        spent=float(timing.get("total") or 0.0),
    )


def _keep(
    reply: Any,
    state: dict[str, Any],
    job: ImageJob,
    fields: ImageFields,
    cancel: bool,
) -> None:
    reply.json(200, _working(job, fields))
    if cancel:
        state.pop("image_hold", None)
        return
    state["image_hold"] = job


def _paused(state: dict[str, Any], job: ImageJob) -> bool:
    reason = _reason(state)
    if reason == "cancel":
        return True
    if reason != "preempt":
        return False
    hold = job.hold
    return hold is None or hold.pixels is None


def _reason(state: dict[str, Any]) -> str | None:
    cancel = state.get("cancel")
    if getattr(cancel, "is_set", lambda: False)():
        return "cancel"
    preempt = state.get("preempt")
    if getattr(preempt, "is_set", lambda: False)():
        return "preempt"
    return None


def _resume(state: dict[str, Any], fields: ImageFields) -> ImageJob | None:
    held = state.pop("image_hold", None)
    if not isinstance(held, ImageJob):
        return None
    same = (
        held.prompt == fields.prompt
        and held.seed == fields.seed
        and held.width == fields.width
        and held.height == fields.height
        and held.count == fields.count
        and held.quality == fields.quality
    )
    if not same:
        return None
    return held


def _finished(job: ImageJob, fields: ImageFields) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created": job.created,
        "background": fields.background,
        "data": job.images,
        "output_format": fields.output_format,
        "quality": fields.quality,
        "size": fields.size,
        "generation_seconds": job.timings[0] if fields.count == 1 else job.timings,
    }
    if fields.safety_identifier:
        payload["safety_identifier"] = fields.safety_identifier
    return payload


def _working(job: ImageJob, fields: ImageFields) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created": job.created,
        "id": job.id,
        "status": "in_progress",
        "background": fields.background,
        "data": list(job.images),
        "output_format": fields.output_format,
        "quality": fields.quality,
        "size": fields.size,
        "progress": _progress(job),
        "stage": _stage(job),
        "generation_seconds": list(job.timings),
    }
    eta = _eta(job)
    if eta is not None:
        payload["eta"] = eta
    if fields.safety_identifier:
        payload["safety_identifier"] = fields.safety_identifier
    return payload


def _progress(job: ImageJob) -> int:
    per = STEPS + 2
    done = job.index * per
    hold = job.hold
    if hold is not None and hold.stage == "step":
        done += hold.step
    elif hold is not None and hold.stage == "decode":
        done += STEPS + 1
    total = job.count * per
    if done <= 0 or total <= 0:
        return 0
    return min(99, (100 * done) // total)


def _stage(job: ImageJob) -> str:
    if job.hold is not None:
        return job.hold.stage
    if job.index:
        return "store"
    return "caption"


def _eta(job: ImageJob) -> float | None:
    times = _step_times(job)
    if not times:
        return None
    mean = sum(times) / len(times)
    later = job.count - job.index - 1
    hold = job.hold
    if hold is None or hold.stage == "caption":
        left = (later + 1) * STEPS
    elif hold.stage == "step":
        left = (STEPS - hold.step) + later * STEPS
    else:
        left = max(later, 0) * STEPS
    if left <= 0:
        return 0.0
    return left * mean


def _step_times(job: ImageJob) -> list[float]:
    hold = job.hold
    if hold is not None and hold.times:
        return hold.times
    if not job.timings:
        return []
    last = job.timings[-1].get("steps")
    if not isinstance(last, list):
        return []
    return [float(item) for item in last]


def _timing(hold: ImageHold) -> dict[str, Any]:
    return {
        "total": hold.spent,
        "caption": hold.caption_s,
        "vae": hold.vae_s,
        "steps": list(hold.times),
    }


def _fields(body: dict[str, Any]) -> ImageFields:
    _absent_or(body, "response_format", {"b64_json"}, "response_format must be b64_json")
    output_format = _absent_or(body, "output_format", {"png"}, "output_format must be png")
    _absent_or(body, "stream", {False}, "stream must be false")
    _bounded_int(body, "partial_images", 0, 0, 0, "partial_images must be 0")
    _absent_or(body, "background", {"auto", "opaque"}, "background must be auto or opaque")
    _absent_or(body, "moderation", {"low", "auto"}, "moderation must be low or auto")
    _absent_or(body, "style", {"vivid", "natural"}, "style must be vivid or natural")
    quality = _absent_or(
        body,
        "quality",
        set(REPORTED_QUALITY),
        "quality must be auto, low, medium, high, xhigh, max, standard, or hd",
    )
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ImageRequestError("prompt is required", "prompt")
    width, height, size = _size(body.get("size"))
    return ImageFields(
        prompt=prompt,
        width=width,
        height=height,
        seed=_optional_int(body, "seed", 0, "seed must be an integer"),
        count=_bounded_int(body, "n", 1, 1, 10, "n must be from 1 to 10"),
        quality=REPORTED_QUALITY[quality or "auto"],
        background="opaque",
        output_format=output_format or "png",
        size=size,
        safety_identifier=_safety_identifier(body),
    )


def _absent_or(body: dict[str, Any], key: str, allowed: set[Any], message: str) -> Any:
    value = body.get(key)
    if value is None:
        return None
    if value not in allowed:
        raise ImageRequestError(message, key)
    return value


def _optional_int(body: dict[str, Any], key: str, default: int, message: str) -> int:
    value = body.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ImageRequestError(message, key)
    return value


def _bounded_int(
    body: dict[str, Any],
    key: str,
    default: int,
    low: int,
    high: int,
    message: str,
) -> int:
    value = _optional_int(body, key, default, message)
    if value < low or value > high:
        raise ImageRequestError(message, key)
    return value


def _safety_identifier(body: dict[str, Any]) -> str | None:
    safety = body.get("safety_identifier")
    user = body.get("user")
    if safety is not None and not isinstance(safety, str):
        raise ImageRequestError("safety_identifier must be a string", "safety_identifier")
    if user is not None and not isinstance(user, str):
        raise ImageRequestError("user must be a string", "user")
    chosen = safety if isinstance(safety, str) and safety else user
    if not isinstance(chosen, str) or not chosen:
        return None
    return chosen


def _size(value: Any) -> tuple[int, int, str]:
    if value is None or value == "auto":
        return 1024, 1024, "1024x1024"
    if not isinstance(value, str) or "x" not in value:
        raise ImageRequestError("size must be auto or WIDTHxHEIGHT", "size")
    left, _, right = value.partition("x")
    if not left.isdigit() or not right.isdigit():
        raise ImageRequestError("size must be auto or WIDTHxHEIGHT", "size")
    width, height = int(left), int(right)
    fits = 16 <= width <= MAX_EDGE and 16 <= height <= MAX_EDGE
    if not fits or width % 16 or height % 16:
        message = f"edge must be a multiple of 16 from 16 to {MAX_EDGE}, got {width}x{height}"
        raise ImageRequestError(message, "size")
    if width * 3 < height or height * 3 < width:
        message = f"aspect ratio must be between 1:3 and 3:1, got {width}x{height}"
        raise ImageRequestError(message, "size")
    return width, height, f"{width}x{height}"


def _reject(reply: Any, message: str, param: str | None) -> None:
    reply.json(
        400,
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )
