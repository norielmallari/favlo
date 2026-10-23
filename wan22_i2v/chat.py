"""HTTP handlers. Chat is refused. Video is one MP4 job."""

from __future__ import annotations

import base64
import secrets
import time
from typing import Any, NamedTuple

import numpy as np

from wan22_i2v.png import decode_png, resize_rgb
from wan22_i2v.progress import ClipProgress, clip_label
from wan22_i2v.schedule import (
    FIVE_SEC_FRAMES,
    FPS,
    P320_HEIGHT,
    P320_WIDTH,
    require_geometry,
)
from wan22_i2v.video import mp4_bytes


class VideoRequestError(ValueError):
    def __init__(self, message: str, param: str | None) -> None:
        super().__init__(message)
        self.param = param


class VideoFields(NamedTuple):
    prompt: str
    image: np.ndarray
    frames: int
    width: int
    height: int
    seed: int


def handle_chat(
    handler: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    _reject(handler, "wan22-i2v generates video at POST /v1/videos", None)


def handle_video(
    handler: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del args
    try:
        fields = video_fields(body)
    except VideoRequestError as exc:
        _reject(handler, str(exc), exc.param)
        return
    job = _job(_model_id(body, selection), fields)
    videos = state.setdefault("videos", {})
    if not isinstance(videos, dict):
        videos = {}
        state["videos"] = videos
    videos[job["id"]] = job
    handler._json(200, job)
    progress = ClipProgress(
        job=job,
        console=getattr(runtime, "console", None),
        clip=clip_label(fields.width, fields.height, fields.frames, FPS),
    )
    try:
        avi, _timing = runtime.generate(
            fields.prompt,
            fields.image,
            fields.frames,
            fields.width,
            fields.height,
            fields.seed,
            progress,
        )
        progress.stage("mp4")
        mp4 = mp4_bytes(avi)
    except Exception as exc:
        progress.close(progress.summary(failed=True))
        job["status"] = "failed"
        job["error"] = {"code": type(exc).__name__, "message": str(exc)}
        return
    blobs = state.setdefault("video_mp4", {})
    if not isinstance(blobs, dict):
        blobs = {}
        state["video_mp4"] = blobs
    blobs[job["id"]] = mp4
    job["completed_at"] = int(time.time())
    progress.complete()
    job["status"] = "completed"
    progress.close(progress.summary())


def video_fields(body: dict[str, Any]) -> VideoFields:
    """One clip request. Omitted size is 320p. Omitted length is 81 frames."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VideoRequestError("prompt is required", "prompt")
    try:
        image = decode_png(_image_bytes(body))
    except VideoRequestError:
        raise
    except (ValueError, TypeError) as exc:
        raise VideoRequestError("input_reference must be a PNG", "input_reference") from exc
    width, height = _size(body.get("size"))
    frames = _frames(body)
    seed = body.get("seed", 0)
    if isinstance(seed, str) and seed.lstrip("-").isdigit():
        seed = int(seed)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise VideoRequestError("seed must be an integer", "seed")
    try:
        require_geometry(frames, height, width)
    except ValueError as exc:
        raise VideoRequestError(str(exc), "frames") from exc
    resized = resize_rgb(image, height, width)
    return VideoFields(prompt.strip(), resized, frames, width, height, seed)


def _job(model: str, fields: VideoFields) -> dict[str, Any]:
    return {
        "id": "video_" + secrets.token_hex(12),
        "object": "video",
        "created_at": int(time.time()),
        "completed_at": None,
        "status": "in_progress",
        "model": model,
        "progress": 0,
        "stage": None,
        "prompt": fields.prompt,
        "seconds": _seconds_label(fields.frames),
        "size": f"{fields.width}x{fields.height}",
        "frames": fields.frames,
        "eta": None,
        "error": None,
    }


def _model_id(body: dict[str, Any], selection: Any) -> str:
    if selection is not None:
        requested = getattr(selection, "requested_id", "")
        if isinstance(requested, str) and requested:
            return requested
    model = body.get("model")
    if isinstance(model, str):
        return model
    return ""


def _image_bytes(body: dict[str, Any]) -> bytes:
    ref = body.get("input_reference", body.get("image"))
    if isinstance(ref, dict):
        url = ref.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        ref = url
    if not isinstance(ref, str) or not ref:
        raise VideoRequestError("input_reference is required", "input_reference")
    if ref.startswith("data:"):
        _header, _sep, ref = ref.partition(",")
    try:
        return base64.b64decode(ref)
    except (ValueError, TypeError) as exc:
        raise VideoRequestError("input_reference must be a PNG", "input_reference") from exc


def _frames(body: dict[str, Any]) -> int:
    if "frames" in body and body.get("frames") is not None:
        return _whole(body.get("frames"), "frames")
    if "seconds" in body and body.get("seconds") is not None:
        seconds = body.get("seconds")
        if isinstance(seconds, str):
            try:
                seconds = float(seconds)
            except ValueError as exc:
                raise VideoRequestError("seconds must be a number", "seconds") from exc
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            raise VideoRequestError("seconds must be a number", "seconds")
        counted = int(round(float(seconds) * FPS))
        return 1 + 4 * int(round((counted - 1) / 4))
    return FIVE_SEC_FRAMES


def _whole(value: Any, name: str) -> int:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VideoRequestError(f"{name} must be an integer", name)
    return value


def _seconds_label(frames: int) -> str:
    text = f"{frames / FPS:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _size(value: Any) -> tuple[int, int]:
    if value is None:
        return P320_WIDTH, P320_HEIGHT
    if not isinstance(value, str) or "x" not in value:
        raise VideoRequestError("size must be WIDTHxHEIGHT", "size")
    left, _, right = value.partition("x")
    if not left.isdigit() or not right.isdigit():
        raise VideoRequestError("size must be WIDTHxHEIGHT", "size")
    return int(left), int(right)


def _reject(handler: Any, message: str, param: str | None) -> None:
    handler._json(
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
