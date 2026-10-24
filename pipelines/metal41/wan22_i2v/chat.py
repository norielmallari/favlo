"""Chat is refused. Video is one MP4 job. The listener turns the reply into HTTP."""

from __future__ import annotations

import base64
import secrets
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from favlo.videos import reported_quality, store_source, sweep_videos
from wan22_i2v.hold import ClipHold, ClipPaused
from wan22_i2v.png import decode_png, resize_rgb
from wan22_i2v.progress import ClipProgress, clip_label
from wan22_i2v.schedule import (
    FIVE_SEC_FRAMES,
    FPS,
    P320_HEIGHT,
    P320_WIDTH,
    STEPS,
    require_geometry,
    require_guidance,
    require_steps,
)
from wan22_i2v.video import write_mp4


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
    quality: str
    negative_prompt: str
    guidance: tuple[float, float]
    steps: int


def handle_chat(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    _reject(reply, "wan22-i2v generates video at POST /v1/videos", None)


def handle_video(
    reply: Any,
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
        _reject(reply, str(exc), exc.param)
        return
    videos = state.setdefault("videos", {})
    if not isinstance(videos, dict):
        videos = {}
        state["videos"] = videos
    held = _held(state, fields)
    if held is None:
        job = _job(_model_id(body, selection), fields)
        videos[job["id"]] = job
        hold = None
    else:
        job = held["job"]
        hold = held["hold"]
    if getattr(reply, "payload", None) is not job:
        reply.json(200, job)
    progress = ClipProgress(
        job=job,
        console=getattr(runtime, "console", None),
        clip=clip_label(fields.width, fields.height, fields.frames, FPS),
    )
    progress.hold = hold
    progress.interrupt = lambda: _reason(state)
    home = state.get("video_home")
    root = home if isinstance(home, Path) else None
    try:
        if hold is not None and hold.avi is not None:
            avi = hold.avi
        else:
            avi, _timing = runtime.generate(
                fields.prompt,
                fields.image,
                fields.frames,
                fields.width,
                fields.height,
                fields.seed,
                progress,
                negative_prompt=fields.negative_prompt,
                guidance=fields.guidance,
                steps=fields.steps,
            )
        reason = _reason(state)
        if reason:
            raise ClipPaused(reason, ClipHold(stage="pack", avi=avi, done=progress.done))
        progress.stage("mp4")
        write_mp4(store_source(job["id"], avi, home=root), fields.quality)
    except ClipPaused as paused:
        progress.close("sample paused" if paused.reason == "preempt" else "sample cancelled")
        if paused.reason == "cancel":
            state.pop("video_hold", None)
            return
        state["video_hold"] = {"job": job, "seed": fields.seed, "hold": paused.hold}
        return
    except Exception as exc:
        state.pop("video_hold", None)
        progress.close(progress.summary(failed=True))
        job["status"] = "failed"
        job["error"] = {"code": type(exc).__name__, "message": str(exc)}
        return
    state.pop("video_hold", None)
    sweep_videos(videos, home=root)
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
    return VideoFields(
        prompt.strip(),
        resized,
        frames,
        width,
        height,
        seed,
        _quality(body),
        _negative(body),
        _guidance(body),
        _steps(body),
    )


def _reason(state: dict[str, Any]) -> str | None:
    cancel = state.get("cancel")
    if getattr(cancel, "is_set", lambda: False)():
        return "cancel"
    preempt = state.get("preempt")
    if getattr(preempt, "is_set", lambda: False)():
        return "preempt"
    return None


def _held(state: dict[str, Any], fields: VideoFields) -> dict[str, Any] | None:
    """The paused clip when this request is that same job. A mismatch is dropped."""
    held = state.get("video_hold")
    if not isinstance(held, dict):
        return None
    job = held.get("job")
    hold = held.get("hold")
    same = (
        isinstance(job, dict)
        and isinstance(hold, ClipHold)
        and job.get("status") == "in_progress"
        and held.get("seed") == fields.seed
        and job.get("prompt") == fields.prompt
        and job.get("frames") == fields.frames
        and job.get("quality") == fields.quality
        and job.get("size") == f"{fields.width}x{fields.height}"
        and job.get("negative_prompt") == fields.negative_prompt
        and job.get("guidance_scale") == [fields.guidance[0], fields.guidance[1]]
        and job.get("steps") == fields.steps
    )
    state.pop("video_hold", None)
    if not same or not isinstance(job, dict) or not isinstance(hold, ClipHold):
        return None
    return {"job": job, "hold": hold}


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
        "quality": fields.quality,
        "negative_prompt": fields.negative_prompt,
        "guidance_scale": [fields.guidance[0], fields.guidance[1]],
        "steps": fields.steps,
        "eta": None,
        "error": None,
    }


def _quality(body: dict[str, Any]) -> str:
    value = body.get("quality")
    if value is None:
        value = "auto"
    if not isinstance(value, str):
        raise VideoRequestError(
            "quality must be auto, low, medium, high, xhigh, max, standard, or hd",
            "quality",
        )
    try:
        return reported_quality(value)
    except ValueError as exc:
        raise VideoRequestError(str(exc), "quality") from exc


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


def _negative(body: dict[str, Any]) -> str:
    value = body.get("negative_prompt", "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise VideoRequestError("negative_prompt must be text", "negative_prompt")
    return value.strip()


def _guidance(body: dict[str, Any]) -> tuple[float, float]:
    value = body.get("guidance_scale")
    if value is None:
        return (1.0, 1.0)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            low = high = _scale(value[0])
        elif len(value) == 2:
            low, high = _scale(value[0]), _scale(value[1])
        else:
            raise VideoRequestError(
                "guidance_scale must be a number or a low, high pair",
                "guidance_scale",
            )
    else:
        low = high = _scale(value)
    try:
        return require_guidance(low, high)
    except ValueError as exc:
        raise VideoRequestError(str(exc), "guidance_scale") from exc


def _scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoRequestError(
            "guidance_scale must be a number or a low, high pair",
            "guidance_scale",
        )
    return float(value)


def _steps(body: dict[str, Any]) -> int:
    if "steps" not in body or body.get("steps") is None:
        return STEPS
    try:
        return require_steps(_whole(body.get("steps"), "steps"))
    except ValueError as exc:
        raise VideoRequestError(str(exc), "steps") from exc


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
