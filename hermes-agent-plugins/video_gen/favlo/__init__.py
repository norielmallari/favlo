"""Favlo video generation provider.

Hermes ``video_generate`` passes ``image_url`` for image-to-video. This backend
posts that still to Favlo ``POST /v1/videos`` and saves the MP4 from the content
route. An omitted length stays 81 frames. The frame is 240p, 320p, 480p, or 720p.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    save_bytes_video,
    success_response,
)

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8129/v1"
API_MODEL = "Wan2.2-I2V-A14B-fp4"
DEFAULT_MODEL = "wan22-i2v"
POLL_SECONDS = 2.0
TOOL_TIMEOUT_SECONDS = 1800.0
DEADLINE_SECONDS = TOOL_TIMEOUT_SECONDS


class FavloVideoGenProvider(VideoGenProvider):
    """Image-to-video against a local Favlo server at localhost:8129."""

    @property
    def name(self) -> str:
        return "favlo_video_gen"

    @property
    def display_name(self) -> str:
        return "Favlo"

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Wan 2.2 I2V",
                "speed": "local",
                "strengths": "240p, 320p, 480p, and 720p image-to-video",
                "modalities": ["image"],
            }
        ]

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Wan 2.2 I2V on localhost:8129",
            "env_vars": [],
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "modalities": ["image"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "resolutions": ["240p", "320p", "480p", "720p"],
            "max_duration": 5,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "supports_seed": True,
            "qualities": ["auto", "low", "medium", "high", "xhigh", "max", "standard", "hd"],
            "supports_upscale": False,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        image_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        duration: int | None = None,
        aspect_ratio: str = "16:9",
        resolution: str = "320p",
        negative_prompt: str | None = None,
        audio: bool | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        quality = kwargs.get("quality")
        del reference_image_urls, negative_prompt, audio, kwargs
        prompt = (prompt or "").strip()
        model_id = model or DEFAULT_MODEL
        if not prompt:
            return error_response(
                error="prompt is required",
                error_type="invalid_request",
                provider=self.name,
                aspect_ratio=aspect_ratio,
            )
        if not image_url:
            return error_response(
                error="image_url is required",
                error_type="invalid_request",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        try:
            encoded = _png_b64(image_url)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_request",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        body: dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": _size(aspect_ratio, resolution),
            "input_reference": encoded,
        }
        if duration:
            body["seconds"] = str(int(duration))
        if seed is not None:
            body["seed"] = int(seed)
        if isinstance(quality, str) and quality:
            body["quality"] = quality
        try:
            job = _create_and_poll(body)
            video_id = str(job["id"])
            content = f"{BASE_URL}/videos/{video_id}/content"
            if isinstance(quality, str) and quality:
                quoted = urllib.parse.quote(quality, safe="")
                content = f"{BASE_URL}/videos/{video_id}/content?quality={quoted}"
            blob = _get_bytes(content)
            path = save_bytes_video(blob, prefix=model_id, extension="mp4")
        except Exception as exc:
            logger.debug("Favlo video generation failed", exc_info=True)
            return error_response(
                error=f"Favlo connection failed: {exc}",
                error_type="connection_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        seconds = job.get("seconds")
        try:
            reported = int(float(seconds)) if seconds else int(duration or 0)
        except (TypeError, ValueError):
            reported = int(duration or 0)
        return success_response(
            video=str(path),
            model=model_id,
            prompt=prompt,
            modality="image",
            aspect_ratio=aspect_ratio,
            duration=reported,
            provider=self.name,
            extra={"size": body["size"], "quality": job.get("quality")},
        )


def register(ctx) -> None:
    _allow_resolutions()
    _allow_quality()
    _raise_tool_timeout("video_generate")
    ctx.register_video_gen_provider(FavloVideoGenProvider())


def _raise_tool_timeout(tool_name: str) -> None:
    """Hermes stops a tool at 420s unless the registry entry asks for longer."""
    try:
        from tools.registry import registry
    except ImportError:
        return
    entry = registry.get_entry(tool_name)
    if entry is None:
        return
    current = entry.execution_timeout
    if current is None or current < TOOL_TIMEOUT_SECONDS:
        entry.execution_timeout = TOOL_TIMEOUT_SECONDS


def _allow_resolutions() -> None:
    """A deferred ``tool_call`` validates the static schema, which omits these."""
    try:
        from tools.video_generation_tool import VIDEO_GENERATE_SCHEMA
    except ImportError:
        return
    resolution = VIDEO_GENERATE_SCHEMA.get("parameters", {}).get("properties", {}).get("resolution")
    if not isinstance(resolution, dict):
        return
    enum = resolution.get("enum")
    if not isinstance(enum, list):
        return
    missing = [name for name in ("240p", "320p", "480p", "720p") if name not in enum]
    if missing:
        resolution["enum"] = missing + enum


def _allow_quality() -> None:
    """The static video tool schema has no quality field. The image request does."""
    try:
        from tools.video_generation_tool import VIDEO_GENERATE_SCHEMA
    except ImportError:
        return
    parameters = VIDEO_GENERATE_SCHEMA.get("parameters")
    if not isinstance(parameters, dict):
        return
    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict) or "quality" in properties:
        return
    properties["quality"] = {
        "type": "string",
        "enum": ["auto", "low", "medium", "high", "xhigh", "max", "standard", "hd"],
        "description": "MP4 quality. auto, standard, and hd are high.",
    }


_SIZES = {
    "240p": {"16:9": "416x240", "9:16": "240x416", "1:1": "256x256"},
    "320p": {"16:9": "576x320", "9:16": "320x576", "1:1": "512x512"},
    "480p": {"16:9": "832x480", "9:16": "480x832", "1:1": "640x640"},
    "720p": {"16:9": "1280x720", "9:16": "720x1280", "1:1": "960x960"},
}


def _size(aspect_ratio: str, resolution: str) -> str:
    table = _SIZES.get(resolution, _SIZES["320p"])
    return table.get(aspect_ratio, table["16:9"])


def _png_b64(image_url: str) -> str:
    if image_url.startswith("data:"):
        header, _, data = image_url.partition(",")
        if "base64" not in header:
            message = "input image must be a PNG"
            raise ValueError(message)
        raw = base64.b64decode(data)
    elif image_url.startswith(("http://", "https://")):
        raw = _get_bytes(image_url)
    else:
        path = image_url[7:] if image_url.startswith("file://") else image_url
        raw = Path(path).read_bytes()
    if not raw.startswith(b"\x89PNG"):
        message = "input image must be a PNG"
        raise ValueError(message)
    return base64.b64encode(raw).decode("ascii")


def _create_and_poll(body: dict[str, Any]) -> dict[str, Any]:
    status, raw = _request("POST", f"{BASE_URL}/videos", body, timeout=DEADLINE_SECONDS)
    job = json.loads(raw.decode("utf-8"))
    if status != 200 or not isinstance(job, dict) or not job.get("id"):
        message = _error_text(job) if isinstance(job, dict) else raw.decode("utf-8", "replace")
        raise RuntimeError(message or f"video create failed ({status})")
    video_id = str(job["id"])
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        current_status = job.get("status")
        if current_status == "completed":
            return job
        if current_status == "failed":
            raise RuntimeError(_error_text(job) or "video failed")
        time.sleep(POLL_SECONDS)
        status, raw = _request("GET", f"{BASE_URL}/videos/{video_id}", None, timeout=60)
        job = json.loads(raw.decode("utf-8"))
        if status != 200 or not isinstance(job, dict):
            message = raw.decode("utf-8", "replace")
            raise RuntimeError(message or f"video poll failed ({status})")
    message = f"video job {video_id} did not finish"
    raise TimeoutError(message)


def _error_text(job: dict[str, Any]) -> str:
    error = job.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return ""


def _get_bytes(url: str) -> bytes:
    status, raw = _request("GET", url, None, timeout=120)
    if status != 200:
        message = raw.decode("utf-8", "replace")[:500]
        raise RuntimeError(message or f"download failed ({status})")
    return raw


def _request(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
