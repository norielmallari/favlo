"""HTTP handlers. Chat is refused. Images use the OpenAI generations body."""

from __future__ import annotations

import base64
import time
from typing import Any

from z_image_turbo.schedule import STEPS


def handle_chat(
    handler: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del runtime, body, selection, args, state
    handler._json(
        400,
        {
            "error": {
                "message": "z-image-turbo generates images at POST /v1/images/generations",
                "type": "invalid_request_error",
            }
        },
    )


def handle_image(
    handler: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    del selection, args, state
    try:
        prompt, width, height, seed = _fields(body)
        png, _digest, timing = runtime.generate(prompt, width, height, STEPS, seed)
    except (TypeError, ValueError) as exc:
        handler._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    encoded = base64.b64encode(png).decode("ascii")
    handler._json(
        200,
        {
            "created": int(time.time()),
            "data": [{"b64_json": encoded}],
            "generation_seconds": timing,
        },
    )


def _fields(body: dict[str, Any]) -> tuple[str, int, int, int]:
    rejected = ("quality", "background", "moderation", "partial_images")
    for key in rejected:
        if key in body:
            message = f"{key} is not accepted"
            raise ValueError(message)
    if body.get("response_format") not in {None, "b64_json"}:
        message = "response_format must be b64_json"
        raise ValueError(message)
    if body.get("output_format") not in {None, "png"}:
        message = "output_format must be png"
        raise ValueError(message)
    if body.get("n") not in {None, 1}:
        message = "n must be 1"
        raise ValueError(message)
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        message = "prompt is required"
        raise ValueError(message)
    width, height = _size(body.get("size"))
    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        message = "seed must be an integer"
        raise ValueError(message)
    return prompt, width, height, seed


def _size(value: Any) -> tuple[int, int]:
    if value is None:
        return 1024, 1024
    if not isinstance(value, str) or "x" not in value:
        message = "size must be WIDTHxHEIGHT"
        raise ValueError(message)
    left, _, right = value.partition("x")
    if not left.isdigit() or not right.isdigit():
        message = "size must be WIDTHxHEIGHT"
        raise ValueError(message)
    return int(left), int(right)
