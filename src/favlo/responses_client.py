"""Direct ``POST /v1/responses`` helpers for non-chat callers.

Loads an image from a path or bytes, builds a data URL, and posts a Responses
request. Used by the gallery index job and available to other modules that need
vision without MAF or the AG-UI runner.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def normalize_v1_base(base: str) -> str:
    """Accept ``http://host:port`` or ``http://host:port/v1``."""
    root = base.rstrip("/")
    if root.endswith("/v1"):
        return root
    return root + "/v1"


def image_data_url(
    source: str | Path | bytes,
    *,
    mime: str | None = None,
) -> str | None:
    """Encode a file path or raw bytes as a ``data:image/…;base64,…`` URL."""
    if isinstance(source, (str, Path)) and not isinstance(source, bytes):
        path = Path(source)
        media_type = mime or _MIME.get(path.suffix.lower())
        if media_type is None:
            return None
        try:
            raw = path.read_bytes()
        except OSError:
            return None
    else:
        raw = source
        media_type = mime or "image/png"
    if not raw:
        return None
    return f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"


def post_responses(base_url: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """POST a Responses body. Returns the JSON object or None on failure."""
    url = normalize_v1_base(base_url) + "/responses"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def output_text(payload: dict[str, Any]) -> str:
    """Collect assistant text from a Responses payload."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                            text = part.get("text")
                            if isinstance(text, str) and text:
                                chunks.append(text)
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks).strip()


def ask_with_image(
    base_url: str,
    *,
    model: str,
    prompt: str,
    image: str | Path | bytes,
    mime: str | None = None,
    detail: str = "auto",
    store: bool = False,
    reasoning_effort: str | None = None,
) -> str | None:
    """Load ``image``, POST Responses with text + ``input_image``, return output text."""
    data_url = image_data_url(image, mime=mime)
    if data_url is None:
        return None
    body: dict[str, Any] = {
        "model": model,
        "store": store,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if reasoning_effort == "none":
        body["enable_thinking"] = False
    payload = post_responses(base_url, body)
    if payload is None:
        return None
    text = output_text(payload)
    return text or None
