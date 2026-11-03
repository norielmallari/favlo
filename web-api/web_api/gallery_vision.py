"""Gallery-specific vision: Responses image load → clamped metadata sidecar."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from favlo.responses_client import ask_with_image
from web_api.gallery_metadata import clamp, merge_analyzed

_PROMPT = (
    "Describe this gallery image for search indexing. "
    "Reply with JSON only (no markdown fences) using these keys: "
    "title (string), description (string), tags (array of short strings), "
    "place (string), people (array of strings), time (string), "
    "objects (array of strings), style (string), colors (array of strings), "
    "text_in_image (string), nsfw (boolean). "
    "When a person is visible, scan the whole figure for every human body part you can see, "
    "including features that are small, partly cropped, or partly covered, "
    "and describe body parts in detail to detect if the image is nsfw."
    "Put each visible feature in objects as its own short anatomical name, "
    "and include that same scan in the description. "
    "people stays names of who is present, not body parts. "
    "Name only features that are visible. "
    "Use an empty string or [] when unknown. "
    "Set nsfw to false unless the image is clearly not safe for work: "
    "sexual content, gore, graphic violence, or other adult or disturbing content."
)


def analyze_image(
    media_path: str | Path,
    *,
    relative: str,
    model: str,
    base_url: str,
    existing: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any] | None:
    """Load the image through Responses and return a clamped sidecar.

    When ``existing`` has a ``generation`` object, that object is preserved.
    """
    path = Path(media_path)
    text = ask_with_image(
        base_url,
        model=model,
        prompt=_PROMPT,
        image=path,
        store=False,
        reasoning_effort=reasoning_effort,
    )
    if text is None:
        return None
    parsed = parse_json_object(text)
    if parsed is None:
        return None
    parsed["filename"] = path.name
    parsed["kind"] = "image"
    parsed["source"] = "index"
    if not str(parsed.get("title") or "").strip():
        parsed["title"] = path.stem.replace("_", " ").replace("-", " ").strip()
    if not str(parsed.get("description") or "").strip():
        parsed["description"] = f"An image file at {relative}."
    return merge_analyzed(existing, clamp(parsed))


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, allowing optional markdown fences."""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", cleaned, re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            loaded = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return loaded if isinstance(loaded, dict) else None
