"""Server-owned host cues for AG-UI threads.

Injected context is a fenced ``user`` row. AG-UI message ``metadata``
(open-by-key; ``ag-ui`` reserved) marks those rows hidden from the transcript
projection. The trusted server writes the rows; clients do not.
"""

from __future__ import annotations

import re
from typing import Any

FENCE_OPEN = "<system_reminder>"
FENCE_CLOSE = "</system_reminder>"

_FENCE = re.compile(
    rf"^\s*{re.escape(FENCE_OPEN)}\s*\n?(.*?)\n?\s*{re.escape(FENCE_CLOSE)}\s*$",
    re.DOTALL,
)

_MEDIA_TAG = {
    "completed": "media_completed",
    "failed": "media_error",
    "cancelled": "media_cancelled",
}
_MEDIA_FENCE = re.compile(
    r"^(?:<(media_completed|media_cancelled|media_error)>\n[^\n]+\n</\1>\n*)+$"
)

# Application key under message.metadata (not the reserved "ag-ui" key).
_METADATA_UI = "ui"
_METADATA_HIDDEN = "hidden"
_HOST_NAME = "favlo"


def fence_cue(body: str) -> str:
    """Wrap host instruction text in the user-role fence."""
    return f"{FENCE_OPEN}\n{body.strip()}\n{FENCE_CLOSE}"


def fence_media_status(filename: str, status: str) -> str:
    """Fence one file name. The tag is the terminal status. An unknown status is empty."""
    tag = _MEDIA_TAG.get(status.strip())
    cleaned = filename.strip()
    if not tag or not cleaned:
        return ""
    return f"<{tag}>\n{cleaned}\n</{tag}>"


def is_media_fence(text: str) -> bool:
    """True when the text is one or more media fences and nothing else."""
    return bool(text) and _MEDIA_FENCE.fullmatch(text.strip()) is not None


def cue_inner(text: str) -> str | None:
    """Inner body of a host fence, or None when the text is not fenced."""
    if not text:
        return None
    match = _FENCE.match(text.strip())
    if match is None:
        return None
    return match.group(1).strip()


def is_host_cue(message: dict[str, Any]) -> bool:
    """True when this snapshot row is a server-owned, UI-hidden host cue."""
    if not isinstance(message, dict):
        return False
    meta = message.get("metadata")
    if isinstance(meta, dict):
        favlo = meta.get("favlo")
        if isinstance(favlo, dict) and favlo.get(_METADATA_UI) == _METADATA_HIDDEN:
            return True
    if str(message.get("name") or "") == _HOST_NAME and message.get("role") == "user":
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if cue_inner(text) is not None or is_media_fence(text):
            return True
    if message.get("role") == "user":
        content = message.get("content")
        if isinstance(content, str) and (
            cue_inner(content) is not None or is_media_fence(content)
        ):
            # Fence alone is enough when metadata was stripped on round-trip.
            return True
    return False


def host_cue_message(body: str, *, message_id: str) -> dict[str, Any]:
    """Durable ``user`` row: fenced body, hidden from the transcript.

    A media fence stays as written. Other cues use ``<system_reminder>``.
    """
    cleaned = body.strip()
    if is_media_fence(cleaned) or cue_inner(cleaned) is not None:
        content = cleaned
    else:
        content = fence_cue(body)
    return {
        "id": message_id,
        "role": "user",
        "name": _HOST_NAME,
        "content": content,
        "metadata": {"favlo": {_METADATA_UI: _METADATA_HIDDEN}},
    }


def host_cue_text(message: dict[str, Any]) -> str | None:
    """Plain host-cue body for history scans, or None when not a host cue."""
    if not is_host_cue(message):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    inner = cue_inner(content)
    return inner if inner is not None else content.strip()
