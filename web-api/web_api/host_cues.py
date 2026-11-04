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

# Application key under message.metadata (not the reserved "ag-ui" key).
_METADATA_UI = "ui"
_METADATA_HIDDEN = "hidden"
_HOST_NAME = "favlo"


def fence_cue(body: str) -> str:
    """Wrap host instruction text in the user-role fence."""
    return f"{FENCE_OPEN}\n{body.strip()}\n{FENCE_CLOSE}"


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
        if cue_inner(text) is not None:
            return True
    if message.get("role") == "user":
        content = message.get("content")
        if isinstance(content, str) and cue_inner(content) is not None:
            # Fence alone is enough when metadata was stripped on round-trip.
            return True
    return False


def host_cue_message(body: str, *, message_id: str) -> dict[str, Any]:
    """Durable ``user`` row: fenced body, hidden from the transcript."""
    return {
        "id": message_id,
        "role": "user",
        "name": _HOST_NAME,
        "content": fence_cue(body),
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
