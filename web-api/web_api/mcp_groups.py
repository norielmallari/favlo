"""Favlo media tools, grouped for one enable switch each.

A registered pipeline is one switch. Listener tools stay on their own switches.
Kind switches are not read.
"""

from __future__ import annotations

import json
from typing import Any

_LISTENER: dict[str, tuple[str, str]] = {
    "media_status": ("media", "Media tasks"),
    "media_cancel": ("media", "Media tasks"),
    "remember": ("memory", "Memory"),
    "recall": ("memory", "Memory"),
    "forget": ("memory", "Memory"),
}

ORDER = ("media", "memory", "tools")
GROUP_NAMES = {group_id: name for group_id, name in _LISTENER.values()}
GROUP_NAMES["tools"] = "Tools"


def group_for(tool: str | dict[str, Any]) -> tuple[str, str]:
    """The switch for one listed tool. A pipeline id comes from the tool meta."""
    if isinstance(tool, str):
        name = tool
        pipeline = ""
    else:
        name = str(tool.get("name") or "")
        found = tool.get("pipeline")
        pipeline = found if isinstance(found, str) else ""
    if name in _LISTENER:
        return _LISTENER[name]
    if pipeline:
        return pipeline, pipeline
    if "__" in name:
        prefix = name.split("__", 1)[0]
        return prefix, prefix
    return ("tools", "Tools")


def default_switch_payload() -> str:
    """Listener switches start on. A pipeline switch stays on until it is saved off."""
    return json.dumps({group_id: True for group_id in ("media", "memory")})
