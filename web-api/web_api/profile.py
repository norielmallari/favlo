"""Host profile the web client reads for folder kinds."""

from __future__ import annotations

from typing import Any

PROFILE = "local"

FOLDER_KINDS: list[dict[str, Any]] = [
    {"kind": "local", "enabled": True},
    {"kind": "cloud", "provider": "onedrive", "enabled": False},
    {"kind": "cloud", "provider": "dropbox", "enabled": False},
]


def document() -> dict[str, Any]:
    return {"profile": PROFILE, "folder_kinds": list(FOLDER_KINDS)}


def cloud_enabled(provider: str | None) -> bool:
    if not provider:
        return False
    for item in FOLDER_KINDS:
        if item.get("kind") == "cloud" and item.get("provider") == provider:
            return bool(item.get("enabled"))
    return False
