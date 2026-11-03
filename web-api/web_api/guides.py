"""Workspace guides: which catalog elements a type starts with."""

from __future__ import annotations

from typing import Any

GUIDES: dict[str, dict[str, Any]] = {
    "gallery": {
        "id": "gallery",
        "version": 1,
        "catalog": ["preview", "filmstrip"],
        "tool_sources": ["gallery", "workspace"],
        "setup": "local_folder",
        "skill": "skills-workspace/gallery/manage-gallery",
    },
}


def get(guide_id: str) -> dict[str, Any] | None:
    return GUIDES.get(guide_id)


def initial_state(guide_id: str) -> dict[str, Any]:
    from web_api.placement import initial_for

    return initial_for(guide_id)
