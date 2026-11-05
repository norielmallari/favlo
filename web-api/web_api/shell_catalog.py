"""Shell and catalog rows seeded into a new database.

The HTTP list reads the tables. These values are the seed, matching the gallery
catalog the placement checker still uses.
"""

from __future__ import annotations

import json
from typing import Any

from web_api.catalog import ELEMENTS

_GALLERY_HINT = "Favlo indexes images, mp3, and mp4 in that folder for the gallery filmstrip."
_CODING_HINT = "Generated files are written at the root of this folder."

# Coding stays in the seed so an existing workspace still opens. It is not offered.
OFFERED_SHELLS = frozenset({"gallery"})


def shells() -> list[dict[str, Any]]:
    return [
        {
            "id": "gallery",
            "label": "Gallery",
            "icon": "image",
            "setup": "local_folder",
            "setup_hint": _GALLERY_HINT,
            "slots": ["stage", "rail"],
            "renderer": "media",
            "version": 1,
            "sort": 0,
        },
        {
            "id": "coding",
            "label": "Coding",
            "icon": "box",
            "setup": "local_folder",
            "setup_hint": _CODING_HINT,
            "slots": [],
            "renderer": "chat",
            "version": 1,
            "sort": 1,
        },
    ]


def catalog_elements() -> list[dict[str, Any]]:
    rows = []
    for name, element in ELEMENTS.items():
        rows.append(
            {
                "name": name,
                "version": int(element["version"]),
                "slots": list(element["slots"]),
                "max_count": int(element["max"]),
                "props_schema": element["props"],
                "gestures": list(element["gestures"]),
            }
        )
    return rows


def shell_elements() -> list[dict[str, Any]]:
    return [
        {
            "shell_id": "gallery",
            "element_name": "preview",
            "slot": "stage",
            "sort": 0,
            "initial_props": {"fit": "contain"},
        },
        {
            "shell_id": "gallery",
            "element_name": "filmstrip",
            "slot": "rail",
            "sort": 1,
            "initial_props": {"title": "", "items": []},
        },
    ]


def dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
