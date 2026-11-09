"""Shared media result. MCP and the media CLI both return this object.

The handler has already written the file and the sidecar. This module loads
those and returns the metadata plus one absolute ``file://`` URI. It does
not read the bytes into the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from favlo.local_uri import file_uri


def pack_media(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    """One object, or a list when the payload stored several files."""
    found = [_one(item_id, state) for item_id in _ids(payload)]
    if not found:
        message = "the pipeline wrote no media"
        raise RuntimeError(message)
    if len(found) == 1:
        return found[0]
    return found


def _ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if isinstance(data, list):
        return [
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        ]
    item_id = payload.get("id")
    if isinstance(item_id, str) and item_id:
        return [item_id]
    return []


def _one(item_id: str, state: dict[str, Any]) -> dict[str, Any]:
    from favlo.media_meta import kind_of, load_media

    kind = kind_of(item_id)
    if kind is None:
        message = "unknown media"
        raise RuntimeError(message)
    path = _stored(state, kind, item_id)
    if path is None:
        message = "stored media is missing"
        raise RuntimeError(message)
    record = load_media(item_id, home=_home(state, kind))
    return {"metadata": record or {}, "mediaUri": file_uri(path)}


def _home(state: dict[str, Any], kind: str) -> Path | None:
    home = state.get(f"{kind}_home")
    if isinstance(home, Path):
        return home
    return None


def _stored(state: dict[str, Any], kind: str, item_id: str) -> Path | None:
    from favlo.images import image_path
    from favlo.logs import favlo_home
    from favlo.music import music_dir
    from favlo.videos import video_path
    from favlo.voice import voice_dir

    home = _home(state, kind)
    if home is None:
        home = favlo_home()
    if kind == "image":
        path = image_path(home, item_id)
    elif kind == "video":
        path = video_path(home, item_id)
    else:
        folder = music_dir(home) if kind == "music" else voice_dir(home)
        for suffix in ("wav", "mp3", "flac"):
            candidate = folder / f"{item_id}.{suffix}"
            if candidate.is_file():
                return candidate
        return None
    if path.is_file():
        return path
    return None
