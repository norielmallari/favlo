"""Parameters stored beside one generated file.

The record is ``<home>/<kind>/<id>.json``, next to the media bytes.
A sweep deletes that file with the media.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home

_FOLDERS = {
    "image": "images",
    "video": "videos",
    "music": "music",
    "voice": "voice",
}
_KINDS = (
    ("image_", "image"),
    ("video_", "video"),
    ("music_", "music"),
    ("voice_", "voice"),
)


def kind_of(name: str) -> str | None:
    """The media kind named by an id prefix, or ``None``."""
    if not isinstance(name, str):
        return None
    for prefix, kind in _KINDS:
        if name.startswith(prefix):
            return kind
    return None


def save_media(
    home: Path,
    kind: str,
    name: str,
    fields: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Write one record and return it."""
    if kind not in _FOLDERS or kind_of(name) != kind:
        message = "refusing media id"
        raise ValueError(message)
    _check(name)
    record: dict[str, Any] = {"name": name, "kind": kind}
    if model:
        record["model"] = model
    if "seed" in fields:
        record["seed"] = fields["seed"]
    for key in sorted(fields):
        if key in {"name", "kind", "model", "seed"}:
            continue
        record[key] = fields[key]
    folder = home / _FOLDERS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    temporary = folder / f"{name}.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def load_media(name: str, *, home: Path | None = None) -> dict[str, Any] | None:
    """The stored record, or ``None`` when the id or the file is missing."""
    kind = kind_of(name)
    if kind is None:
        return None
    try:
        _check(name)
    except ValueError:
        return None
    root = favlo_home() if home is None else home
    path = root / _FOLDERS[kind] / f"{name}.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    return record


def drop_media(home: Path, kind: str, name: str) -> None:
    """Delete the JSON record beside one media id."""
    if kind not in _FOLDERS or kind_of(name) != kind:
        return
    try:
        _check(name)
    except ValueError:
        return
    path = home / _FOLDERS[kind] / f"{name}.json"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _check(name: str) -> None:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "." in name
        or "\x00" in name
    ):
        message = "refusing media id"
        raise ValueError(message)
