"""Standard metadata for one generated media id.

Each kind folder stores the sidecar under that folder's
``.favlo_metadata/<filename>.json``.
An image record is ``images/.favlo_metadata/<id>.png.json``. Other kinds use
``<id>.mp4.json`` or ``<id>.wav.json``. Generation parameters live under
``generation``. A sweep moves that sidecar to Trash with the media.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home
from favlo.media_metadata import (
    from_generation,
    read_sidecar,
    remove_sidecar,
    write_sidecar,
)

_FOLDERS = {
    "image": "images",
    "video": "videos",
    "music": "audio",
    "voice": "audio",
}
_FILE = {
    "image": "png",
    "video": "mp4",
    "music": "wav",
    "voice": "wav",
}
_KINDS = (
    ("image_", "image"),
    ("video_", "video"),
    ("music_", "music"),
    ("voice_", "voice"),
)
_META_KIND = {
    "image": "image",
    "video": "video",
    "music": "audio",
    "voice": "audio",
}


def kind_of(name: str) -> str | None:
    """The media kind named by an id prefix, or ``None``."""
    if not isinstance(name, str):
        return None
    for prefix, kind in _KINDS:
        if name.startswith(prefix):
            return kind
    return None


def media_relative(kind: str, name: str) -> str:
    """Filename used as the sidecar relative path inside the kind folder."""
    suffix = _FILE[kind]
    return f"{name}.{suffix}"


def save_media(
    home: Path,
    kind: str,
    name: str,
    fields: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Write one standardized sidecar and return it."""
    if kind not in _FOLDERS or kind_of(name) != kind:
        message = "refusing media id"
        raise ValueError(message)
    _check(name)
    generation: dict[str, Any] = {}
    if model:
        generation["model"] = model
    if "seed" in fields:
        generation["seed"] = fields["seed"]
    for key in sorted(fields):
        if key in {"name", "kind", "model", "seed"}:
            continue
        generation[key] = fields[key]
    relative = media_relative(kind, name)
    width = height = None
    size = generation.get("size")
    if isinstance(size, str) and "x" in size.lower():
        left, _, right = size.lower().partition("x")
        try:
            width = int(left)
            height = int(right)
        except ValueError:
            width = height = None
    duration = generation.get("seconds")
    if not isinstance(duration, (int, float)):
        duration = None
    record = from_generation(
        filename=relative,
        kind=_META_KIND[kind],
        generation=generation,
        relative=relative,
        width=width,
        height=height,
        duration_seconds=float(duration) if duration is not None else None,
    )
    folder = home / _FOLDERS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    write_sidecar(folder, relative, record)
    _drop_legacy(folder, kind, name)
    return record


def load_media(name: str, *, home: Path | None = None) -> dict[str, Any] | None:
    """The standardized sidecar, or ``None`` when the id or the file is missing."""
    kind = kind_of(name)
    if kind is None:
        return None
    try:
        _check(name)
    except ValueError:
        return None
    root = favlo_home() if home is None else home
    folder = root / _FOLDERS[kind]
    relative = media_relative(kind, name)
    record = read_sidecar(folder, relative)
    if record is not None:
        return record
    return _load_legacy(folder, kind, name)


def drop_media(home: Path, kind: str, name: str) -> None:
    """Move the sidecar beside one media id to Trash."""
    if kind not in _FOLDERS or kind_of(name) != kind:
        return
    try:
        _check(name)
    except ValueError:
        return
    folder = home / _FOLDERS[kind]
    remove_sidecar(folder, media_relative(kind, name))
    _drop_legacy(folder, kind, name)


def _load_legacy(folder: Path, kind: str, name: str) -> dict[str, Any] | None:
    """Convert an older flat job JSON into the standardized sidecar shape."""
    for path in _legacy_paths(folder, kind, name):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(loaded, dict):
            continue
        generation: dict[str, Any] = {}
        model = loaded.get("model")
        if isinstance(model, str) and model:
            generation["model"] = model
        if "seed" in loaded:
            generation["seed"] = loaded["seed"]
        for key, value in loaded.items():
            if key in {"name", "kind", "model", "seed", "version", "filename", "title", "description"}:
                continue
            generation[key] = value
        relative = media_relative(kind, name)
        return from_generation(
            filename=relative,
            kind=_META_KIND[kind],
            generation=generation,
            relative=relative,
        )
    return None


def _drop_legacy(folder: Path, kind: str, name: str) -> None:
    from favlo.trash import trash_path

    for path in _legacy_paths(folder, kind, name):
        try:
            trash_path(path)
        except OSError:
            continue


def _legacy_paths(folder: Path, kind: str, name: str) -> tuple[Path, ...]:
    """Older adjacent job-parameter paths."""
    suffix = _FILE.get(kind)
    paths = [folder / f"{name}.json"]
    if suffix is not None:
        paths.insert(0, folder / f"{name}.{suffix}.json")
    return tuple(paths)


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
