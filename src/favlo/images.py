"""Stored PNGs. The process entry owns this module.

Each finished image is ``<home>/images/<id>.png``. ``~/.favlo/settings.json``
key ``image_ttl_seconds`` is how long that file stays. The default is one day.
A sweep at startup and after each stored image deletes an older file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home, image_ttl_seconds, load_settings

_DIR = "images"


def images_dir(home: Path) -> Path:
    return home / _DIR


def store_image(
    image_id: str,
    payload: bytes,
    *,
    home: Path | None = None,
    images: dict[str, Any] | None = None,
) -> Path:
    """Write one PNG, then drop images older than the configured expiry."""
    root = favlo_home() if home is None else home
    path = image_path(root, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    sweep_images(images, home=root)
    return path


def load_image(image_id: str, *, home: Path | None = None) -> bytes | None:
    """The PNG bytes, or None when the id is unsafe or the file is gone."""
    root = favlo_home() if home is None else home
    try:
        path = image_path(root, image_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path.read_bytes()


def sweep_images(
    images: dict[str, Any] | None = None,
    *,
    home: Path | None = None,
    now: float | None = None,
) -> None:
    """Delete a PNG whose age has reached ``image_ttl_seconds``."""
    root = favlo_home() if home is None else home
    folder = images_dir(root)
    if not folder.is_dir():
        return
    ttl = image_ttl_seconds(load_settings(root))
    clock = time.time() if now is None else now
    for path in folder.glob("*.png"):
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if clock - modified < ttl:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        if isinstance(images, dict):
            images.pop(path.stem, None)


def image_path(home: Path, image_id: str) -> Path:
    if (
        not isinstance(image_id, str)
        or not image_id
        or image_id in {".", ".."}
        or "/" in image_id
        or "\\" in image_id
        or "\x00" in image_id
    ):
        message = "refusing image id"
        raise ValueError(message)
    return images_dir(home) / f"{image_id}.png"
