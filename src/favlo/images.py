"""Stored images. The process entry owns this module.

Each image keeps ``<home>/images/<id>.rgb`` and the PNG encoded from it.
``~/.favlo/settings.json`` key ``image_ttl_seconds`` is how long both stay.
The default is one day. A sweep at startup and after each stored PNG deletes
an older pair.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home, image_ttl_seconds, load_settings

_DIR = "images"


def images_dir(home: Path) -> Path:
    return home / _DIR


def store_source(
    image_id: str,
    payload: bytes,
    *,
    home: Path | None = None,
) -> Path:
    """Write the uncompressed RGB. The PNG is encoded from this file."""
    root = favlo_home() if home is None else home
    path = source_path(root, image_id)
    _replace(path, payload)
    return path


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
    _replace(path, payload)
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
    """Delete a pair once its PNG, or a lone RGB file, has reached the expiry."""
    root = favlo_home() if home is None else home
    folder = images_dir(root)
    if not folder.is_dir():
        return
    ttl = image_ttl_seconds(load_settings(root))
    clock = time.time() if now is None else now
    stems = {path.stem for path in folder.glob("*.png")}
    stems.update(path.stem for path in folder.glob("*.rgb"))
    for stem in stems:
        png = folder / f"{stem}.png"
        rgb = folder / f"{stem}.rgb"
        anchor = png if png.is_file() else rgb
        try:
            modified = anchor.stat().st_mtime
        except OSError:
            continue
        if clock - modified < ttl:
            continue
        for suffix in ("png", "rgb"):
            path = folder / f"{stem}.{suffix}"
            try:
                path.unlink()
            except OSError:
                continue
        if isinstance(images, dict):
            images.pop(stem, None)


def source_path(home: Path, image_id: str) -> Path:
    """Uncompressed RGB for one image."""
    return _image_path(home, image_id, "rgb")


def image_path(home: Path, image_id: str) -> Path:
    """Served PNG for one image."""
    return _image_path(home, image_id, "png")


def _image_path(home: Path, image_id: str, suffix: str) -> Path:
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
    return images_dir(home) / f"{image_id}.{suffix}"


def _replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
