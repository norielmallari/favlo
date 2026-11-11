"""A bare filename is a file under ``~/.favlo/<kind>/``.

A value with a scheme, a directory, or no media suffix is left alone.
``.png``, ``.jpg``, and ``.jpeg`` are ``images``. ``.mp4`` and ``.avi`` are
``videos``. ``.wav``, ``.mp3``, and ``.flac`` are ``audio``.
"""

from __future__ import annotations

from pathlib import Path

_FOLDERS = {
    ".png": ("images",),
    ".jpg": ("images",),
    ".jpeg": ("images",),
    ".webp": ("images",),
    ".mp4": ("videos",),
    ".avi": ("videos",),
    ".mov": ("videos",),
    ".webm": ("videos",),
    ".wav": ("audio",),
    ".mp3": ("audio",),
    ".flac": ("audio",),
}


class MissingMediaFile(ValueError):
    """The filename is plain and the kind folder has no such file."""


def is_plain_filename(value: str) -> bool:
    """True when ``value`` is one media filename and not a path or a URI."""
    name = value.strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or name.startswith((".", "~", "file:", "data:", "favlo:"))
    ):
        return False
    return Path(name).suffix.lower() in _FOLDERS


def kind_folders(value: str) -> tuple[str, ...]:
    """Kind directories for a plain filename. Empty when it is not one."""
    name = value.strip()
    if not is_plain_filename(name):
        return ()
    return _FOLDERS[Path(name).suffix.lower()]


def read_plain_filename(value: str, *, home: Path | None = None) -> bytes | None:
    """Bytes in the kind folder. ``None`` when ``value`` is not a plain filename."""
    path = locate_plain_filename(value, home=home)
    if path is None:
        return None
    return path.read_bytes()


def locate_plain_filename(value: str, *, home: Path | None = None) -> Path | None:
    """The file under the kind folder. ``None`` when ``value`` is not a plain filename."""
    name = value.strip()
    folders = kind_folders(name)
    if not folders:
        return None
    root = _root(home)
    for folder in folders:
        path = _inside(root / folder, name)
        if path.is_file():
            return path
    raise MissingMediaFile("stored file is missing")


def resolve_media_input(path: Path, *, home: Path | None = None) -> Path:
    """A bare filename becomes the kind-folder file. Any other path stays."""
    if path.is_absolute() or path.parent != Path("."):
        return path
    located = locate_plain_filename(path.name, home=home)
    if located is None:
        return path
    return located


def _root(home: Path | None) -> Path:
    if home is not None:
        return home
    from favlo.logs import favlo_home

    return favlo_home()


def _inside(folder: Path, name: str) -> Path:
    path = (folder / name).resolve()
    try:
        path.relative_to(folder.resolve())
    except ValueError as exc:
        message = "refusing filename"
        raise ValueError(message) from exc
    return path
