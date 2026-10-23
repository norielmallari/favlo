"""Stored video clips. The process entry owns this module.

Each finished MP4 is ``<home>/videos/<id>.mp4``. ``~/.favlo/settings.json``
key ``video_ttl_seconds`` is how long that file stays. The default is one day.
A sweep at startup and after each stored clip deletes an older file and drops
the matching job.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home, load_settings, video_ttl_seconds

_DIR = "videos"


def videos_dir(home: Path) -> Path:
    return home / _DIR


def store_video(
    video_id: str,
    payload: bytes,
    *,
    home: Path | None = None,
    videos: dict[str, Any] | None = None,
) -> Path:
    """Write one MP4, then drop clips older than the configured expiry."""
    root = favlo_home() if home is None else home
    path = video_path(root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    sweep_videos(videos, home=root)
    return path


def load_video(video_id: str, *, home: Path | None = None) -> bytes | None:
    """The MP4 bytes, or None when the id is unsafe or the file is gone."""
    root = favlo_home() if home is None else home
    try:
        path = video_path(root, video_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path.read_bytes()


def sweep_videos(
    videos: dict[str, Any] | None = None,
    *,
    home: Path | None = None,
    now: float | None = None,
) -> None:
    """Delete an MP4 whose age has reached ``video_ttl_seconds``."""
    root = favlo_home() if home is None else home
    folder = videos_dir(root)
    if not folder.is_dir():
        return
    ttl = video_ttl_seconds(load_settings(root))
    clock = time.time() if now is None else now
    for path in folder.glob("*.mp4"):
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
        if isinstance(videos, dict):
            videos.pop(path.stem, None)


def video_path(home: Path, video_id: str) -> Path:
    if (
        not isinstance(video_id, str)
        or not video_id
        or video_id in {".", ".."}
        or "/" in video_id
        or "\\" in video_id
        or "\x00" in video_id
    ):
        message = "refusing video id"
        raise ValueError(message)
    return videos_dir(home) / f"{video_id}.mp4"
