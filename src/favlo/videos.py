"""Stored video clips. The process entry owns this module.

Each clip keeps ``<home>/videos/<id>.avi`` and the MP4 encoded from it.
A later quality is ``<id>.<quality>.mp4``, encoded from that same AVI.
``~/.favlo/settings.json`` key ``video_ttl_seconds`` is how long they stay.
The default is one day. A sweep at startup and after each stored MP4 deletes
an older clip and drops the matching job.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from favlo.logs import favlo_home, load_settings, video_ttl_seconds

_DIR = "videos"
_CRF = {"low": 32, "medium": 28, "high": 23, "xhigh": 18, "max": 14}
_ALIASES = {name: name for name in _CRF}
_ALIASES.update(auto="high", standard="high", hd="high")
_encode_lock = threading.Lock()


def videos_dir(home: Path) -> Path:
    return home / _DIR


def store_source(
    video_id: str,
    payload: bytes,
    *,
    home: Path | None = None,
) -> Path:
    """Write the uncompressed AVI. The MP4 is encoded from this file."""
    root = favlo_home() if home is None else home
    path = source_path(root, video_id)
    _replace(path, payload)
    return path


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
    _replace(path, payload)
    sweep_videos(videos, home=root)
    return path


def reported_quality(value: str) -> str:
    """Map a request quality onto ``low``, ``medium``, ``high``, ``xhigh``, or ``max``."""
    reported = _ALIASES.get(value)
    if reported is None:
        message = "quality must be auto, low, medium, high, xhigh, max, standard, or hd"
        raise ValueError(message)
    return reported


def encode_mp4(source: Path, quality: str, dest: Path | None = None) -> Path:
    """H.264 MP4 from one saved AVI. The default file replaces ``<id>.mp4``."""
    crf = _CRF.get(quality)
    if crf is None:
        message = f"quality must be a reported image level, got {quality}"
        raise ValueError(message)
    target = source.with_suffix(".mp4") if dest is None else dest
    temporary = target.with_name(target.stem + ".tmp.mp4")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        message = detail or "ffmpeg failed"
        raise RuntimeError(message)
    temporary.replace(target)
    return target


def load_video(
    video_id: str,
    *,
    home: Path | None = None,
    quality: str | None = None,
) -> bytes | None:
    """The MP4 bytes, or None when the id is unsafe or the file is gone.

    ``quality`` is a reported level. A level other than the stored file is
    encoded from the AVI the first time, then read on the next request.
    """
    root = favlo_home() if home is None else home
    try:
        if quality is None:
            path = video_path(root, video_id)
        else:
            path = _variant_path(root, video_id, quality)
        if quality is not None and not path.is_file():
            source = source_path(root, video_id)
            if not source.is_file():
                return None
            with _encode_lock:
                if not path.is_file():
                    encode_mp4(source, quality, path)
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
    """Delete a pair once its MP4, or a lone AVI, has reached the expiry."""
    root = favlo_home() if home is None else home
    folder = videos_dir(root)
    if not folder.is_dir():
        return
    ttl = video_ttl_seconds(load_settings(root))
    clock = time.time() if now is None else now
    stems = {_clip_id(path.stem) for path in folder.glob("*.mp4")}
    stems.update(_clip_id(path.stem) for path in folder.glob("*.avi"))
    for stem in stems:
        mp4 = folder / f"{stem}.mp4"
        avi = folder / f"{stem}.avi"
        variants = [folder / f"{stem}.{name}.mp4" for name in _CRF]
        existing = [path for path in (mp4, avi, *variants) if path.is_file()]
        if not existing:
            continue
        anchor = mp4 if mp4.is_file() else avi if avi.is_file() else existing[0]
        try:
            modified = anchor.stat().st_mtime
        except OSError:
            continue
        if clock - modified < ttl:
            continue
        for path in existing:
            try:
                path.unlink()
            except OSError:
                continue
        if isinstance(videos, dict):
            videos.pop(stem, None)


def source_path(home: Path, video_id: str) -> Path:
    """Uncompressed AVI for one job."""
    return _clip_path(home, video_id, "avi")


def video_path(home: Path, video_id: str) -> Path:
    """Served MP4 for one job."""
    return _clip_path(home, video_id, "mp4")


def _variant_path(home: Path, video_id: str, quality: str) -> Path:
    if quality not in _CRF:
        message = f"quality must be a reported image level, got {quality}"
        raise ValueError(message)
    return _clip_path(home, video_id, f"{quality}.mp4")


def _clip_id(stem: str) -> str:
    for name in _CRF:
        suffix = f".{name}"
        if stem.endswith(suffix) and stem[: -len(suffix)]:
            return stem[: -len(suffix)]
    return stem


def _clip_path(home: Path, video_id: str, suffix: str) -> Path:
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
    return videos_dir(home) / f"{video_id}.{suffix}"


def _replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
