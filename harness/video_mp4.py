"""Encode an AVI blob to an H.264 MP4 with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def first_frame_png(path: Path) -> bytes:
    """PNG bytes of frame 0."""
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ],
        None,
    )
    if not result.stdout.startswith(b"\x89PNG"):
        message = "ffmpeg did not return a PNG frame"
        raise RuntimeError(message)
    return result.stdout


def write_mp4(avi: bytes, dest: Path) -> tuple[int, int, int]:
    """Write ``dest`` and return frame count, width, and height."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        avi,
    )
    return _geometry(dest)


def save_mp4(blob: bytes, dest: Path) -> tuple[int, int, int]:
    """Store an MP4 and return frame count, width, and height."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return _geometry(dest)


def _geometry(path: Path) -> tuple[int, int, int]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        None,
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        message = "ffprobe found no video stream"
        raise RuntimeError(message)
    stream = streams[0]
    return (
        int(stream["nb_read_frames"]),
        int(stream["width"]),
        int(stream["height"]),
    )


def _run(command: list[str], stdin: bytes | None) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, input=stdin, capture_output=True, check=False)
    except FileNotFoundError as exc:
        message = f"{command[0]} is missing"
        raise RuntimeError(message) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        message = detail or f"{command[0]} failed"
        raise RuntimeError(message)
    return result
