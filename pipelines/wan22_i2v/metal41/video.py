"""Uncompressed AVI and the served MP4. The caller passes RGB. The DIB stores bottom-up BGR."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np

# Shared short-form delivery: TikTok, Instagram Reels, Facebook Reels, YouTube Shorts.
# TikTok API: MP4/H.264, 23–60 fps, 360–4096 px. Practical 1080×1920 @ 30 fps ~8–10 Mbps.
# Instagram Graph API: H.264/HEVC, closed GOP, 4:2:0, 23–60 fps, 9:16 recommended, ≤25 Mbps.
# Facebook Reels API: 1080×1920 recommended, 24–60 fps, H.264, closed GOP 2–5 s, AAC 48 kHz.
# YouTube: H.264 High, progressive, closed GOP, 1080p SDR ~8 Mbps at 24–30 fps.
_SOCIAL_FPS = 30
_SOCIAL_BITRATE = {
    "low": "4M",
    "medium": "6M",
    "high": "8M",
    "xhigh": "10M",
    "max": "12M",
}


def write_avi(frames: np.ndarray, fps: int = 16) -> bytes:
    """``frames`` is uint8 ``[count, height, width, 3]``."""
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.uint8:
        message = f"AVI expects uint8 NHWC, got {frames.shape} {frames.dtype}"
        raise ValueError(message)
    count, height, width, _ = frames.shape
    if fps < 1:
        message = f"fps must be positive, got {fps}"
        raise ValueError(message)
    row = (width * 3 + 3) & ~3
    image_size = row * height
    pictures = b"".join(_frame(frames[index], row) for index in range(count))
    micro = int(1_000_000 / fps)
    avih = struct.pack(
        "<16I",
        micro,
        image_size * fps,
        0,
        0x10,
        count,
        0,
        1,
        image_size,
        width,
        height,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        count,
        image_size,
        0xFFFFFFFF,
        0,
        0,
        0,
        width,
        height,
    )
    strf = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, image_size, 0, 0, 0, 0)
    stream = _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf))
    hdrl = _list(b"hdrl", _chunk(b"avih", avih) + stream)
    movi = _list(b"movi", pictures)
    body = b"AVI " + hdrl + movi
    return b"RIFF" + struct.pack("<I", len(body)) + body


def social_frame(width: int, height: int) -> tuple[int, int]:
    """1080p canvas that keeps the AVI orientation."""
    if width < 1 or height < 1:
        message = f"AVI size must be positive, got {width}x{height}"
        raise ValueError(message)
    if height > width:
        return 1080, 1920
    if width > height:
        return 1920, 1080
    return 1080, 1080


def write_mp4(source: Path, quality: str = "high", *, social: bool = False) -> Path:
    """H.264 MP4 beside one saved AVI. The served file is the MP4.

    ``social`` pads into the matching 1080p frame, motion-interpolates to 30 fps,
    and encodes H.264 High for TikTok, Instagram Reels, Facebook Reels, and
    YouTube Shorts. Otherwise the shared CRF encode keeps the AVI size and rate.
    """
    if social:
        return _write_social_mp4(source, quality)
    from favlo.videos import encode_mp4

    return encode_mp4(source, quality)


def _write_social_mp4(source: Path, quality: str) -> Path:
    from favlo.videos import reported_quality

    _count, width, height = avi_meta(_avi_head(source))
    out_w, out_h = social_frame(width, height)
    level = reported_quality(quality)
    bitrate = _SOCIAL_BITRATE[level]
    target = source.with_suffix(".mp4")
    temporary = target.with_name(target.stem + ".tmp.mp4")
    video_filter = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:"
        f"flags=lanczos,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,"
        f"minterpolate=fps={_SOCIAL_FPS}:mi_mode=mci:mc_mode=aobmc:"
        f"me_mode=bidir:vsbmc=1"
    )
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
            "-vf",
            video_filter,
            "-r",
            str(_SOCIAL_FPS),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            bitrate,
            "-maxrate",
            bitrate,
            "-bufsize",
            "16M",
            "-g",
            str(_SOCIAL_FPS * 2),
            "-keyint_min",
            str(_SOCIAL_FPS * 2),
            "-sc_threshold",
            "0",
            "-bf",
            "2",
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


def _avi_head(path: Path) -> bytes:
    """Enough of the AVI for ``avi_meta`` without reading every frame."""
    with path.open("rb") as handle:
        head = handle.read(1 << 20)
    if len(head) < 12:
        message = "AVI signature is missing"
        raise ValueError(message)
    return head


def avi_meta(blob: bytes) -> tuple[int, int, int]:
    """Frame count, width, and height from the ``avih`` chunk."""
    if blob[:4] != b"RIFF" or blob[8:12] != b"AVI ":
        message = "AVI signature is missing"
        raise ValueError(message)
    index = blob.find(b"avih")
    if index < 0:
        message = "AVI header is missing"
        raise ValueError(message)
    payload = blob[index + 8 : index + 8 + 64]
    fields = struct.unpack_from("<10I", payload)
    count = int(fields[4])
    width = int(fields[8])
    height = int(fields[9])
    return count, width, height


def _frame(image: np.ndarray, row: int) -> bytes:
    height, width, _ = image.shape
    raw = bytearray(row * height)
    for src, dst in enumerate(range(height - 1, -1, -1)):
        line = np.ascontiguousarray(image[src, :, ::-1]).tobytes()
        raw[dst * row : dst * row + width * 3] = line
    return _chunk(b"00db", bytes(raw))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) % 2 else b""
    return tag + struct.pack("<I", len(payload)) + payload + pad


def _list(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    pad = b"\x00" if len(body) % 2 else b""
    return b"LIST" + struct.pack("<I", len(body)) + body + pad
