"""Uncompressed AVI. The caller passes RGB. The DIB stores bottom-up BGR."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


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


def write_mp4(source: Path, quality: str = "high") -> Path:
    """H.264 MP4 beside one saved AVI. The served file is the MP4."""
    from favlo.videos import encode_mp4

    return encode_mp4(source, quality)


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
