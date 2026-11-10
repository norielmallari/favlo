"""PNG bytes for one RGB or RGBA image. The encoder is the stdlib deflate stream.

The samples are the uint8 decode. ``quality`` selects the zlib level, and a
lower quality uses higher compression: ``low`` 9, ``medium`` 7, ``high`` 5,
``xhigh`` 3, ``max`` 1. An RGBA image writes color type 6.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

REPORTED_QUALITY = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "auto": "high",
    "standard": "high",
    "hd": "high",
}
ZLIB_LEVEL = {"low": 9, "medium": 7, "high": 5, "xhigh": 3, "max": 1}
_RGB = b"RGB8"
_RGBA = b"RGBA"


def encode_png(image: np.ndarray, quality: str = "high") -> bytes:
    """``image`` is uint8 RGB or RGBA HWC. ``quality`` is a reported level."""
    level = ZLIB_LEVEL.get(quality)
    if level is None:
        message = f"quality must be a reported image level, got {quality}"
        raise ValueError(message)
    if image.ndim != 3 or image.shape[2] not in {3, 4} or image.dtype != np.uint8:
        message = f"PNG expects uint8 HWC RGB or RGBA, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, channels = image.shape
    color = 6 if channels == 4 else 2
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(raw, level)),
            _chunk(b"IEND", b""),
        )
    )


def pack_rgb(image: np.ndarray) -> bytes:
    """Uncompressed RGB8 or RGBA8. Width and height, then the uint8 samples."""
    if image.ndim != 3 or image.shape[2] not in {3, 4} or image.dtype != np.uint8:
        message = f"RGB expects uint8 HWC RGB or RGBA, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, channels = image.shape
    tag = _RGBA if channels == 4 else _RGB
    samples = np.ascontiguousarray(image).tobytes()
    return struct.pack(">4sII", tag, width, height) + samples


def unpack_rgb(blob: bytes) -> np.ndarray:
    """The uint8 HWC array stored by ``pack_rgb``."""
    if len(blob) < 12 or blob[:4] not in {_RGB, _RGBA}:
        message = "RGB signature is missing"
        raise ValueError(message)
    channels = 4 if blob[:4] == _RGBA else 3
    width, height = struct.unpack(">II", blob[4:12])
    samples = np.frombuffer(blob[12:], dtype=np.uint8)
    if width < 1 or height < 1 or samples.size != height * width * channels:
        message = "RGB payload does not match its header"
        raise ValueError(message)
    return samples.reshape(height, width, channels).copy()


def write_png(source: Path, quality: str = "high") -> Path:
    """PNG beside one saved RGB or RGBA file. The served file is the PNG."""
    dest = source.with_suffix(".png")
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_bytes(encode_png(unpack_rgb(source.read_bytes()), quality))
    temporary.replace(dest)
    return dest


def _chunk(kind: bytes, payload: bytes) -> bytes:
    check = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)
