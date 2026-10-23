"""PNG bytes for one RGB image. The encoder is the stdlib deflate stream.

The samples are the uint8 decode. ``quality`` selects the zlib level, and a
lower quality uses higher compression: ``low`` 9, ``medium`` 7, ``high`` 5,
``xhigh`` 3, ``max`` 1.
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


def encode_png(image: np.ndarray, quality: str = "high") -> bytes:
    """``image`` is uint8 ``[height, width, 3]``. ``quality`` is a reported level."""
    level = ZLIB_LEVEL.get(quality)
    if level is None:
        message = f"quality must be a reported image level, got {quality}"
        raise ValueError(message)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"PNG expects uint8 HWC, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(raw, level)),
            _chunk(b"IEND", b""),
        )
    )


def pack_rgb(image: np.ndarray) -> bytes:
    """Uncompressed RGB8. Width and height, then the uint8 samples."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"RGB expects uint8 HWC, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _ = image.shape
    samples = np.ascontiguousarray(image).tobytes()
    return struct.pack(">4sII", _RGB, width, height) + samples


def unpack_rgb(blob: bytes) -> np.ndarray:
    """The uint8 HWC array stored by ``pack_rgb``."""
    if len(blob) < 12 or blob[:4] != _RGB:
        message = "RGB signature is missing"
        raise ValueError(message)
    width, height = struct.unpack(">II", blob[4:12])
    samples = np.frombuffer(blob[12:], dtype=np.uint8)
    if width < 1 or height < 1 or samples.size != height * width * 3:
        message = "RGB payload does not match its header"
        raise ValueError(message)
    return samples.reshape(height, width, 3).copy()


def write_png(source: Path, quality: str = "high") -> Path:
    """PNG beside one saved RGB file. The served file is the PNG."""
    dest = source.with_suffix(".png")
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_bytes(encode_png(unpack_rgb(source.read_bytes()), quality))
    temporary.replace(dest)
    return dest


def _chunk(kind: bytes, payload: bytes) -> bytes:
    check = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)
