"""PNG bytes for one RGB image. The encoder is the stdlib deflate stream."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def encode_png(image: np.ndarray) -> bytes:
    """``image`` is uint8 ``[height, width, 3]``."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"PNG expects uint8 HWC, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _chunk(b"IDAT", zlib.compress(raw, 9)),
            _chunk(b"IEND", b""),
        )
    )


def _chunk(kind: bytes, payload: bytes) -> bytes:
    check = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)
