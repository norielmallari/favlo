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


def decode_png(blob: bytes) -> np.ndarray:
    """uint8 HWC RGB from an 8-bit non-interlaced RGB PNG."""
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        message = "PNG signature is missing"
        raise ValueError(message)
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset + 8 <= len(blob):
        length = struct.unpack_from(">I", blob, offset)[0]
        kind = blob[offset + 4 : offset + 8]
        payload = blob[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", payload)
            if (depth, color, comp, filt, interlace) != (8, 2, 0, 0, 0):
                message = "PNG must be 8-bit RGB without interlacing"
                raise ValueError(message)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if width < 1 or height < 1:
        message = "PNG header is missing"
        raise ValueError(message)
    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    image = np.empty((height, width, 3), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row in range(height):
        filt = raw[cursor]
        cursor += 1
        scan = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=cursor).copy()
        cursor += stride
        _unfilter(filt, scan, previous)
        image[row] = scan.reshape(width, 3)
        previous = scan
    return image


def resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of uint8 HWC. An image that already fits is returned."""
    src_h, src_w, _channels = image.shape
    if (src_h, src_w) == (height, width):
        return image
    y = np.linspace(0.0, src_h - 1, height, dtype=np.float32)
    x = np.linspace(0.0, src_w - 1, width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    top = image[y0][:, x0] * (1.0 - wx)[None, :, None]
    top += image[y0][:, x1] * wx[None, :, None]
    bot = image[y1][:, x0] * (1.0 - wx)[None, :, None]
    bot += image[y1][:, x1] * wx[None, :, None]
    mixed = top * (1.0 - wy)[:, None, None] + bot * wy[:, None, None]
    return np.rint(np.clip(mixed, 0.0, 255.0)).astype(np.uint8)


def _unfilter(kind: int, scan: np.ndarray, previous: np.ndarray) -> None:
    stride = int(scan.shape[0])
    if kind == 0:
        return
    if kind == 1:
        for index in range(3, stride):
            scan[index] = (int(scan[index]) + int(scan[index - 3])) & 255
        return
    if kind == 2:
        scan[:] = (scan.astype(np.int16) + previous.astype(np.int16)) & 255
        return
    if kind == 3:
        for index in range(stride):
            left = int(scan[index - 3]) if index >= 3 else 0
            scan[index] = (int(scan[index]) + ((left + int(previous[index])) // 2)) & 255
        return
    if kind == 4:
        for index in range(stride):
            left = int(scan[index - 3]) if index >= 3 else 0
            up = int(previous[index])
            upper_left = int(previous[index - 3]) if index >= 3 else 0
            scan[index] = (int(scan[index]) + _paeth(left, up, upper_left)) & 255
        return
    message = f"PNG filter {kind} is not supported"
    raise ValueError(message)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    nearest = left
    distance = abs(estimate - left)
    if abs(estimate - up) < distance:
        nearest = up
        distance = abs(estimate - up)
    if abs(estimate - upper_left) < distance:
        nearest = upper_left
    return nearest


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
