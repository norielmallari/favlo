"""PNG bytes for one RGB image. The codec is the stdlib deflate stream."""

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


def decode_png(blob: bytes) -> np.ndarray:
    """uint8 ``[height, width, 3]`` from an 8-bit RGB PNG."""
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
        _unfilter(filt, scan, previous, stride)
        image[row] = scan.reshape(width, 3)
        previous = scan
    return image


def resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of uint8 HWC. An image that already fits is returned."""
    src_h, src_w, _ = image.shape
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


def _unfilter(kind: int, scan: np.ndarray, previous: np.ndarray, stride: int) -> None:
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
            ul = int(previous[index - 3]) if index >= 3 else 0
            scan[index] = (int(scan[index]) + _paeth(left, up, ul)) & 255
        return
    message = f"PNG filter {kind} is not supported"
    raise ValueError(message)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    dl = abs(estimate - left)
    du = abs(estimate - up)
    dlu = abs(estimate - upper_left)
    if dl <= du and dl <= dlu:
        return left
    if du <= dlu:
        return up
    return upper_left


def _chunk(kind: bytes, payload: bytes) -> bytes:
    check = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)
