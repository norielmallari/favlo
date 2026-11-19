"""PNG encode and decode for 8-bit stills. No third-party image library."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def encode_png(rgb: np.ndarray) -> bytes:
    """``rgb`` is uint8 HWC with 3 channels."""
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        message = "PNG encode expects RGB"
        raise ValueError(message)
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _signature() + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")


def decode_png(payload: bytes) -> np.ndarray:
    """Return uint8 HWC. Gray and gray-alpha expand to RGB. Palette is refused."""
    if not payload.startswith(_signature()):
        message = "not a PNG"
        raise ValueError(message)
    offset = 8
    width = height = 0
    color = 0
    depth = 0
    interlace = 0
    compressed = bytearray()
    palette = b""
    while offset + 8 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        offset += 4
        tag = payload[offset : offset + 4]
        offset += 4
        data = payload[offset : offset + length]
        offset += length + 4
        if tag == b"IHDR":
            width, height, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", data)
        elif tag == b"PLTE":
            palette = data
        elif tag == b"IDAT":
            compressed.extend(data)
        elif tag == b"IEND":
            break
    if depth != 8 or interlace != 0:
        message = "only 8-bit non-interlaced PNG"
        raise ValueError(message)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    if channels is None:
        message = "unsupported PNG color"
        raise ValueError(message)
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    rows = []
    previous = bytearray(stride)
    cursor = 0
    for _row in range(height):
        filt = raw[cursor]
        cursor += 1
        scan = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        current = _unfilter(filt, scan, previous, channels)
        previous = current
        rows.append(current)
    samples = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)
    if color == 3:
        message = "paletted PNG is not supported"
        raise ValueError(message)
    if color in {0, 4}:
        gray = samples[:, :, :1]
        rgb = np.repeat(gray, 3, axis=2)
        return np.ascontiguousarray(rgb)
    if color == 6:
        return np.ascontiguousarray(samples[:, :, :3])
    return np.ascontiguousarray(samples)


def _unfilter(kind: int, scan: bytearray, previous: bytearray, bpp: int) -> bytearray:
    out = bytearray(len(scan))
    for index, value in enumerate(scan):
        left = out[index - bpp] if index >= bpp else 0
        up = previous[index]
        upper_left = previous[index - bpp] if index >= bpp else 0
        if kind == 0:
            extra = 0
        elif kind == 1:
            extra = left
        elif kind == 2:
            extra = up
        elif kind == 3:
            extra = (left + up) // 2
        elif kind == 4:
            extra = _paeth(left, up, upper_left)
        else:
            message = "bad PNG filter"
            raise ValueError(message)
        out[index] = (value + extra) & 0xFF
    return out


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    dl = abs(estimate - left)
    du = abs(estimate - up)
    dul = abs(estimate - upper_left)
    if dl <= du and dl <= dul:
        return left
    if du <= dul:
        return up
    return upper_left


def _signature() -> bytes:
    return b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
