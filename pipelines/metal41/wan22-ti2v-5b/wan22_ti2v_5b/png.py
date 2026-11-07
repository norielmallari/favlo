"""8-bit PNG, color types 2 and 6. No interlacing."""

from __future__ import annotations

import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_png(image: np.ndarray) -> bytes:
    """``image`` is ``[H, W, 3]`` uint8."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = "png encode wants uint8 RGB"
        raise ValueError(message)
    height, width, _channels = image.shape
    raw = bytearray()
    for row in image:
        raw.append(0)
        raw.extend(row.tobytes())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            _SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            _chunk(b"IEND", b""),
        )
    )


def decode_png(payload: bytes) -> np.ndarray:
    """Return ``[H, W, 3]`` uint8. Alpha is dropped."""
    if not payload.startswith(_SIGNATURE):
        message = "png signature is missing"
        raise ValueError(message)
    offset = len(_SIGNATURE)
    width = height = 0
    color = 0
    idat = bytearray()
    while offset + 8 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        start = offset + 8
        data = payload[start : start + length]
        if start + length + 4 > len(payload):
            message = "png chunk is truncated"
            raise ValueError(message)
        if kind == b"IHDR":
            width, height, depth, color, interlace = struct.unpack(">IIBBBBB", data)[:5]
            if depth != 8 or color not in (2, 6) or interlace != 0:
                message = "png must be 8-bit RGB or RGBA without interlacing"
                raise ValueError(message)
        elif kind == b"IDAT":
            idat.extend(data)
        elif kind == b"IEND":
            break
        offset = start + length + 4
    if width < 1 or height < 1 or not idat:
        message = "png has no image"
        raise ValueError(message)
    channels = 3 if color == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    image = np.empty((height, width, 3), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row in range(height):
        filt = raw[cursor]
        cursor += 1
        plane = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=cursor).copy()
        cursor += stride
        _unfilter(filt, plane, prior, channels)
        image[row] = plane.reshape(width, channels)[:, :3]
        prior = plane
    return image


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _unfilter(kind: int, plane: np.ndarray, prior: np.ndarray, channels: int) -> None:
    if kind == 0:
        return
    if kind == 1:
        for index in range(channels, plane.shape[0]):
            plane[index] = (int(plane[index]) + int(plane[index - channels])) & 0xFF
        return
    if kind == 2:
        plane[:] = (plane.astype(np.uint16) + prior.astype(np.uint16)) & 0xFF
        return
    if kind == 3:
        for index in range(plane.shape[0]):
            left = int(plane[index - channels]) if index >= channels else 0
            plane[index] = (int(plane[index]) + ((left + int(prior[index])) // 2)) & 0xFF
        return
    if kind == 4:
        for index in range(plane.shape[0]):
            left = int(plane[index - channels]) if index >= channels else 0
            up = int(prior[index])
            upper_left = int(prior[index - channels]) if index >= channels else 0
            plane[index] = (int(plane[index]) + _paeth(left, up, upper_left)) & 0xFF
        return
    message = f"png filter {kind} is unknown"
    raise ValueError(message)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    distance_l = abs(estimate - left)
    distance_u = abs(estimate - up)
    distance_ul = abs(estimate - upper_left)
    if distance_l <= distance_u and distance_l <= distance_ul:
        return left
    if distance_u <= distance_ul:
        return up
    return upper_left
