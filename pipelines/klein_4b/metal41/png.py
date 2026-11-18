"""PNG for one uint8 RGB image. The uncompressed sibling is RGB8."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"
_RGB = b"RGB8"


def encode_png(image: np.ndarray) -> bytes:
    """``image`` is uint8 HWC RGB. Each row uses filter None."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"PNG expects uint8 HWC RGB, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _channels = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            _SIG,
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(raw, 6)),
            _chunk(b"IEND", b""),
        )
    )


def decode_png(blob: bytes) -> np.ndarray:
    """uint8 HWC RGB, dropping alpha. Adam7 is refused."""
    if not blob.startswith(_SIG):
        message = "PNG signature is missing"
        raise ValueError(message)
    offset = len(_SIG)
    width = height = 0
    color = 0
    parts: list[bytes] = []
    while offset + 8 <= len(blob):
        length = struct.unpack(">I", blob[offset : offset + 4])[0]
        kind = blob[offset + 4 : offset + 8]
        start = offset + 8
        payload = blob[start : start + length]
        check = struct.unpack(">I", blob[start + length : start + length + 4])[0]
        if (zlib.crc32(kind + payload) & 0xFFFFFFFF) != check:
            message = "PNG chunk crc mismatch"
            raise ValueError(message)
        offset = start + length + 4
        if kind == b"IHDR":
            width, height, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", payload)
            if depth != 8 or color not in {2, 6} or interlace != 0:
                message = "PNG must be 8-bit RGB or RGBA without interlacing"
                raise ValueError(message)
        elif kind == b"IDAT":
            parts.append(payload)
        elif kind == b"IEND":
            break
    channels = 4 if color == 6 else 3
    raw = zlib.decompress(b"".join(parts))
    stride = width * channels
    rows = np.empty((height, width, channels), dtype=np.uint8)
    cursor = 0
    prior = np.zeros(stride, dtype=np.uint8)
    for row in range(height):
        filt = raw[cursor]
        cursor += 1
        scan = np.frombuffer(raw[cursor : cursor + stride], dtype=np.uint8).copy()
        cursor += stride
        _unfilter(filt, scan, prior, channels)
        rows[row] = scan.reshape(width, channels)
        prior = scan
    if channels == 4:
        rows = rows[:, :, :3]
    return rows


def pack_rgb(image: np.ndarray) -> bytes:
    """Uncompressed RGB8. Width and height, then the uint8 samples."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"RGB expects uint8 HWC RGB, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _channels = image.shape
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


def write_png(source: Path) -> Path:
    """PNG beside one saved RGB file. The served file is the PNG."""
    dest = source.with_suffix(".png")
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_bytes(encode_png(unpack_rgb(source.read_bytes())))
    temporary.replace(dest)
    return dest


def _chunk(kind: bytes, payload: bytes) -> bytes:
    check = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)


def _unfilter(filt: int, scan: np.ndarray, prior: np.ndarray, channels: int) -> None:
    if filt == 0:
        return
    if filt == 1:
        for index in range(channels, scan.shape[0]):
            scan[index] = (int(scan[index]) + int(scan[index - channels])) & 255
        return
    if filt == 2:
        scan[:] = (scan.astype(np.int16) + prior.astype(np.int16)) & 255
        return
    if filt == 3:
        for index in range(scan.shape[0]):
            left = int(scan[index - channels]) if index >= channels else 0
            scan[index] = (int(scan[index]) + ((left + int(prior[index])) // 2)) & 255
        return
    if filt == 4:
        for index in range(scan.shape[0]):
            left = int(scan[index - channels]) if index >= channels else 0
            up = int(prior[index])
            upper_left = int(prior[index - channels]) if index >= channels else 0
            scan[index] = (int(scan[index]) + _paeth(left, up, upper_left)) & 255
        return
    message = f"PNG filter {filt} is not supported"
    raise ValueError(message)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    dist_left = abs(estimate - left)
    dist_up = abs(estimate - up)
    dist_upper = abs(estimate - upper_left)
    if dist_left <= dist_up and dist_left <= dist_upper:
        return left
    if dist_up <= dist_upper:
        return up
    return upper_left
