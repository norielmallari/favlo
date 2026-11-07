"""Uncompressed 24-bit AVI. Frames are RGB uint8, stored bottom-up BGR."""

from __future__ import annotations

import struct

import numpy as np

from wan22_ti2v_5b.spec import FPS


def encode_avi(frames: np.ndarray, fps: int = FPS) -> bytes:
    """``frames`` is ``[T, H, W, 3]`` uint8, top-down RGB."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        message = "avi encode wants uint8 RGB frames"
        raise ValueError(message)
    count, height, width, _channels = frames.shape
    row = (width * 3 + 3) & ~3
    size = row * height
    pictures = [_dib(frame, row) for frame in frames]
    movi = bytearray()
    index = bytearray()
    offset = 4
    for picture in pictures:
        movi.extend(b"00db")
        movi.extend(struct.pack("<I", size))
        movi.extend(picture)
        index.extend(b"00db")
        index.extend(struct.pack("<III", 16, offset, size))
        offset += 8 + size
    movi_size = 4 + len(movi)
    hdrl = _hdrl(count, width, height, size, fps)
    riff = b"AVI " + b"LIST" + struct.pack("<I", 4 + len(hdrl)) + b"hdrl" + hdrl
    riff += b"LIST" + struct.pack("<I", movi_size) + b"movi" + movi
    riff += b"idx1" + struct.pack("<I", len(index)) + index
    return b"RIFF" + struct.pack("<I", len(riff)) + riff


def _dib(frame: np.ndarray, row: int) -> bytes:
    height, width, _channels = frame.shape
    picture = np.zeros((height, row), dtype=np.uint8)
    bgr = frame[:, :, ::-1]
    picture[:, : width * 3] = bgr.reshape(height, width * 3)
    return np.ascontiguousarray(picture[::-1]).tobytes()


def _hdrl(count: int, width: int, height: int, size: int, fps: int) -> bytes:
    micro = int(1_000_000 / fps)
    avih = struct.pack(
        "<IIIIIIIIIIIIII",
        micro,
        size * fps,
        0,
        0x10 | 0x100,
        count,
        0,
        1,
        size,
        width,
        height,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIHHHH",
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
        size,
        0xFFFFFFFF,
        size,
        0,
        0,
        width,
        height,
    )
    strf = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, size, 0, 0, 0, 0)
    stream = b"strh" + struct.pack("<I", len(strh)) + strh
    stream += b"strf" + struct.pack("<I", len(strf)) + strf
    body = b"avih" + struct.pack("<I", len(avih)) + avih
    body += b"LIST" + struct.pack("<I", 4 + len(stream)) + b"strl" + stream
    return body
