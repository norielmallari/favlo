"""Reference pixels. Edges are multiples of 16. The file is a PNG."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from .constants import ALIGN, MAX_INPUT_SIDE, MAX_SIDE, MIN_SIDE
from .png import decode_png


def read_image(value: str) -> np.ndarray:
    """uint8 HWC RGB from a path, a ``file://`` URI, a kind-folder filename, an image id, or a base64 PNG."""
    from favlo.media_name import MissingMediaFile

    try:
        image = decode_png(_bytes(value))
    except MissingMediaFile:
        raise
    except ValueError as exc:
        message = "image is not a PNG"
        raise ValueError(message) from exc
    if image.ndim != 3 or image.shape[2] != 3:
        message = "image is not an RGB image"
        raise ValueError(message)
    return image


def resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest resize. A matching size is returned unchanged."""
    src_h, src_w = image.shape[:2]
    if src_w == width and src_h == height:
        return image
    rows = np.linspace(0, src_h - 1, height).astype(np.int32)
    cols = np.linspace(0, src_w - 1, width).astype(np.int32)
    return np.ascontiguousarray(image[rows][:, cols])


def snap_crop(image: np.ndarray) -> np.ndarray:
    """Center-crop to multiples of 16. Does not scale."""
    src_h, src_w = image.shape[:2]
    height = snap_side(src_h)
    width = snap_side(src_w)
    if height == src_h and width == src_w:
        return image
    top = (src_h - height) // 2
    left = (src_w - width) // 2
    return np.ascontiguousarray(image[top : top + height, left : left + width])


def fit_input(image: np.ndarray) -> np.ndarray:
    """Shrink so the long side is at most 1024, keep aspect, snap to 16."""
    src_h, src_w = image.shape[:2]
    longest = max(src_h, src_w)
    if longest <= MAX_INPUT_SIDE:
        return snap_crop(image)
    scale = MAX_INPUT_SIDE / float(longest)
    width = snap_side(int(round(src_w * scale)))
    height = snap_side(int(round(src_h * scale)))
    return resize_nearest(image, width, height)


def snap_side(value: int) -> int:
    """The multiple of 16 inside 16..2048, rounding down."""
    snapped = max(MIN_SIDE, (int(value) // ALIGN) * ALIGN)
    return min(MAX_SIDE, snapped)


def to_model(image: np.ndarray) -> np.ndarray:
    """uint8 HWC to float16 NCHW in ``[-1, 1]``."""
    plane = image.astype(np.float32) / np.float32(127.5) - np.float32(1)
    return np.ascontiguousarray(plane.transpose(2, 0, 1), dtype=np.float16)


def to_u8(plane: np.ndarray) -> np.ndarray:
    """float CHW in ``[-1, 1]`` to uint8 HWC."""
    clipped = np.clip(plane.astype(np.float32), -1.0, 1.0)
    values = np.rint((clipped + 1.0) * 127.5)
    raw = np.clip(values, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(raw.transpose(1, 2, 0))


def _bytes(value: str) -> bytes:
    if value.startswith("file://"):
        from favlo.local_uri import read_file_uri

        try:
            return read_file_uri(value)
        except (OSError, ValueError) as exc:
            message = "image is not a PNG file"
            raise ValueError(message) from exc
    if value.startswith("data:"):
        _header, _sep, payload = value.partition(",")
        return base64.b64decode(payload)
    named = _named(value)
    if named is not None:
        return named
    if value.startswith("image_"):
        from favlo.images import load_image

        raw = load_image(value)
        if raw is None:
            message = "image is not a PNG file"
            raise ValueError(message)
        return raw
    path = Path(value)
    if len(value) < 4096 and path.is_file():
        return path.read_bytes()
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        message = "image is not a PNG file"
        raise ValueError(message) from exc


def _named(value: str) -> bytes | None:
    from favlo.media_name import read_plain_filename

    return read_plain_filename(value)
