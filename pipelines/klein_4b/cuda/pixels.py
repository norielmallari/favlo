"""Still sizes, file reads, and the float pixel conversion."""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from favlo.media_pipeline import MediaRequestError
from klein_4b.cuda.constants import ALIGN, MAX_INPUT_SIDE, MAX_SIDE, MIN_SIDE
from klein_4b.cuda.png import decode_png


def fit_reference(rgb: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(rgb)
    height, width = image.shape[:2]
    if max(height, width) > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / max(height, width)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
        image = _nearest(image, height, width)
    snapped_h = (image.shape[0] // ALIGN) * ALIGN
    snapped_w = (image.shape[1] // ALIGN) * ALIGN
    if snapped_h < MIN_SIDE or snapped_w < MIN_SIDE:
        message = "a reference must be at least 16 pixels on both edges"
        raise MediaRequestError(message, "image")
    if snapped_h != image.shape[0] or snapped_w != image.shape[1]:
        y0 = (image.shape[0] - snapped_h) // 2
        x0 = (image.shape[1] - snapped_w) // 2
        image = image[y0 : y0 + snapped_h, x0 : x0 + snapped_w]
    return np.ascontiguousarray(image)


def read_still(value: str) -> np.ndarray:
    payload = _bytes(value)
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        message = "image must be a PNG"
        raise MediaRequestError(message, "image")
    try:
        rgb = decode_png(payload)
    except ValueError as exc:
        raise MediaRequestError(str(exc), "image") from exc
    return fit_reference(rgb)


def parse_size(value: object, *, fallback: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return fallback
    if not isinstance(value, str) or "x" not in value.lower():
        message = "size must be WxH"
        raise MediaRequestError(message, "size")
    left, right = value.lower().split("x", 1)
    try:
        width = int(left)
        height = int(right)
    except ValueError as exc:
        message = "size must be WxH"
        raise MediaRequestError(message, "size") from exc
    if width % ALIGN or height % ALIGN or width < MIN_SIDE or height < MIN_SIDE or width > MAX_SIDE or height > MAX_SIDE:
        message = "size must be multiples of 16 from 16 to 2048"
        raise MediaRequestError(message, "size")
    return width, height


def to_model(rgb: np.ndarray) -> np.ndarray:
    channels = np.transpose(rgb.astype(np.float32), (2, 0, 1))
    return channels / np.float32(127.5) - np.float32(1.0)


def to_u8(channels: np.ndarray) -> np.ndarray:
    clipped = np.clip(channels, -1.0, 1.0)
    values = np.rint((clipped + 1.0) * 127.5).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(values, (1, 2, 0)))


def _nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.clip(np.rint(np.arange(height) * (image.shape[0] / height)).astype(np.int64), 0, image.shape[0] - 1)
    cols = np.clip(np.rint(np.arange(width) * (image.shape[1] / width)).astype(np.int64), 0, image.shape[1] - 1)
    return np.ascontiguousarray(image[rows][:, cols])


def _bytes(value: str) -> bytes:
    if value.startswith("file://"):
        return _uri_path(value).read_bytes()
    if value.startswith("data:"):
        text = value.split(",", 1)[1]
        return base64.b64decode(text)
    if len(value) < 1024:
        path = Path(value)
        if path.is_file():
            return path.read_bytes()
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        message = "image must be a PNG path or base64"
        raise MediaRequestError(message, "image") from exc


def _uri_path(value: str) -> Path:
    path = unquote(urlparse(value).path)
    if len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)
