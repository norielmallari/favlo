"""Image tokens for the Qwen3.5 vision tower.

The chat template writes one ``<|image_pad|>``. This module expands that token
to the merged grid and builds the interleaved temporal, height, and width
positions. A chat image is an ``http`` or ``https`` URL, a data URL, or
base64, and PNG, JPEG, GIF, and WebP decode to RGB. Pixel values follow the
checkpoint processor: rescale by 1/255, then mean 0.5 and standard deviation
0.5. A side that is already a multiple of 32 is kept. Any other side is
resampled with Catmull-Rom so both sides stay on that grid and the pixel
count stays inside 65536 through 16777216. ``detail`` and per-part
``min_pixels`` and ``max_pixels`` choose the budget inside those edges.
"""

from __future__ import annotations

import base64
import ipaddress
import math
import socket
import struct
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any

import numpy as np

IMAGE_PAD = 248056
PATCH = 16
TEMPORAL = 2
MERGE = 2
FACTOR = PATCH * MERGE
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
MAX_RATIO = 200
MAX_IMAGE_BYTES = 64 * 1024 * 1024
CHANNELS = 3
_FETCH_SECONDS = 30.0

_PNG = b"\x89PNG\r\n\x1a\n"


def collect_images(
    messages: list[dict[str, Any]], *, note: Any = None
) -> list[tuple[bytes, int, int]]:
    """Image bytes and the pixel budget, in placeholder order.

    ``note`` receives ``vision image N`` before that image's bytes are read.
    """
    found: list[tuple[bytes, int, int]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if index == 0 and message.get("role") == "system":
            continue
        _walk_content(message.get("content"), found, note)
    return found


def expand_image_tokens(ids: list[int], counts: list[int]) -> list[int]:
    """Replace each image placeholder with ``counts`` copies of itself."""
    expanded: list[int] = []
    image = 0
    for token in ids:
        if token != IMAGE_PAD:
            expanded.append(int(token))
            continue
        if image >= len(counts):
            message = "the prompt has an image token with no image"
            raise RuntimeError(message)
        expanded.extend([IMAGE_PAD] * counts[image])
        image += 1
    if image != len(counts):
        message = "the prompt has fewer image tokens than images"
        raise RuntimeError(message)
    return expanded


def language_positions(ids: list[int], merged: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    """Interleaved M-RoPE ids. Text uses one id on every axis.

    An image keeps a constant temporal id and writes the merged row and column
    on the other two axes. The next text id is one past the largest id the
    image used.
    """
    positions = np.zeros((len(ids), 3), dtype=np.uint32)
    cursor = 0
    image = 0
    index = 0
    while index < len(ids):
        if ids[index] != IMAGE_PAD:
            positions[index] = cursor
            cursor += 1
            index += 1
            continue
        if image >= len(merged):
            message = "image tokens ran past the prepared grids"
            raise RuntimeError(message)
        height, width = merged[image]
        image += 1
        count = height * width
        if index + count > len(ids) or any(token != IMAGE_PAD for token in ids[index : index + count]):
            message = "image tokens do not match the merged grid"
            raise RuntimeError(message)
        flat = 0
        for row in range(height):
            for col in range(width):
                positions[index + flat, 0] = cursor
                positions[index + flat, 1] = cursor + row
                positions[index + flat, 2] = cursor + col
                flat += 1
        cursor += max(height, width, 1)
        index += count
    if image != len(merged):
        message = "a prepared image was not in the prompt"
        raise RuntimeError(message)
    return positions, cursor


def prepare_image(
    blob: bytes,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Normalized patches in merge-block order, and the patch grid ``(h, w)``."""
    image = decode_image(blob)
    height, width = _smart_size(
        image.shape[0], image.shape[1], min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (height, width) != image.shape[:2]:
        image = _resize(image, height, width)
    pixels = image.astype(np.float32) * np.float32(1.0 / 255.0)
    pixels = (pixels - np.float32(0.5)) / np.float32(0.5)
    grid_h = height // PATCH
    grid_w = width // PATCH
    return _patches(pixels, grid_h, grid_w), (grid_h, grid_w)


def merged_grid(grid: tuple[int, int]) -> tuple[int, int]:
    height, width = grid
    return height // MERGE, width // MERGE


def log_prepared(runtime: Any, prepared: list[tuple[np.ndarray, tuple[int, int]]]) -> None:
    """Log each still's patch grid and merged token count."""
    note = getattr(runtime, "_status", None)
    if not callable(note):
        return
    for index, (patches, grid) in enumerate(prepared, start=1):
        height, width = merged_grid(grid)
        note(
            f"vision image {index} patches {int(patches.shape[0])} "
            f"grid {grid[0]}x{grid[1]} merged {height * width}"
        )


def decode_png(blob: bytes) -> np.ndarray:
    """uint8 RGB from an 8-bit PNG. The alpha channel is dropped."""
    if not blob.startswith(_PNG):
        message = "image is not a PNG"
        raise ValueError(message)
    offset = len(_PNG)
    width = height = 0
    depth = color = interlace = 0
    parts: list[bytes] = []
    while offset + 8 <= len(blob):
        length = struct.unpack_from(">I", blob, offset)[0]
        kind = blob[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(blob):
            message = "PNG chunk is truncated"
            raise ValueError(message)
        chunk = blob[start:end]
        if kind == b"IHDR":
            width, height, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            parts.append(chunk)
        elif kind == b"IEND":
            break
        offset = end + 4
    if depth != 8 or interlace != 0 or color not in {2, 6}:
        message = "PNG must be 8-bit RGB or RGBA without interlacing"
        raise ValueError(message)
    channels = 3 if color == 2 else 4
    raw = zlib.decompress(b"".join(parts))
    stride = width * channels
    rows = np.empty((height, width, 3), dtype=np.uint8)
    cursor = 0
    previous = np.zeros(stride, dtype=np.uint8)
    for row in range(height):
        if cursor + 1 + stride > len(raw):
            message = "PNG rows are truncated"
            raise ValueError(message)
        filt = raw[cursor]
        cursor += 1
        current = np.frombuffer(raw[cursor : cursor + stride], dtype=np.uint8).copy()
        cursor += stride
        _unfilter(filt, current, previous, channels)
        previous = current
        view = current.reshape(width, channels)
        rows[row] = view[:, :3]
    return rows


def decode_image(blob: bytes) -> np.ndarray:
    """RGB uint8 from a PNG, JPEG, GIF, or WebP payload."""
    if blob.startswith(_PNG):
        try:
            return decode_png(blob)
        except (RuntimeError, ValueError):
            pass
    return _decode_bitmap(blob)


def _walk_content(content: Any, found: list[tuple[bytes, int, int]], note: Any) -> None:
    if isinstance(content, str) or content is None:
        return
    if not isinstance(content, list):
        return
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "video" or "video" in item or "video_url" in item:
            message = "video input is not supported"
            raise RuntimeError(message)
        if kind in {"image", "image_url"} or "image" in item or "image_url" in item:
            if callable(note):
                note(f"vision image {len(found) + 1}")
            low, high = _pixel_bounds(item)
            found.append((_image_bytes(item), low, high))


def _image_bytes(item: dict[str, Any]) -> bytes:
    raw = item.get("image_url", item.get("image", item.get("url")))
    if isinstance(raw, dict):
        raw = raw.get("url")
    if not isinstance(raw, str) or not raw:
        message = "image part has no data"
        raise RuntimeError(message)
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        if "base64" not in header:
            message = "image data url is not base64"
            raise RuntimeError(message)
        try:
            return base64.b64decode("".join(payload.split()), validate=True)
        except (ValueError, TypeError) as exc:
            message = "image data url is not base64"
            raise RuntimeError(message) from exc
    if "://" in raw:
        if not raw.startswith(("http://", "https://")):
            message = "image url must be http or https"
            raise RuntimeError(message)
        return _fetch_image(raw)
    try:
        return base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        message = "image url must be http, https, or base64"
        raise RuntimeError(message) from exc


def _pixel_bounds(item: dict[str, Any]) -> tuple[int, int]:
    nested = item.get("image_url")
    if not isinstance(nested, dict):
        nested = item.get("image")
    if not isinstance(nested, dict):
        nested = {}
    low = _pixel_field(item, nested, "min_pixels")
    high = _pixel_field(item, nested, "max_pixels")
    detail = nested.get("detail", item.get("detail"))
    if detail is not None and detail not in {"auto", "low", "high", "original"}:
        message = f"image detail {detail} is unsupported"
        raise RuntimeError(message)
    if low is None and high is None:
        if detail == "low":
            return MIN_PIXELS, MIN_PIXELS
        return MIN_PIXELS, MAX_PIXELS
    low = MIN_PIXELS if low is None else low
    high = MAX_PIXELS if high is None else high
    if low < MIN_PIXELS or high > MAX_PIXELS or low > high:
        message = f"image pixels {low}..{high} are outside {MIN_PIXELS}..{MAX_PIXELS}"
        raise RuntimeError(message)
    return low, high


def _pixel_field(item: dict[str, Any], nested: dict[str, Any], key: str) -> int | None:
    value = item[key] if key in item else nested.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"image {key} is not an integer"
        raise RuntimeError(message)
    return value


def _fetch_image(url: str) -> bytes:
    _check_url(url)
    opener = urllib.request.build_opener(_PublicRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "favlo"})
    try:
        with opener.open(request, timeout=_FETCH_SECONDS) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(65536)
                if not block:
                    break
                total += len(block)
                if total > MAX_IMAGE_BYTES:
                    message = "image url is too large"
                    raise RuntimeError(message)
                chunks.append(block)
    except urllib.error.URLError as exc:
        message = "image url could not be fetched"
        raise RuntimeError(message) from exc
    if not chunks:
        message = "image url is empty"
        raise RuntimeError(message)
    return b"".join(chunks)


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        message = "image url must be http or https"
        raise RuntimeError(message)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        message = "image url did not resolve"
        raise RuntimeError(message) from exc
    for info in found:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            message = "image url is not a public address"
            raise RuntimeError(message)


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode_bitmap(blob: bytes) -> np.ndarray:
    from AppKit import NSBitmapImageRep
    from Foundation import NSData

    data = NSData.dataWithBytes_length_(blob, len(blob))
    rep = NSBitmapImageRep.imageRepWithData_(data)
    if rep is None:
        message = "image format is unsupported"
        raise RuntimeError(message)
    width = int(rep.pixelsWide())
    height = int(rep.pixelsHigh())
    channels = int(rep.bitsPerPixel()) // 8
    row_bytes = int(rep.bytesPerRow())
    if (
        width < 1
        or height < 1
        or channels < 3
        or int(rep.bitsPerSample()) != 8
        or bool(rep.isPlanar())
        or row_bytes < width * channels
    ):
        message = "image format is unsupported"
        raise RuntimeError(message)
    raw = np.frombuffer(rep.bitmapData(), dtype=np.uint8, count=row_bytes * height)
    view = raw.reshape(height, row_bytes)[:, : width * channels]
    rgb = view.reshape(height, width, channels)
    return np.ascontiguousarray(rgb[:, :, :3])


def _smart_size(
    height: int,
    width: int,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """The Qwen2-VL resize: multiples of 32 inside the pixel limits."""
    if height < 1 or width < 1:
        message = f"image size {width}x{height} is empty"
        raise RuntimeError(message)
    if min_pixels < 1 or max_pixels < min_pixels:
        message = f"image pixels {min_pixels}..{max_pixels} are empty"
        raise RuntimeError(message)
    if max(height, width) / min(height, width) > MAX_RATIO:
        message = f"image aspect {width}x{height} exceeds {MAX_RATIO}"
        raise RuntimeError(message)

    def nearest(value: int) -> int:
        return max(FACTOR, int(math.floor(value / FACTOR + 0.5)) * FACTOR)

    out_h = nearest(height)
    out_w = nearest(width)
    pixels = out_h * out_w
    if pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        out_h = max(FACTOR, int(math.floor(height / beta / FACTOR)) * FACTOR)
        out_w = max(FACTOR, int(math.floor(width / beta / FACTOR)) * FACTOR)
    elif pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        out_h = int(math.ceil(height * beta / FACTOR)) * FACTOR
        out_w = int(math.ceil(width * beta / FACTOR)) * FACTOR
    return out_h, out_w


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Catmull-Rom resize of uint8 HWC."""
    src_h, src_w, channels = image.shape
    out = np.empty((height, width, channels), dtype=np.uint8)
    ys = np.linspace(0, src_h - 1, height)
    xs = np.linspace(0, src_w - 1, width)
    padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="edge")
    for row, y in enumerate(ys):
        y0 = int(math.floor(y))
        fy = y - y0
        for col, x in enumerate(xs):
            x0 = int(math.floor(x))
            fx = x - x0
            acc = np.zeros(channels, dtype=np.float32)
            for ky in range(-1, 3):
                wy = _cubic(fy - ky)
                sample = padded[y0 + ky + 2, x0 + 1 : x0 + 5].astype(np.float32)
                weights = np.array([_cubic(fx - kx) for kx in range(-1, 3)], dtype=np.float32)
                acc += wy * (sample * weights[:, None]).sum(axis=0)
            out[row, col] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    return out


def _cubic(offset: float) -> float:
    ax = abs(offset)
    if ax <= 1.0:
        return (1.5 * ax - 2.5) * ax * ax + 1.0
    if ax < 2.0:
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0
    return 0.0


def _patches(pixels: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Conv3d patches. Time repeats the frame. Order is a 2×2 merge block."""
    count = grid_h * grid_w
    packed = np.empty((count, CHANNELS * TEMPORAL * PATCH * PATCH), dtype=np.float32)
    blocks_w = grid_w // MERGE
    seq = 0
    for block in range((grid_h // MERGE) * blocks_w):
        bh = block // blocks_w
        bw = block - bh * blocks_w
        for intra in range(MERGE * MERGE):
            ih = intra // MERGE
            iw = intra - ih * MERGE
            y0 = (bh * MERGE + ih) * PATCH
            x0 = (bw * MERGE + iw) * PATCH
            tile = pixels[y0 : y0 + PATCH, x0 : x0 + PATCH]
            cursor = 0
            flat = packed[seq]
            for channel in range(CHANNELS):
                plane = tile[:, :, channel]
                for _time in range(TEMPORAL):
                    flat[cursor : cursor + PATCH * PATCH] = plane.reshape(-1)
                    cursor += PATCH * PATCH
            seq += 1
    return packed


def _unfilter(kind: int, current: np.ndarray, previous: np.ndarray, channels: int) -> None:
    """Undo one PNG scanline. The left neighbor is one pixel, ``channels`` bytes back."""
    if kind == 0:
        return
    if kind == 1:
        for index in range(len(current)):
            left = int(current[index - channels]) if index >= channels else 0
            current[index] = (int(current[index]) + left) & 0xFF
        return
    if kind == 2:
        current[:] = (current.astype(np.uint16) + previous.astype(np.uint16)) & 0xFF
        return
    if kind == 3:
        for index in range(len(current)):
            left = int(current[index - channels]) if index >= channels else 0
            current[index] = (int(current[index]) + ((left + int(previous[index])) // 2)) & 0xFF
        return
    if kind == 4:
        for index in range(len(current)):
            left = int(current[index - channels]) if index >= channels else 0
            up = int(previous[index])
            upper_left = int(previous[index - channels]) if index >= channels else 0
            current[index] = (int(current[index]) + _paeth(left, up, upper_left)) & 0xFF
        return
    message = f"PNG filter {kind} is not supported"
    raise ValueError(message)


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
