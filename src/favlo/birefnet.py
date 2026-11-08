"""Shared BiRefNet cutout.

One ONNX session serves every pipeline. A transparent request and a relight
both call ``cutout``. The weights live at ``models/BiRefNet/model.onnx``.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO = "ZhengPeng7/BiRefNet"
FILE = "BiRefNet-general-epoch_244.onnx"
URL = f"https://github.com/{REPO}/releases/download/v1/{FILE}"
EXPECTED = 972_666_916
INPUT = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_session: Any | None = None


def matte_dir(root: Path | None = None) -> Path:
    from favlo.cli import repo_root

    base = root or repo_root()
    return base / "models" / "BiRefNet"


def ensure_matte(dest: Path | None = None) -> Path:
    """Download the BiRefNet general ONNX when the finished file is missing."""
    folder = dest or matte_dir()
    path = folder / "model.onnx"
    if path.is_file() and path.stat().st_size == EXPECTED:
        return path
    folder.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    have = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "favlo-setup"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", 200))
        if status == 200 and have:
            have = 0
        mode = "ab" if status == 206 and have else "wb"
        written = have if mode == "ab" else 0
        last = time.time()
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last > 30:
                    print(f"setup-matte {FILE} {written}/{EXPECTED}", flush=True)
                    last = now
    if partial.stat().st_size != EXPECTED:
        message = f"{FILE} size {partial.stat().st_size} != {EXPECTED}"
        raise RuntimeError(message)
    os.replace(partial, path)
    return path


def cutout(image: np.ndarray, *, session: Any | None = None) -> np.ndarray:
    """Return uint8 RGBA. ``image`` is uint8 HWC RGB."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        message = f"matte expects uint8 RGB HWC, got {image.shape} {image.dtype}"
        raise ValueError(message)
    height, width, _ = image.shape
    alpha = _alpha(image, session=_session_or(session))
    if alpha.shape != (height, width):
        alpha = _resize(alpha, height, width)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = image
    rgba[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return rgba


def _session_or(session: Any | None) -> Any:
    if session is not None:
        return session
    global _session
    if _session is None:
        _session = _open(ensure_matte())
    return _session


def _open(path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        message = "onnxruntime is required for a transparent cutout"
        raise RuntimeError(message) from exc
    # CoreML partitions this graph into hundreds of pieces and stalls on first load.
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _alpha(image: np.ndarray, *, session: Any) -> np.ndarray:
    batch = _prepare(image)
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: batch})[0]
    if not isinstance(logits, np.ndarray) or logits.ndim != 4:
        message = f"BiRefNet returned shape {getattr(logits, 'shape', None)}"
        raise RuntimeError(message)
    return _sigmoid(logits[0, 0].astype(np.float32, copy=False))


def _prepare(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    if height == INPUT and width == INPUT:
        resized = image.astype(np.float32) / 255.0
    else:
        resized = _resize_rgb(image.astype(np.float32) / 255.0, INPUT, INPUT)
    for channel, mean, std in zip(range(3), MEAN, STD, strict=True):
        resized[..., channel] = (resized[..., channel] - mean) / std
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    planes = [_resize(image[..., channel], height, width) for channel in range(3)]
    return np.stack(planes, axis=-1)


def _resize(plane: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of one float plane to ``height`` by ``width``."""
    source_h, source_w = plane.shape
    if source_h == height and source_w == width:
        return plane
    if height < 1 or width < 1:
        message = f"resize target {height}x{width} is empty"
        raise ValueError(message)
    rows = np.linspace(0.0, source_h - 1.0, height, dtype=np.float64)
    cols = np.linspace(0.0, source_w - 1.0, width, dtype=np.float64)
    row0 = np.floor(rows).astype(np.int64)
    col0 = np.floor(cols).astype(np.int64)
    row1 = np.minimum(row0 + 1, source_h - 1)
    col1 = np.minimum(col0 + 1, source_w - 1)
    row_w = rows - row0
    col_w = cols - col0
    top = plane[row0][:, col0] * (1.0 - col_w) + plane[row0][:, col1] * col_w
    bottom = plane[row1][:, col0] * (1.0 - col_w) + plane[row1][:, col1] * col_w
    return (top * (1.0 - row_w)[:, None] + bottom * row_w[:, None]).astype(
        np.float32, copy=False
    )
