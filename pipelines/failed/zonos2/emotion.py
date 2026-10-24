"""Emotion tags. The ids are the Zyphra ``emotion_directions`` set."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zonos2.audio import accurate_token, background_token, quality_token
from zonos2.layout import DIM

_ROOT = Path(__file__).resolve().parent / "emotions"
_TAG = re.compile(r"\[([A-Za-z][A-Za-z_]*)(?::([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?\]")
_STRENGTH = 1.0

_directions: dict[str, np.ndarray] | None = None
_order: tuple[str, ...] | None = None
_calibration: dict[str, float] | None = None
_global_default = 3.0


@dataclass(frozen=True)
class Span:
    """One speak. ``weights`` is None when the span has no tag."""

    text: str
    weights: tuple[tuple[str, float], ...] | None


def list_emotions() -> list[dict[str, object]]:
    """Ids a client may write as a square-bracket tag."""
    _load()
    assert _order is not None
    return [{"id": name, "sliders": {name: 1.0}} for name in _order]


def parse(text: str) -> list[Span]:
    """Split ``text`` into spoken spans. Tags are not spoken."""
    if not isinstance(text, str):
        message = "input is required"
        raise ValueError(message)
    _load()
    _reject_broken_brackets(text)
    spans: list[Span] = []
    weights: dict[str, float] | None = None
    cursor = 0
    for match in _TAG.finditer(text):
        between = text[cursor:match.start()]
        if between.strip():
            spans.append(_span(between, weights))
            weights = {}
        elif weights is None:
            weights = {}
        name = match.group(1).lower()
        scale = 1.0 if match.group(2) is None else float(match.group(2))
        if not math.isfinite(scale):
            message = "an emotion tag is incomplete"
            raise ValueError(message)
        weights = _add(weights or {}, name, scale)
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        spans.append(_span(tail, weights))
    elif weights is not None:
        message = "an emotion tag has no words"
        raise ValueError(message)
    elif not spans:
        message = "text is empty"
        raise ValueError(message)
    return spans


def delta_for(weights: tuple[tuple[str, float], ...] | None) -> np.ndarray | None:
    """Projected-speaker delta. None adds no direction."""
    if not weights:
        return None
    _load()
    assert _directions is not None and _calibration is not None
    total = np.zeros(DIM, dtype=np.float32)
    used = False
    for name, weight in weights:
        if weight == 0.0:
            continue
        gain = _STRENGTH * weight * _calibration.get(name, _global_default)
        total += np.float32(gain) * _directions[name]
        used = True
    if not used or not np.any(total):
        return None
    return np.ascontiguousarray(total, dtype=np.float32)


def conditioning_tokens(*, trailing: bool) -> tuple[int, ...]:
    """Noisy background, accurate mode, then the trailing-silence token on the last span."""
    tokens = [background_token(False), accurate_token()]
    if trailing:
        tokens.append(quality_token(5, 3))
    return tuple(tokens)


def _load() -> None:
    global _directions, _order, _calibration, _global_default
    if _directions is not None:
        return
    manifest_path = _ROOT / "manifest.json"
    if not manifest_path.is_file():
        message = "emotion directions are missing"
        raise ValueError(message)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("space")) != "proj" or int(manifest.get("dim", 0)) != DIM:
        message = "emotion directions are not projected speaker vectors"
        raise ValueError(message)
    directions: dict[str, np.ndarray] = {}
    order: list[str] = []
    for name, entry in manifest["directions"].items():
        vector = np.load(_ROOT / str(entry["file"]))
        if getattr(vector, "shape", None) != (DIM,) or not np.isfinite(vector).all():
            message = "an emotion direction is 2048 floats"
            raise ValueError(message)
        directions[str(name)] = np.ascontiguousarray(vector, dtype=np.float32)
        order.append(str(name))
    calibration = {name: _global_default for name in order}
    calibration_path = _ROOT / "calibration.json"
    if calibration_path.is_file():
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        _global_default = float(data.get("global_default", _global_default))
        for name in order:
            calibration[name] = _global_default
        for name, value in data.get("default", {}).items():
            if name in calibration:
                calibration[str(name)] = float(value)
    _directions = directions
    _order = tuple(order)
    _calibration = calibration


def _reject_broken_brackets(text: str) -> None:
    covered: set[int] = set()
    for match in _TAG.finditer(text):
        covered.update(range(match.start(), match.end()))
    if any(index not in covered for index, char in enumerate(text) if char in "[]"):
        message = "an emotion tag is incomplete"
        raise ValueError(message)


def _span(text: str, weights: dict[str, float] | None) -> Span:
    spoken = text.strip()
    if not spoken:
        message = "an emotion tag has no words"
        raise ValueError(message)
    if weights is None:
        return Span(spoken, None)
    return Span(spoken, tuple(weights.items()))


def _add(weights: dict[str, float], name: str, scale: float) -> dict[str, float]:
    assert _directions is not None
    if name not in _directions:
        message = "emotion is not in the collection"
        raise ValueError(message)
    weights[name] = weights.get(name, 0.0) + scale
    return weights
