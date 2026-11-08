"""Vision prompt expansion for qwen35-4b. No device tower."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41" / "qwen35-4b"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.vision import bind_vision, merged_grid, pack_image  # noqa: E402


def test_merged_grid_matches_the_pack() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    _packed, grid_h, grid_w = pack_image(image, min_pixels=32 * 32, max_pixels=32 * 32)
    assert merged_grid(image, min_pixels=32 * 32, max_pixels=32 * 32) == (
        grid_h // 2,
        grid_w // 2,
    )


class _Upload:
    def __init__(self) -> None:
        self.blobs: list[bytes] = []

    def upload(self, blob: bytes) -> bytes:
        self.blobs.append(blob)
        return blob


class _Log:
    def __init__(self) -> None:
        self.notes: list[str] = []

    def log(self, text: str) -> None:
        self.notes.append(text)


def test_a_covered_image_skips_the_tower(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tower")

    monkeypatch.setattr("qwen35_4b.vision._vision_row", fail)
    console = _Log()
    runtime = type("Runtime", (), {})()
    runtime.console = console
    runtime._vision_plan = [(b"png", 1, 2, 4, 2, 3)]
    runtime._vision_rows = object()
    runtime._vision_index = object()
    bind_vision(runtime, 4 + 6)
    assert console.notes == ["vision cached 6"]
    assert runtime._vision_rows is None
    assert runtime._vision_index is None


def test_an_image_past_the_hit_is_the_only_one_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bytes] = []

    def row(_runtime: object, blob: bytes, **_kwargs: object) -> tuple[np.ndarray, int, int]:
        calls.append(blob)
        return np.ones((2, 4), dtype=np.float32), 2, 4

    monkeypatch.setattr("qwen35_4b.vision._vision_row", row)
    console = _Log()
    runtime = type("Runtime", (), {})()
    runtime.console = console
    runtime._vision_plan = [
        (b"kept", 1, 2, 0, 2, 2),
        (b"new", 1, 2, 8, 1, 2),
    ]
    runtime._rope_length = 12
    runtime.session = _Upload()
    bind_vision(runtime, 8)
    assert calls == [b"new"]
    assert console.notes == ["vision cached 4"]
    index = np.frombuffer(runtime.session.blobs[1], dtype=np.uint32)
    assert index.tolist()[:8] == [0xFFFFFFFF] * 8
    assert index[8] == 0
    assert index[9] == 1
