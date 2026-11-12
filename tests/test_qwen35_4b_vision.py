"""Vision prompt expansion for qwen35_4b. No device tower."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41" / "qwen35_4b"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.metal41.vision import bind_vision, merged_grid, pack_image  # noqa: E402


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

    monkeypatch.setattr("qwen35_4b.metal41.vision._vision_row", fail)
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

    monkeypatch.setattr("qwen35_4b.metal41.vision._vision_row", row)
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


def test_row_file_is_the_only_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from favlo import vision_cache
    from qwen35_4b.metal41 import vision
    from qwen35_4b.metal41.chat import Pipeline
    from qwen35_4b.metal41.vision import release_vision

    monkeypatch.setattr(vision_cache, "cache_dir", lambda pipeline: tmp_path / pipeline)
    assert not hasattr(vision, "_ROW_CACHE")
    key = ("record", b"\x11" * 32, 32, 64)
    rows = np.arange(6, dtype=np.float32).reshape(3, 2)
    vision._store_vision_disk(key, rows, 4, 6)
    loaded = vision._load_vision_disk(key)
    assert loaded is not None
    embedded, grid_h, grid_w = loaded
    assert embedded.shape == (3, 2)
    assert float(embedded[2, 1]) == 5.0
    assert grid_h == 4 and grid_w == 6
    assert list((tmp_path / "qwen35_4b").glob("*.npz"))
    assert Pipeline().has_vision() is True
    runtime = type("Runtime", (), {})()
    runtime._vision_rows = object()
    runtime._vision_index = object()
    runtime._rope_device = object()
    runtime._rope_length = 4
    runtime._rope_delta = 1
    runtime._vision_plan = [object()]
    release_vision(runtime)
    assert runtime._vision_rows is None
    assert runtime._vision_index is None
    assert runtime._rope_device is None
    assert runtime._rope_length == 0
    assert runtime._vision_plan is None
