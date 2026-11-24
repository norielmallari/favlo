"""Matmul tiles run on every pytest. Set WAN22_SMOKE=1 to generate the 320p clip."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "wan22_i2v" / "smoke.json"
OUT = ROOT / "artifacts" / "wan22_i2v-smoke.mp4"
_PIPELINE = "wan22_i2v"


def _installed_record() -> Path | None:
    from favlo.registry import choose_complete

    try:
        path, _selection = choose_complete(
            model=None, record=None, pipeline=_PIPELINE
        )
    except LookupError:
        return None
    if not (path / "manifest.json").is_file():
        return None
    return path


def test_kernels() -> None:
    from wan22_i2v.metal41.commands import kernels

    assert kernels() == 0


@pytest.mark.skipif(
    os.environ.get("WAN22_SMOKE") != "1",
    reason="set WAN22_SMOKE=1 to generate",
)
def test_shortest_320p_video() -> None:
    record = _installed_record()
    if record is None:
        pytest.fail("packed record is not present")
    from harness.video_mp4 import write_mp4
    from wan22_i2v.metal41.runtime import Runtime
    from wan22_i2v.metal41.video import avi_meta

    note = json.loads(NOTE.read_text(encoding="utf-8"))
    frames = int(note["frames"])
    width = int(note["width"])
    height = int(note["height"])
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[80:200, 200:360] = (220, 30, 30)
    runtime = Runtime(record)
    try:
        avi, timing = runtime.generate(
            str(note["prompt"]),
            image,
            frames,
            width,
            height,
            int(note["seed"]),
        )
    finally:
        runtime.close()
    assert avi_meta(avi) == (frames, width, height)
    assert len(timing["steps"]) == 4
    assert write_mp4(avi, OUT) == (frames, width, height)
