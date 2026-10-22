"""Shortest 320p clip from the packed record. Set WAN22_SMOKE=1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WAN22_SMOKE") != "1",
    reason="set WAN22_SMOKE=1 to generate",
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "models" / "Wan2.2-I2V-A14B-fp4"
NOTE = ROOT / "docs" / "wan22-i2v" / "smoke.json"
OUT = ROOT / "artifacts" / "wan22-i2v-smoke.mp4"


def test_shortest_320p_video() -> None:
    if not (RECORD / "manifest.json").is_file():
        pytest.fail("packed record is not present")
    from harness.video_mp4 import write_mp4
    from wan22_i2v.commands import kernels
    from wan22_i2v.runtime import Runtime
    from wan22_i2v.video import avi_meta

    note = json.loads(NOTE.read_text(encoding="utf-8"))
    frames = int(note["frames"])
    width = int(note["width"])
    height = int(note["height"])
    kernels()
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[80:200, 200:360] = (220, 30, 30)
    runtime = Runtime(RECORD)
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
