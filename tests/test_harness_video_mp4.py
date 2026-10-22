"""ffmpeg encode of one uncompressed AVI. It does not start serve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from harness.video_mp4 import first_frame_png, write_mp4
from wan22_i2v.png import decode_png
from wan22_i2v.video import write_avi


def test_write_mp4_keeps_the_frame_geometry(tmp_path: Path) -> None:
    frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    frames[:, 4:12, 4:12] = (220, 30, 30)
    dest = tmp_path / "clip.mp4"
    saved = write_mp4(write_avi(frames, fps=16), dest)
    assert dest.is_file()
    assert saved == (5, 16, 16)
    frame = decode_png(first_frame_png(dest))
    assert int(frame[6, 6, 0]) > 150
    assert int(frame[6, 6, 2]) < 80
