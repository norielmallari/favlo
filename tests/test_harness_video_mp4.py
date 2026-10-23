"""ffmpeg encode of one uncompressed AVI. It does not start serve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from harness.video_mp4 import first_frame_png, frame_png, write_mp4
from wan22_i2v.png import decode_png
from wan22_i2v.video import write_avi


def test_write_mp4_keeps_the_frame_geometry(tmp_path: Path) -> None:
    frames = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    frames[0, 4:12, 4:12] = (220, 30, 30)
    frames[2, 4:12, 4:12] = (30, 220, 30)
    frames[4, 4:12, 4:12] = (30, 30, 220)
    dest = tmp_path / "clip.mp4"
    saved = write_mp4(write_avi(frames, fps=16), dest)
    assert dest.is_file()
    assert saved == (5, 16, 16)
    first = decode_png(first_frame_png(dest))
    middle = decode_png(frame_png(dest, 2))
    last = decode_png(frame_png(dest, 4))
    assert int(first[6, 6, 0]) > 150
    assert int(middle[6, 6, 1]) > 150
    assert int(last[6, 6, 2]) > 150
