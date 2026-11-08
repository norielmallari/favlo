"""Turn the horse still into two 240p clips, at low and max.

The still is ``tests/assets/favlo-horse.png``. The pipeline resizes it to
416×240. ``wan22_i2v__generate_video`` then runs twice, at ``quality`` ``low`` and
``max``, with no chat between them. Each result is read from
``favlo://videos/{id}.mp4`` and saved as ``artifacts/video-quality-low.mp4``
and ``artifacts/video-quality-max.mp4``. Exit 0 means both files are 5 frames
at 416×240 and the max file is larger. This command is not part of pytest.
The 320p clip stays ``harness.video_smoke_agent``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from harness.image_attachment import HORSE
from harness.mcp_media import generate, read_link
from harness.server import close_listener, open_listener
from harness.video_mp4 import save_mp4
from harness.video_smoke_agent import _image_block, smoke_paths

FRAMES = 5
WIDTH = 416
HEIGHT = 240
QUALITIES = ("low", "max")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn the horse still into low and max clips.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    video_note = smoke_paths(root)[1]
    still = root / HORSE
    if not still.is_file():
        print(f"missing {HORSE.as_posix()}", flush=True)
        return 1
    video_body = {
        "model": video_note["model"],
        "prompt": video_note["video_prompt"],
        "size": f"{WIDTH}x{HEIGHT}",
        "frames": FRAMES,
        "seed": video_note["seed"],
    }
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        sizes: dict[str, int] = {}
        for quality in QUALITIES:
            print(f"video {quality}", flush=True)
            try:
                blob = _clip(port, video_body, still, quality)
            except RuntimeError as exc:
                print(exc, flush=True)
                return 1
            out = root / "artifacts" / f"video-quality-{quality}.mp4"
            saved = save_mp4(blob, out)
            sizes[quality] = out.stat().st_size
            frames, width, height = saved
            print(
                f"wrote {out.name} frames {frames} size {width}x{height} bytes {sizes[quality]}",
                flush=True,
            )
            if saved != (FRAMES, WIDTH, HEIGHT):
                return 1
        if sizes["max"] <= sizes["low"]:
            print("max is not larger than low", flush=True)
            return 1
        print("max is larger", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


def _clip(port: int, body: dict[str, Any], still: Path, quality: str) -> bytes:
    result = generate(
        port,
        "generate_video",
        {
            "model": body["model"],
            "prompt": body["prompt"],
            "image": _image_block(still),
            "size": body["size"],
            "frames": body["frames"],
            "seed": body["seed"],
            "quality": quality,
        },
    )
    return read_link(port, result, "video/mp4")


if __name__ == "__main__":
    raise SystemExit(main())
