"""Agent that draws a 240p still, then reads one clip at low and max.

The Microsoft Agent Framework agent calls the image route when the still is
missing. The still is 416×240. The next turn posts one 5-frame
``POST /v1/videos``. The harness then reads
``GET /v1/videos/{id}/content?quality=low`` and ``quality=max`` and saves
``artifacts/video-quality-low.mp4`` and ``artifacts/video-quality-max.mp4``.
Exit 0 means both files are 5 frames at 416×240 and the max file is larger.
This command is not part of pytest. The 320p clip stays ``harness.video_smoke_agent``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
from pathlib import Path
from typing import Annotated, Any

from agent_framework import tool
from pydantic import Field

from harness.video_mp4 import save_mp4
from harness.video_smoke_agent import (
    JUDGE,
    _listening,
    _open_server,
    _png_size,
    _run,
    get_bytes,
    get_json,
    post_json,
    smoke_paths,
)

FRAMES = 5
WIDTH = 416
HEIGHT = 240
QUALITIES = ("low", "max")


def video_tools(
    port: int,
    image_body: dict[str, Any],
    video_body: dict[str, Any],
    still: Path,
    ids: list[str],
) -> list[object]:
    prompt = str(image_body["prompt"])

    @tool(approval_mode="never_require")
    def ensure_image(
        prompt_text: Annotated[str, Field(description="Exact sentence to draw.")],
    ) -> str:
        """Draw the 240p still when the file is missing."""
        if still.is_file():
            return "image exists"
        if prompt_text.strip() != prompt:
            return f"use this prompt exactly: {prompt}"
        payload = post_json(port, "/v1/images/generations", image_body)
        still.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(base64.b64decode(payload["data"][0]["b64_json"]))
        return "image ready"

    @tool(approval_mode="never_require")
    def generate_video() -> str:
        """Generate the 5-frame 240p clip from the still."""
        if not still.is_file():
            return "image is missing"
        body = dict(video_body)
        body["input_reference"] = base64.b64encode(still.read_bytes()).decode("ascii")
        job = post_json(port, "/v1/videos", body)
        video_id = str(job["id"])
        while True:
            current = get_json(port, f"/v1/videos/{video_id}")
            status = current.get("status")
            if status == "completed":
                break
            if status == "failed":
                error = current.get("error")
                message = error.get("message") if isinstance(error, dict) else "video failed"
                raise RuntimeError(str(message or "video failed"))
            time.sleep(0.25)
        ids.append(video_id)
        return "video ready"

    return [ensure_image, generate_video]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw a 240p still, then low and max clips.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--judge", default=JUDGE)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    video_note = smoke_paths(root)[1]
    still = root / "artifacts" / "video-quality.png"
    image_body = {
        "model": "Z-Image-Turbo-fp4",
        "prompt": video_note["prompt"],
        "size": f"{WIDTH}x{HEIGHT}",
        "n": 1,
        "seed": video_note["seed"],
    }
    video_body = {
        "model": video_note["model"],
        "prompt": video_note["video_prompt"],
        "size": f"{WIDTH}x{HEIGHT}",
        "frames": FRAMES,
        "seed": video_note["seed"],
    }
    was_up = _listening(8129)
    owned = None
    created = False
    from favlo.cli import repo_root
    from favlo.smoke_lock import hold, release

    try:
        created = hold(root=repo_root(), pipeline="wan22-i2v", command="video-quality")
        port, owned = _open_server(was_up)
        base = f"http://127.0.0.1:{port}/v1"
        print(f"image on {port}", flush=True)
        if still.is_file() and _png_size(still.read_bytes()) != (WIDTH, HEIGHT):
            still.unlink()
        sentence = f"Create an image of this exact sentence: {image_body['prompt']}"
        ids: list[str] = []
        image_tools = video_tools(port, image_body, video_body, still, ids)
        asyncio.run(_run(base, [image_tools[0]], "ensure_image", sentence, args.judge))
        if not still.is_file():
            print("image is missing", flush=True)
            return 1
        if _png_size(still.read_bytes()) != (WIDTH, HEIGHT):
            print(f"image is not {WIDTH}x{HEIGHT}", flush=True)
            return 1
        print("video", flush=True)
        asyncio.run(
            _run(
                base,
                [image_tools[1]],
                "generate_video",
                f"Generate the {FRAMES}-frame 240p video from the image.",
                args.judge,
                iterations=1,
            )
        )
        if not ids:
            print("video is missing", flush=True)
            return 1
        sizes: dict[str, int] = {}
        for quality in QUALITIES:
            blob = get_bytes(port, f"/v1/videos/{ids[-1]}/content?quality={quality}")
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
        if owned is not None:
            from harness.benchmarks.serve import stop_server

            stop_server(owned[0], owned[1])
        if created:
            release(root=repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
