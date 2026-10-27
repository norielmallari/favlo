"""Agent that draws a still, then the shortest 320p clip.

The Microsoft Agent Framework agent calls ``generate_image`` when the still is
missing. The still is drawn at the clip's width and height. The next turn
calls ``generate_video_i2v`` and reads ``favlo://videos/{id}.mp4`` into
``artifacts/video-smoke-agent.mp4``. The 35B
vision tower then reads frame 0, the middle frame, and the last frame of
that file and must affirm the still sentence. Exit 0 means each of those
frames was affirmed and the file has the smoke.json frame count at 576×320.
This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import urllib.request
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, tool
from pydantic import Field

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.image_smoke_agent import affirms
from harness.mcp_media import generate, read_link
from harness.server import close_listener, open_listener
from harness.video_mp4 import frame_png, save_mp4

HOST = "127.0.0.1"


def smoke_paths(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    image = json.loads((root / "docs" / "z-image-turbo" / "smoke.json").read_text(encoding="utf-8"))
    video = json.loads((root / "docs" / "wan22-i2v" / "smoke.json").read_text(encoding="utf-8"))
    return image, video


def post_json(port: int, route: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://{HOST}:{port}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=None) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        message = "response is not an object"
        raise RuntimeError(message)
    return payload


def _image_block(path: Path) -> dict[str, str]:
    return {
        "type": "image",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "mimeType": "image/png",
    }


def video_tools(
    port: int,
    image_body: dict[str, Any],
    video_body: dict[str, Any],
    still: Path,
    clip: list[bytes],
) -> list[object]:
    prompt = str(image_body["prompt"])

    @tool(approval_mode="never_require")
    def ensure_image(
        prompt_text: Annotated[str, Field(description="Exact sentence to draw.")],
    ) -> str:
        """Draw the still when the file is missing."""
        if still.is_file():
            return "image exists"
        if prompt_text.strip() != prompt:
            return f"use this prompt exactly: {prompt}"
        result = generate(port, "generate_image", image_body)
        still.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(read_link(port, result, "image/png"))
        return "image ready"

    @tool(approval_mode="never_require")
    def generate_video() -> str:
        """Generate the 320p clip from the still."""
        if not still.is_file():
            return "image is missing"
        result = generate(
            port,
            "generate_video_i2v",
            {
                "model": video_body["model"],
                "prompt": video_body["prompt"],
                "image": _image_block(still),
                "size": video_body["size"],
                "frames": video_body["frames"],
                "seed": video_body["seed"],
            },
        )
        clip.append(read_link(port, result, "video/mp4"))
        return "video ready"

    return [ensure_image, generate_video]


def _agent(
    base_url: str,
    tools: list[object],
    name: str,
    text: str,
    model: str,
    iterations: int = 2,
) -> int:
    try:
        asyncio.run(_run(base_url, tools, name, text, model, iterations))
    except Exception as exc:
        print(exc, flush=True)
        return 1
    return 0


async def _run(
    base_url: str, tools: list[object], name: str, text: str, model: str, iterations: int = 2
) -> None:
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = iterations
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="video-smoke-agent",
        instructions="Call the named tool once. Then stop.",
        tools=tools,
        default_options={"temperature": 0},
    )
    await agent.run(
        text,
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": name},
        },
    )


def check_still(port: int, png: bytes, sentence: str, model: str) -> str:
    """Ask the vision tower whether the still shows ``sentence``."""
    encoded = base64.b64encode(png).decode("ascii")
    question = f"Does this picture show {sentence}? Reply with one word: yes or no."
    payload = post_json(
        port,
        "/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 8,
            "enable_thinking": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
        },
    )
    return str(payload["choices"][0]["message"].get("content") or "")


def _png_size(blob: bytes) -> tuple[int, int]:
    if len(blob) < 24 or not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return (0, 0)
    width, height = struct.unpack_from(">II", blob, 16)
    return int(width), int(height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw a still, then the 320p clip.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--judge",
        default=None,
        help="chat model id; default is the registry default",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    video_note = smoke_paths(root)[1]
    still = root / "artifacts" / "video-smoke-agent.png"
    image_body = {
        "model": "Z-Image-Turbo-fp4",
        "prompt": video_note["prompt"],
        "size": f"{video_note['width']}x{video_note['height']}",
        "n": 1,
        "seed": video_note["seed"],
    }
    video_body = {
        "model": video_note["model"],
        "prompt": video_note["video_prompt"],
        "size": f"{video_note['width']}x{video_note['height']}",
        "frames": video_note["frames"],
        "seed": video_note["seed"],
    }
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        print(f"image on {port}", flush=True)
        expected_size = (int(video_note["width"]), int(video_note["height"]))
        if still.is_file() and _png_size(still.read_bytes()) != expected_size:
            still.unlink()
        sentence = f"Create an image of this exact sentence: {image_body['prompt']}"
        image_tools = video_tools(port, image_body, video_body, still, [])
        image_failed = _agent(base, [image_tools[0]], "ensure_image", sentence, judge_id)
        if not still.is_file():
            if image_failed != 0:
                return 1
            print("image is missing", flush=True)
            return 1
        if _png_size(still.read_bytes()) != expected_size:
            print(f"image is not {expected_size[0]}x{expected_size[1]}", flush=True)
            return 1
        clip: list[bytes] = []
        video_only = video_tools(port, image_body, video_body, still, clip)
        print("video", flush=True)
        failed = _agent(
            base,
            [video_only[1]],
            "generate_video",
            f"Generate the {video_note['frames']}-frame 320p video from the image.",
            judge_id,
            iterations=1,
        )
        if not clip:
            if failed != 0:
                return 1
            print("video is missing", flush=True)
            return 1
        out = root / "artifacts" / "video-smoke-agent.mp4"
        saved = save_mp4(clip[-1], out)
        print(f"wrote {out} frames {saved[0]} size {saved[1]}x{saved[2]}", flush=True)
        expected = (
            video_note["frames"],
            video_note["width"],
            video_note["height"],
        )
        if saved != expected:
            return 1
        sentence = str(video_note["prompt"])
        seen: list[int] = []
        for index in (0, (saved[0] - 1) // 2, saved[0] - 1):
            if index in seen:
                continue
            seen.append(index)
            print(f"checking frame {index}", flush=True)
            reply = check_still(port, frame_png(out, index), sentence, judge_id)
            print(reply, flush=True)
            if not affirms(reply):
                print("frame does not match", flush=True)
                return 1
        print("frames match", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
