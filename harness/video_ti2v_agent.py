"""Agent for the Wan2.2 TI2V-5B text clip and still clip.

The text clip omits the still and uses ``width`` and ``height``. The still
clip draws one PNG at ``i2v_width`` and ``i2v_height``, then passes that PNG
to ``generate_video_t2v`` and ``generate_video_i2v``. Exit 0 means both files match
``docs/wan22-ti2v-5b/smoke.json`` and the vision tower affirms frame
0, the middle frame, and the last frame of the still clip. This command is
not part of pytest.
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


def smoke_card(root: Path) -> dict[str, Any]:
    path = root / "docs" / "wan22-ti2v-5b" / "smoke.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
    text_clip: list[bytes],
    still_clip: list[bytes],
    still_size: str | None = None,
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
    def generate_text_video() -> str:
        """Generate the text clip. Do not pass an image."""
        result = generate(port, "generate_video_t2v", dict(video_body))
        text_clip.append(read_link(port, result, "video/mp4"))
        return "text video ready"

    @tool(approval_mode="never_require")
    def generate_still_video() -> str:
        """Generate the clip from the still."""
        if not still.is_file():
            return "image is missing"
        body = dict(video_body)
        if still_size is not None:
            body["size"] = still_size
        body["image"] = _image_block(still)
        result = generate(port, "generate_video_i2v", body)
        still_clip.append(read_link(port, result, "video/mp4"))
        return "still video ready"

    return [ensure_image, generate_text_video, generate_still_video]


def _agent(base_url: str, tools: list[object], name: str, text: str, model: str) -> int:
    try:
        asyncio.run(_run(base_url, tools, name, text, model))
    except Exception as exc:
        print(exc, flush=True)
        return 1
    return 0


async def _run(base_url: str, tools: list[object], name: str, text: str, model: str) -> None:
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="video-ti2v-agent",
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


def check_frame(port: int, png: bytes, sentence: str, model: str) -> str:
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
    parser = argparse.ArgumentParser(description="Text clip and still clip for TI2V-5B.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--judge",
        default=None,
        help="chat model id; default is the registry default",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    card = smoke_card(root)
    still = root / "artifacts" / "video-ti2v-agent.png"
    text_size = f"{card['width']}x{card['height']}"
    still_size = f"{card['i2v_width']}x{card['i2v_height']}"
    image_body = {
        "model": "Z-Image-Turbo-fp4",
        "prompt": card["prompt"],
        "size": still_size,
        "seed": card["seed"],
    }
    video_body = {
        "model": card["model"],
        "prompt": card["prompt"],
        "size": text_size,
        "frames": card["frames"],
        "seed": card["seed"],
        "steps": 4,
    }
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        expected_size = (int(card["i2v_width"]), int(card["i2v_height"]))
        if still.is_file() and _png_size(still.read_bytes()) != expected_size:
            still.unlink()
        tools = video_tools(port, image_body, video_body, still, [], [], still_size)
        sentence = f"Create an image of this exact sentence: {card['prompt']}"
        if _agent(base, [tools[0]], "ensure_image", sentence, judge_id) != 0:
            return 1
        if not still.is_file() or _png_size(still.read_bytes()) != expected_size:
            print("image is missing", flush=True)
            return 1
        text_clip: list[bytes] = []
        still_clip: list[bytes] = []
        tools = video_tools(
            port, image_body, video_body, still, text_clip, still_clip, still_size
        )
        print("text video", flush=True)
        if _agent(base, [tools[1]], "generate_text_video", "Generate the text video.", judge_id):
            return 1
        print("still video", flush=True)
        still_text = "Generate the still video."
        if _agent(base, [tools[2]], "generate_still_video", still_text, judge_id):
            return 1
        if not text_clip or not still_clip:
            print("video is missing", flush=True)
            return 1
        text_expected = (int(card["frames"]), int(card["width"]), int(card["height"]))
        still_expected = (int(card["frames"]), int(card["i2v_width"]), int(card["i2v_height"]))
        text_path = root / "artifacts" / "video-ti2v-t2v.mp4"
        still_path = root / "artifacts" / "video-ti2v-agent.mp4"
        if save_mp4(text_clip[-1], text_path) != text_expected:
            print("text clip geometry does not match", flush=True)
            return 1
        if save_mp4(still_clip[-1], still_path) != still_expected:
            print("still clip geometry does not match", flush=True)
            return 1
        print(f"wrote {still_path.name}", flush=True)
        seen: list[int] = []
        for index in (0, (still_expected[0] - 1) // 2, still_expected[0] - 1):
            if index in seen:
                continue
            seen.append(index)
            print(f"checking frame {index}", flush=True)
            reply = check_frame(port, frame_png(still_path, index), str(card["prompt"]), judge_id)
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
