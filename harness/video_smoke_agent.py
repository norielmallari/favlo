"""Agent that turns the horse still into the smoke clip.

The tool names come from the command. The listener already published them.
A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers those names
and no others. The still is ``tests/assets/favlo-horse.png``. The pipeline
resizes it to the clip size. The run calls the first name with the prompt,
size, frames, and seed in ``docs/<pipeline>/smoke.json``. The MP4 is
``artifacts/video-smoke-agent.mp4``. The vision tower then reads frame 0,
the middle frame, and the last frame and must affirm the still sentence.
Exit 0 means each of those frames was affirmed and the file has the smoke
geometry. This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import urllib.request
from pathlib import Path
from typing import Any

from agent_framework import Agent

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.image_attachment import HORSE
from harness.image_smoke_agent import affirms, image_mcp, registered, smoke_note
from harness.mcp_media import published_name
from harness.server import close_listener, open_listener
from harness.video_mp4 import frame_png, save_mp4

HOST = "127.0.0.1"


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


def video_body(note: dict[str, Any], still: Path, model: str | None) -> dict[str, Any]:
    """One I2V body. ``video_prompt`` is the motion line when present."""
    prompt = note.get("video_prompt") or note["prompt"]
    body: dict[str, Any] = {
        "prompt": prompt,
        "image": _image_block(still),
        "size": f"{note['width']}x{note['height']}",
        "frames": note["frames"],
        "seed": note["seed"],
    }
    folder = model or note.get("model")
    if isinstance(folder, str) and folder:
        body["model"] = folder
    return body


async def create_video(
    base_url: str,
    port: int,
    names: list[str],
    body: dict[str, Any],
    model: str,
) -> bytes:
    clip: list[bytes] = []
    call = names[0]
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    async with image_mcp(port, names, clip) as mcp:
        registered(mcp, names)
        agent = Agent(
            client=client,
            name="video-smoke-agent",
            instructions=f"Call {call} once with the exact arguments. Then stop.",
            tools=mcp,
            default_options={"temperature": 0},
        )
        extra: dict[str, Any] = {"image": body["image"]}
        folder = body.get("model")
        if isinstance(folder, str) and folder:
            extra["model"] = folder
        await agent.run(
            _ask(call, body),
            options={
                "temperature": 0,
                "tool_choice": {"mode": "required", "required_function_name": call},
                "additional_function_arguments": extra,
            },
        )
    if not clip:
        message = "the agent did not generate a video"
        raise RuntimeError(message)
    return clip[-1]


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
    parser = argparse.ArgumentParser(description="Turn the horse still into the smoke clip.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pipeline", required=True, help="pipeline id")
    parser.add_argument(
        "--tool",
        action="append",
        required=True,
        dest="tools",
        help="local tool name the pipeline already publishes",
    )
    parser.add_argument(
        "--judge",
        default=None,
        help="chat model id; default is the registry default",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="registry folder; omit for the note's model or the host default",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    note = smoke_note(root, args.pipeline)
    names = [published_name(args.pipeline, tool) for tool in args.tools]
    still = root / HORSE
    if not still.is_file():
        print(f"missing {HORSE.as_posix()}", flush=True)
        return 1
    body = video_body(note, still, args.model)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        print(f"video on {port}", flush=True)
        blob = asyncio.run(create_video(base, port, names, body, judge_id))
        out = root / "artifacts" / "video-smoke-agent.mp4"
        saved = save_mp4(blob, out)
        print(f"wrote {out} frames {saved[0]} size {saved[1]}x{saved[2]}", flush=True)
        expected = (int(note["frames"]), int(note["width"]), int(note["height"]))
        if saved != expected:
            return 1
        sentence = str(note["prompt"])
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


def _ask(tool: str, body: dict[str, Any]) -> str:
    lines = [f"Call {tool} once with these arguments, then stop."]
    for key in ("prompt", "size", "frames", "seed"):
        if key in body:
            lines.append(f"{key}: {body[key]}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
