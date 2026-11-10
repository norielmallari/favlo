"""Agent for one TI2V pipeline's text clip and still clip.

The tool names come from the command. Pass the text tool first and the still
tool second. A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers
those published names and no others. The text clip omits the still. The still
clip passes ``tests/assets/favlo-horse.png``. Sizes and frames come from
``docs/<pipeline>/smoke.json``. Exit 0 means both files match that note and
the vision tower affirms frame 0, the middle frame, and the last frame of the
still clip. This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
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
from harness.video_smoke_agent import _image_block

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


async def create_clip(
    base_url: str,
    port: int,
    name: str,
    body: dict[str, Any],
    model: str,
    *,
    text: str,
) -> bytes:
    clip: list[bytes] = []
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    async with image_mcp(port, [name], clip) as mcp:
        registered(mcp, [name])
        agent = Agent(
            client=client,
            name="video-ti2v-agent",
            instructions=f"Call {name} once with the exact arguments. Then stop.",
            tools=mcp,
            default_options={"temperature": 0},
        )
        options: dict[str, Any] = {
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": name},
        }
        extra: dict[str, Any] = {}
        if "image" in body:
            extra["image"] = body["image"]
        folder = body.get("model")
        if isinstance(folder, str) and folder:
            extra["model"] = folder
        if extra:
            options["additional_function_arguments"] = extra
        await agent.run(_ask(name, body), options=options)
    if not clip:
        message = "the agent did not generate a video"
        raise RuntimeError(message)
    return clip[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Text clip and still clip for one TI2V pipeline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pipeline", required=True, help="pipeline id")
    parser.add_argument(
        "--tool",
        action="append",
        required=True,
        dest="tools",
        help="local tool name; pass text tool first, still tool second",
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
    if len(args.tools) < 2:
        print("pass the text tool and the still tool", flush=True)
        return 1
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    card = smoke_note(root, args.pipeline)
    names = [published_name(args.pipeline, tool) for tool in args.tools]
    still = root / HORSE
    if not still.is_file():
        print(f"missing {HORSE.as_posix()}", flush=True)
        return 1
    text_size = f"{card['width']}x{card['height']}"
    still_size = f"{card['i2v_width']}x{card['i2v_height']}"
    folder = args.model or card.get("model")
    base_body: dict[str, Any] = {
        "prompt": card["prompt"],
        "size": text_size,
        "frames": card["frames"],
        "seed": card["seed"],
        "steps": card.get("steps", 4),
    }
    if isinstance(folder, str) and folder:
        base_body["model"] = folder
    still_body = dict(base_body)
    still_body["size"] = still_size
    still_body["image"] = _image_block(still)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        print("text video", flush=True)
        text_blob = asyncio.run(
            create_clip(base, port, names[0], base_body, judge_id, text="Generate the text video.")
        )
        print("still video", flush=True)
        still_blob = asyncio.run(
            create_clip(base, port, names[1], still_body, judge_id, text="Generate the still video.")
        )
        text_expected = (int(card["frames"]), int(card["width"]), int(card["height"]))
        still_expected = (int(card["frames"]), int(card["i2v_width"]), int(card["i2v_height"]))
        text_path = root / "artifacts" / "video-ti2v-t2v.mp4"
        still_path = root / "artifacts" / "video-ti2v-agent.mp4"
        if save_mp4(text_blob, text_path) != text_expected:
            print("text clip geometry does not match", flush=True)
            return 1
        if save_mp4(still_blob, still_path) != still_expected:
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


def _ask(tool: str, body: dict[str, Any]) -> str:
    lines = [f"Call {tool} once with these arguments, then stop."]
    for key in ("prompt", "size", "frames", "seed", "steps"):
        if key in body:
            lines.append(f"{key}: {body[key]}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
