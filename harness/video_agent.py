"""Agent that draws a still, then the shortest 320p clip.

The Microsoft Agent Framework agent calls the image route when the still is
missing. The still is drawn at the clip's width and height. The next turn
posts `POST /v1/videos`, polls the job, and saves
`GET /v1/videos/{id}/content` to ``artifacts/video-agent.mp4``. The 35B
vision tower then reads frame 0 of that file and must affirm the still
sentence. Exit 0 means that frame was affirmed and the file has the
smoke.json frame count at 576×320. This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import socket
import struct
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field

from harness.image_agent import affirms
from harness.video_mp4 import first_frame_png, save_mp4

JUDGE = "qwen36-35b-a3b:instruct"
HOST = "127.0.0.1"
PORT = 8129


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


def get_json(port: int, route: str) -> dict[str, Any]:
    request = urllib.request.Request(f"http://{HOST}:{port}{route}")
    with urllib.request.urlopen(request, timeout=None) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        message = "response is not an object"
        raise RuntimeError(message)
    return payload


def get_bytes(port: int, route: str) -> bytes:
    request = urllib.request.Request(f"http://{HOST}:{port}{route}")
    with urllib.request.urlopen(request, timeout=None) as response:
        return response.read()


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
        payload = post_json(port, "/v1/images/generations", image_body)
        still.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(base64.b64decode(payload["data"][0]["b64_json"]))
        return "image ready"

    @tool(approval_mode="never_require")
    def generate_video() -> str:
        """Generate the 320p clip from the still."""
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
        clip.append(get_bytes(port, f"/v1/videos/{video_id}/content"))
        return "video ready"

    return [ensure_image, generate_video]


async def _run(
    base_url: str, tools: list[object], name: str, text: str, model: str, iterations: int = 2
) -> None:
    client = OpenAIChatCompletionClient(model=model, api_key="favlo", base_url=base_url)
    client.function_invocation_configuration["max_iterations"] = iterations
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="video-agent",
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


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, port)) == 0


def _hold_machine(root: Path) -> bool:
    from favlo.smoke_lock import hold

    return hold(root=root, pipeline="wan22-i2v", command="video-agent")


def _open_server(was_up: bool) -> tuple[int, Any]:
    from harness.benchmarks.serve import free_port, start_server

    if was_up:
        port = free_port()
        pid, stop, _log = start_server(port)
        return port, (pid, stop)
    start_server(PORT)
    return PORT, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw a still, then the 320p clip.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--judge", default=JUDGE)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    video_note = smoke_paths(root)[1]
    still = root / "artifacts" / "video-agent.png"
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
    was_up = _listening(PORT)
    owned = None
    created = False
    from favlo.cli import repo_root
    from favlo.smoke_lock import release

    try:
        created = _hold_machine(repo_root())
        port, owned = _open_server(was_up)
        base = f"http://{HOST}:{port}/v1"
        print(f"image on {port}", flush=True)
        expected_size = (int(video_note["width"]), int(video_note["height"]))
        if still.is_file() and _png_size(still.read_bytes()) != expected_size:
            still.unlink()
        sentence = f"Create an image of this exact sentence: {image_body['prompt']}"
        image_tools = video_tools(port, image_body, video_body, still, [])
        asyncio.run(_run(base, [image_tools[0]], "ensure_image", sentence, args.judge))
        if not still.is_file():
            print("image is missing", flush=True)
            return 1
        if _png_size(still.read_bytes()) != expected_size:
            print(f"image is not {expected_size[0]}x{expected_size[1]}", flush=True)
            return 1
        clip: list[bytes] = []
        video_only = video_tools(port, image_body, video_body, still, clip)
        print("video", flush=True)
        asyncio.run(
            _run(
                base,
                [video_only[1]],
                "generate_video",
                f"Generate the {video_note['frames']}-frame 320p video from the image.",
                args.judge,
                iterations=1,
            )
        )
        if not clip:
            print("video is missing", flush=True)
            return 1
        out = root / "artifacts" / "video-agent.mp4"
        saved = save_mp4(clip[-1], out)
        print(f"wrote {out} frames {saved[0]} size {saved[1]}x{saved[2]}", flush=True)
        expected = (
            video_note["frames"],
            video_note["width"],
            video_note["height"],
        )
        if saved != expected:
            return 1
        print("checking frame 0", flush=True)
        reply = check_still(port, first_frame_png(out), str(video_note["prompt"]), args.judge)
        print(reply, flush=True)
        if not affirms(reply):
            print("frame does not match", flush=True)
            return 1
        print("frame matches", flush=True)
        return 0
    finally:
        if owned is not None:
            from harness.benchmarks.serve import stop_server

            stop_server(owned[0], owned[1])
        if created:
            release(root=repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
