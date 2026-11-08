"""Agent that generates two images and checks them with the vision tower.

The Microsoft Agent Framework agent calls ``generate_image`` on ``POST /mcp``
at ``quality`` ``low``, then ``max``. Pass ``--transparent`` to set MCP
``transparent`` true (BiRefNet after decode). Each later turn sends that PNG
back to the same model and asks whether the picture matches the prompt. Exit 0
means both replies affirm it and the max file is larger. With ``--transparent``,
each PNG must also be RGBA.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, tool
from pydantic import Field

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.mcp_media import generate, read_link
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
QUESTION = "Does this picture show a red cube on a white table? Reply with one word: yes or no."
QUALITIES = ("low", "max")


def affirms(text: str) -> bool:
    """True when the first word is an affirmative."""
    for word in text.lower().split():
        token = "".join(ch for ch in word if ch.isalpha())
        if token:
            return token in {"yes", "yeah", "yep"}
    return False


def smoke_body(root: Path, model: str | None = None) -> dict[str, Any]:
    """Prompt, size, and seed from smoke.json. Omit ``model`` for the registry default."""
    note = root / "docs" / "z-image-turbo" / "smoke.json"
    stored = json.loads(note.read_text(encoding="utf-8"))
    body: dict[str, Any] = {
        "prompt": stored["prompt"],
        "size": f"{stored['width']}x{stored['height']}",
        "seed": stored["seed"],
    }
    if model:
        body["model"] = model
    return body


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


def image_tools(port: int, body: dict[str, Any], png: list[bytes]) -> list[object]:
    prompt = str(body["prompt"])

    @tool(approval_mode="never_require")
    def generate_image(
        prompt_text: Annotated[str, Field(description="Exact sentence to draw.")],
    ) -> str:
        """Generate one image from the exact prompt."""
        if prompt_text.strip() != prompt:
            return f"use this prompt exactly: {prompt}"
        result = generate(port, "generate_image", body)
        png.append(read_link(port, result, "image/png"))
        return "image ready"

    return [generate_image]


async def create_image(
    base_url: str, port: int, body: dict[str, Any], model: str
) -> bytes:
    png: list[bytes] = []
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="image-smoke-agent",
        instructions="Call generate_image once with the exact prompt. Then stop.",
        tools=image_tools(port, body, png),
        default_options={"temperature": 0},
    )
    await agent.run(
        f"Create an image of this exact sentence: {body['prompt']}",
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": "generate_image"},
        },
    )
    if not png:
        message = "the agent did not generate an image"
        raise RuntimeError(message)
    return png[-1]


def judge(port: int, png: bytes, model: str) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    payload = post_json(
        port,
        "/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 16,
            "enable_thinking": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QUESTION},
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


def png_color_type(blob: bytes) -> int:
    """IHDR color type. 2 is RGB. 6 is RGBA."""
    if len(blob) < 26 or blob[:8] != b"\x89PNG\r\n\x1a\n":
        message = "not a PNG"
        raise ValueError(message)
    return blob[25]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a low and a max image and check both.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--judge",
        default=None,
        help="chat model id for generation and vision checks; default is the registry default",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="image model id; default is the registry media image default",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="pass transparent true so BiRefNet writes an RGBA PNG",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    body = smoke_body(root, args.model)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}
        suffix = "-transparent" if args.transparent else ""
        for quality in QUALITIES:
            label = f"{quality}{suffix}"
            print(f"generating {label} on {port}", flush=True)
            request = {**body, "quality": quality, "transparent": args.transparent}
            png = asyncio.run(create_image(base, port, request, judge_id))
            out = root / "artifacts" / f"image-smoke-agent-{quality}{suffix}.png"
            out.write_bytes(png)
            saved[quality] = out
            print(f"wrote {out.name} bytes {out.stat().st_size}", flush=True)
            if args.transparent and png_color_type(png) != 6:
                print(f"{out.name} is not RGBA", flush=True)
                return 1
            if not args.transparent and png_color_type(png) != 2:
                print(f"{out.name} is not RGB", flush=True)
                return 1
        if saved["max"].stat().st_size <= saved["low"].stat().st_size:
            print("max quality is not larger", flush=True)
            return 1
        for quality in QUALITIES:
            print(f"checking {quality}{suffix}", flush=True)
            reply = judge(port, saved[quality].read_bytes(), judge_id)
            print(reply, flush=True)
            if not affirms(reply):
                print(f"{quality}{suffix} image does not match", flush=True)
                return 1
        print("images match", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
