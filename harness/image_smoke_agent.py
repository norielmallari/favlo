"""Agent that generates two images and checks them with the 35B vision tower.

The Microsoft Agent Framework agent calls ``generate_image`` on ``POST /mcp``
at ``quality`` ``low``, then ``max``. Each later turn sends that PNG back to
the same model and asks whether the picture matches the prompt. Exit 0 means
both replies affirm it and the max file is larger.
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
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field

from harness.harness_model import judge_model
from harness.mcp_media import generate, read_link
from harness.server import close_listener, open_listener

IMAGE_MODEL = "Z-Image-Turbo-fp4"
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


def smoke_body(root: Path) -> dict[str, Any]:
    note = root / "docs" / "z-image-turbo" / "smoke.json"
    stored = json.loads(note.read_text(encoding="utf-8"))
    return {
        "model": IMAGE_MODEL,
        "prompt": stored["prompt"],
        "size": f"{stored['width']}x{stored['height']}",
        "n": 1,
        "seed": stored["seed"],
    }


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
    client = OpenAIChatCompletionClient(model=model, api_key="favlo", base_url=base_url)
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
            "max_tokens": 8,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a low and a max image and check both.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--judge",
        default=None,
        help="chat model id for generation and vision checks; default is the registry default",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    body = smoke_body(root)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}
        for quality in QUALITIES:
            print(f"generating {quality} on {port}", flush=True)
            request = {**body, "quality": quality}
            png = asyncio.run(create_image(base, port, request, judge_id))
            out = root / "artifacts" / f"image-smoke-agent-{quality}.png"
            out.write_bytes(png)
            saved[quality] = out
            print(f"wrote {out.name} bytes {out.stat().st_size}", flush=True)
        if saved["max"].stat().st_size <= saved["low"].stat().st_size:
            print("max quality is not larger", flush=True)
            return 1
        for quality in QUALITIES:
            print(f"checking {quality}", flush=True)
            reply = judge(port, saved[quality].read_bytes(), judge_id)
            print(reply, flush=True)
            if not affirms(reply):
                print(f"{quality} image does not match", flush=True)
                return 1
        print("images match", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
