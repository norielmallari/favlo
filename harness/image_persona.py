"""Favlo draws a picture of himself through a published image tool.

The tool names come from the command. The listener already published them.
A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers those names
and no others. The agent instructions are the horse persona. The user asks
for a picture of that horse. The image prompt is the 2.5D mascot. ``--size``
sets the edge; when omitted, ``docs/<pipeline>/smoke.json`` width and height
are used. Exit 0 means the vision check affirms it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

from agent_framework import Agent

from favlo.chat_client import open_chat_client
from favlo.persona import PERSONA
from harness.harness_model import judge_model
from harness.image_smoke_agent import HOST, affirms, image_mcp, post_json, registered, smoke_note
from harness.mcp_media import published_name
from harness.server import close_listener, open_listener

IMAGE_PROMPT = (
    "A cute 2.5D stylized young horse, not a photograph and not a flat cartoon. "
    "Chestnut brown coat, a broad white blaze down the face, a white muzzle, "
    "and huge glossy blue eyes with long lashes and bright catchlights. "
    "Fluffy brown mane with a pale streak, white socks on the front legs, "
    "and a navy bandana at the neck. "
    "The horse sits at a wooden desk, both front hooves on the desk, head tilted, "
    "soft smile. Warm desk-lamp light, cool blue monitor glow, dim room, "
    "soft fur, polished semi-realistic illustration."
)
QUESTION = (
    "Does this picture show a cute stylized horse with large blue eyes and a white blaze,"
    " rather than a real horse? Reply with one word: yes or no."
)
ASK = "Draw a picture of yourself."


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


async def create_image(
    base_url: str,
    port: int,
    names: list[str],
    size: str,
    model: str,
) -> bytes:
    png: list[bytes] = []
    call = names[0]
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    async with image_mcp(port, names, png) as mcp:
        registered(mcp, names)
        agent = Agent(
            client=client,
            name="favlo",
            instructions=(
                PERSONA
                + f" When someone asks for a picture of you, call {call} once"
                + " with this prompt exactly, then stop: "
                + IMAGE_PROMPT
            ),
            tools=mcp,
            default_options={"temperature": 0},
        )
        await agent.run(
            ASK,
            options={
                "temperature": 0,
                "tool_choice": {"mode": "required", "required_function_name": call},
                "additional_function_arguments": {
                    "prompt": IMAGE_PROMPT,
                    "size": size,
                    "seed": 0,
                },
            },
        )
    if not png:
        message = "the agent did not generate an image"
        raise RuntimeError(message)
    return png[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Favlo draws himself and the vision model checks it.")
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
        "--size",
        default=None,
        help="WIDTHxHEIGHT; default is docs/<pipeline>/smoke.json width and height",
    )
    parser.add_argument("--judge", default=None, help="chat model id; default is the registry default")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    note = smoke_note(root, args.pipeline)
    names = [published_name(args.pipeline, tool) for tool in args.tools]
    size = args.size or f"{note['width']}x{note['height']}"
    model = args.judge or judge_model(root)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        print(f"generating on {port} size {size}", flush=True)
        png = asyncio.run(create_image(base, port, names, size, model))
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "favlo-horse.png").write_bytes(png)
        print(f"image bytes {len(png)}", flush=True)
        print("checking", flush=True)
        reply = judge(port, png, model)
        print(reply, flush=True)
        if affirms(reply):
            print("image matches", flush=True)
            return 0
        print("image does not match", flush=True)
        return 1
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
