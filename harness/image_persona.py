"""Favlo draws a picture of himself at the largest edge the machine can hold.

The agent instructions are the terminal persona. The user asks for a picture
of that horse. The image prompt is the 2.5D mascot: a chestnut horse with a
white blaze and large blue eyes, sitting at a desk. The image call uses
``MAX_EDGE`` on each side. Exit 0 means the 35B vision check affirms it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field
from z_image_turbo.scratch import MAX_EDGE

from favlo.persona import INSTRUCTIONS
from harness.image_agent import (
    HOST,
    IMAGE_MODEL,
    JUDGE,
    PORT,
    _hold_machine,
    _listening,
    _open_server,
    affirms,
    post_json,
)

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


def judge(port: int, png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    payload = post_json(
        port,
        "/v1/chat/completions",
        {
            "model": JUDGE,
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


def image_tools(port: int, png: list[bytes], timing: list[dict[str, Any]]) -> list[object]:
    @tool(approval_mode="never_require")
    def generate_image(
        prompt_text: Annotated[str, Field(description="What to draw.")],
    ) -> str:
        """Draw one 2.5D picture of the cute horse mascot."""
        if prompt_text.strip() != IMAGE_PROMPT:
            return f"use this prompt exactly: {IMAGE_PROMPT}"
        payload = post_json(
            port,
            "/v1/images/generations",
            {
                "model": IMAGE_MODEL,
                "prompt": IMAGE_PROMPT,
                "size": f"{MAX_EDGE}x{MAX_EDGE}",
                "n": 1,
                "seed": 0,
            },
        )
        png.append(base64.b64decode(payload["data"][0]["b64_json"]))
        timing.append(payload["generation_seconds"])
        return "image ready"

    return [generate_image]


async def create_image(
    base_url: str, port: int, png: list[bytes], timing: list[dict[str, Any]]
) -> None:
    client = OpenAIChatCompletionClient(model=JUDGE, api_key="favlo", base_url=base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="favlo",
        instructions=(
            INSTRUCTIONS
            + " When someone asks for a picture of you, call generate_image once"
            + " with this prompt exactly, then stop: "
            + IMAGE_PROMPT
        ),
        tools=image_tools(port, png, timing),
        default_options={"temperature": 0},
    )
    await agent.run(
        ASK,
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": "generate_image"},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Favlo draws himself and the 35B checks it.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    was_up = _listening(PORT)
    owned = None
    png: list[bytes] = []
    timing: list[dict[str, Any]] = []
    from favlo.cli import repo_root
    from favlo.smoke_lock import release

    _hold_machine(repo_root())
    try:
        port, owned = _open_server(root, was_up)
        base = f"http://{HOST}:{port}/v1"
        print(f"generating on {port} size {MAX_EDGE}", flush=True)
        asyncio.run(create_image(base, port, png, timing))
        if not png:
            print("the agent did not generate an image", flush=True)
            return 1
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "favlo-horse.png").write_bytes(png[-1])
        print(json.dumps(timing[-1]), flush=True)
        print("checking", flush=True)
        reply = judge(port, png[-1])
        print(reply, flush=True)
        if affirms(reply):
            print("image matches", flush=True)
            return 0
        print("image does not match", flush=True)
        return 1
    finally:
        if owned is not None:
            from harness.benchmarks.serve import stop_server

            stop_server(owned[0], owned[1])
        release(root=repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
