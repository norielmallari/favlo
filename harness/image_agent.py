"""Agent that generates one image and checks it with the 35B vision tower.

The Microsoft Agent Framework agent calls ``POST /v1/images/generations``.
The next turn sends that PNG back to the same model and asks whether the
picture matches the prompt. Exit 0 means the reply affirms it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import socket
import urllib.request
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field

JUDGE = "qwen36-35b-a3b:instruct"
IMAGE_MODEL = "Z-Image-Turbo-fp4"
HOST = "127.0.0.1"
PORT = 8129
QUESTION = "Does this picture show a red cube on a white table? Reply with one word: yes or no."


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
        payload = post_json(port, "/v1/images/generations", body)
        encoded = payload["data"][0]["b64_json"]
        png.append(base64.b64decode(encoded))
        return "image ready"

    return [generate_image]


async def create_image(base_url: str, port: int, body: dict[str, Any]) -> bytes:
    png: list[bytes] = []
    client = OpenAIChatCompletionClient(model=JUDGE, api_key="favlo", base_url=base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="image-agent",
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


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, port)) == 0


def _hold_machine(root: Path) -> bool:
    from favlo.smoke_lock import hold

    return hold(root=root, pipeline="qwen36-35b-a3b", command="image-agent")


def _open_server(root: Path, was_up: bool) -> tuple[int, Any]:
    from harness.benchmarks.serve import free_port, start_server

    if was_up:
        port = free_port()
        pid, stop, _log = start_server(port)
        return port, (pid, stop)
    start_server(PORT)
    return PORT, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an image and check it.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    body = smoke_body(root)
    was_up = _listening(PORT)
    owned = None
    created = False
    from favlo.cli import repo_root
    from favlo.smoke_lock import release

    try:
        created = _hold_machine(repo_root())
        port, owned = _open_server(root, was_up)
        base = f"http://{HOST}:{port}/v1"
        print(f"generating on {port}", flush=True)
        png = asyncio.run(create_image(base, port, body))
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "image-agent.png").write_bytes(png)
        print("checking", flush=True)
        reply = judge(port, png)
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
        if created:
            release(root=repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
