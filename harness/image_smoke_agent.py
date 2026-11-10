"""Agent that generates the pictures named by a pipeline smoke note.

The tool names come from the command. The listener already published them.
A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers those names
and no others. Each run calls the first name once with the prompt, size, and
seed in ``docs/<pipeline>/smoke.json``. Quality stays at the pipeline default.
The next turn sends that PNG to the chat model.
Exit 0 means the reply affirms the picture and the PNG is RGB.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path
from typing import Any

from agent_framework import Agent, MCPStreamableHTTPTool

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.mcp_media import published_name
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"


def affirms(text: str) -> bool:
    """True when the first word is an affirmative."""
    for word in text.lower().split():
        token = "".join(ch for ch in word if ch.isalpha())
        if token:
            return token in {"yes", "yeah", "yep"}
    return False


def smoke_note(root: Path, pipeline: str) -> dict[str, Any]:
    """The request stored beside that pipeline's note."""
    path = root / "docs" / pipeline / "smoke.json"
    note = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(note, dict):
        message = "smoke.json is not an object"
        raise RuntimeError(message)
    return note


def smoke_runs(note: dict[str, Any], model: str | None) -> list[dict[str, Any]]:
    """One body. Quality stays at the pipeline default."""
    base: dict[str, Any] = {
        "prompt": note["prompt"],
        "size": f"{note['width']}x{note['height']}",
        "seed": note["seed"],
    }
    if model:
        base["model"] = model
    negative = note.get("negative_prompt")
    if isinstance(negative, str) and negative:
        base["negative_prompt"] = negative
    return [base]


def image_mcp(
    port: int,
    names: list[str],
    png: list[bytes],
) -> MCPStreamableHTTPTool:
    """Register exactly the published names the harness was given."""
    return MCPStreamableHTTPTool(
        name="favlo",
        url=f"http://{HOST}:{port}/mcp",
        load_prompts=False,
        approval_mode="never_require",
        allowed_tools=names,
        additional_tool_argument_names=["model", "image"],
        parse_tool_results=_capture(png),
        request_timeout=None,
    )


def registered(mcp: MCPStreamableHTTPTool, names: list[str]) -> None:
    """Raise when a passed name is not among the loaded functions."""
    found = {func.name for func in mcp.functions}
    missing = [name for name in names if name not in found]
    if missing:
        message = f"listener did not register {', '.join(missing)}"
        raise RuntimeError(message)


async def create_image(
    base_url: str,
    port: int,
    names: list[str],
    body: dict[str, Any],
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
            name="image-smoke-agent",
            instructions=f"Call {call} once with the exact arguments. Then stop.",
            tools=mcp,
            default_options={"temperature": 0},
        )
        options: dict[str, Any] = {
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": call},
        }
        folder = body.get("model")
        if isinstance(folder, str) and folder:
            options["additional_function_arguments"] = {"model": folder}
        await agent.run(_ask(call, body), options=options)
    if not png:
        message = "the agent did not generate an image"
        raise RuntimeError(message)
    return png[-1]


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


def judge(port: int, png: bytes, model: str, question: str) -> str:
    """One yes or no. Thinking stays off, and the reply stops at 256 tokens."""
    encoded = base64.b64encode(png).decode("ascii")
    payload = post_json(
        port,
        "/v1/responses",
        {
            "model": model,
            "temperature": 0,
            "max_output_tokens": 256,
            "reasoning": {"effort": "none"},
            "enable_thinking": False,
            "store": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encoded}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
        },
    )
    return _reply_text(payload)


def png_color_type(blob: bytes) -> int:
    """IHDR color type. 2 is RGB. 6 is RGBA."""
    if len(blob) < 26 or blob[:8] != b"\x89PNG\r\n\x1a\n":
        message = "not a PNG"
        raise ValueError(message)
    return blob[25]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the smoke pictures and check them.")
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
        help="chat model id for generation and vision checks; default is the registry default",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="registry folder; omit for the pipeline_default row, else the first folder",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    note = smoke_note(root, args.pipeline)
    names = [published_name(args.pipeline, tool) for tool in args.tools]
    runs = smoke_runs(note, args.model)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        base = f"http://{HOST}:{port}/v1"
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        request = runs[0]
        print(f"generating on {port}", flush=True)
        png = asyncio.run(create_image(base, port, names, request, judge_id))
        out = root / "artifacts" / "image-smoke-agent.png"
        out.write_bytes(png)
        print(f"wrote {out.name} bytes {out.stat().st_size}", flush=True)
        if png_color_type(png) != 2:
            print(f"{out.name} is not RGB", flush=True)
            return 1
        print(f"checking {out.name}", flush=True)
        reply = judge(port, png, judge_id, _question(note))
        print(reply, flush=True)
        if not affirms(reply):
            print(f"{out.name} does not match", flush=True)
            return 1
        print("image matches", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


def _reply_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                parts.append(str(block.get("text") or ""))
    return "".join(parts)


def _question(note: dict[str, Any]) -> str:
    question = note.get("question")
    if isinstance(question, str) and question.strip():
        return question
    return (
        "Does this picture match this description: "
        f"{note['prompt']}? Reply with one word: yes or no."
    )


def _ask(tool: str, body: dict[str, Any]) -> str:
    lines = [f"Call {tool} once with these arguments, then stop."]
    for key in ("prompt", "negative_prompt", "size", "seed"):
        if key in body:
            lines.append(f"{key}: {body[key]}")
    return "\n".join(lines)


def _capture(png: list[bytes]):
    def parse(result: Any) -> str:
        content = getattr(result, "content", None) or []
        texts = [block.text for block in content if isinstance(getattr(block, "text", None), str)]
        packed = "\n".join(texts)
        raw = _png_from_text(packed)
        if raw is not None:
            png.append(raw)
        return packed or "done"

    return parse


def _png_from_text(text: str) -> bytes | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    rows = parsed if isinstance(parsed, list) else [parsed]
    for row in rows:
        if not isinstance(row, dict):
            continue
        uri = row.get("mediaUri")
        if isinstance(uri, str) and uri.startswith("file://"):
            from favlo.local_uri import read_file_uri

            return read_file_uri(uri)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
