"""Agent that generates one spoken line through MCP for one pipeline.

The tool names come from the command. The listener already published them.
A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers those names
and no others. The body comes from ``docs/<pipeline>/smoke.json``. The saved
file is checked for the note's sample rate, mono, a level above the silence
floor, and a duration inside that note's cap. The command does not take the
machine lock. This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path
from typing import Any

import numpy as np
from agent_framework import Agent

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.image_smoke_agent import image_mcp, registered, smoke_note
from harness.mcp_media import published_name
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
RMS_FLOOR = 1e-4


def _report(path: Path) -> str:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        pcm = np.frombuffer(handle.readframes(frames), dtype="<i2").astype(np.float32) / 32768.0
    seconds = frames / rate if rate else 0.0
    rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
    return f"rate {rate}\nchannels {channels}\nseconds {seconds:.3f}\nrms {rms:.6f}\n"


def report_ok(text: str, *, rate: int, max_seconds: float) -> bool:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key in {"rate", "channels", "seconds", "rms"} and value:
            fields[key] = value.split()[0]
    try:
        got_rate = int(fields["rate"])
        channels = int(fields["channels"])
        duration = float(fields["seconds"])
        rms = float(fields["rms"])
    except (KeyError, ValueError):
        return False
    return (
        channels == 1
        and got_rate == rate
        and 0.0 < duration <= max_seconds
        and rms > RMS_FLOOR
    )


async def create_voice(
    base_url: str,
    port: int,
    names: list[str],
    body: dict[str, Any],
    model: str,
    out: Path,
) -> str:
    wav: list[bytes] = []
    call = names[0]
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    async with image_mcp(port, names, wav) as mcp:
        registered(mcp, names)
        agent = Agent(
            client=client,
            name="voice-smoke-agent",
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
    if not wav:
        message = "the agent did not generate voice"
        raise RuntimeError(message)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_bytes(wav[-1])
    temporary.replace(out)
    return _report(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one spoken line through MCP.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pipeline", required=True, help="pipeline id")
    parser.add_argument(
        "--tool",
        action="append",
        required=True,
        dest="tools",
        help="local tool name the pipeline already publishes",
    )
    parser.add_argument("--judge", default=None, help="chat model id; default is the registry default")
    parser.add_argument(
        "--model",
        default=None,
        help="registry folder; omit for the note's model or the host default",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    note = smoke_note(root, args.pipeline)
    names = [published_name(args.pipeline, tool) for tool in args.tools]
    body: dict[str, Any] = {"input": note["input"], "response_format": "wav"}
    folder = args.model or note.get("model")
    if isinstance(folder, str) and folder:
        body["model"] = folder
    instructions = note.get("instructions")
    if isinstance(instructions, str) and instructions:
        body["instructions"] = instructions
    rate = int(note["sample_rate"])
    max_seconds = float(note["max_seconds"])
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        out = root / "artifacts" / "voice-smoke-agent.wav"
        print("voice", flush=True)
        model = args.judge or judge_model(root)
        report = asyncio.run(
            create_voice(f"http://{HOST}:{port}/v1", port, names, body, model, out)
        )
        print(report, end="", flush=True)
        if not report_ok(report, rate=rate, max_seconds=max_seconds):
            print("voice file failed the format check", flush=True)
            return 1
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


def _ask(tool: str, body: dict[str, Any]) -> str:
    lines = [f"Call {tool} once with these arguments, then stop."]
    for key in ("input", "instructions", "response_format"):
        if key in body:
            lines.append(f"{key}: {body[key]}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
