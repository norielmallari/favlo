"""Agent that generates one spoken line through MCP.

The agent calls ``generate_voice`` on ``POST /mcp`` and reads
``favlo://voice/{id}.wav``. The saved file is checked for the sample rate
in ``docs/voxcpm2/smoke.json``, mono, a level above the silence floor, and
a duration inside that note's cap. The command does not take the machine
lock. This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from harness.mcp_media import generate, read_link
from harness.server import close_listener, open_listener

JUDGE = "qwen36-35b-a3b:instruct"
HOST = "127.0.0.1"
RMS_FLOOR = 1e-4


def voice_tool(port: int, body: dict[str, Any], out: Path, reports: list[str]) -> object:
    @tool(approval_mode="never_require")
    def generate_voice() -> str:
        """Speak the smoke sentence. The harness inspects the file."""
        result = generate(port, "generate_voice", {**body, "response_format": "wav"})
        payload = read_link(port, result, "audio/wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(out)
        reports.append(_report(out))
        return "voice ready"

    return generate_voice


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


async def _run(base_url: str, voice: object) -> None:
    client = OpenAIChatCompletionClient(model=JUDGE, api_key="favlo", base_url=base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="voice-smoke-agent",
        instructions="Call generate_voice once. Then stop.",
        tools=[voice],
        default_options={"temperature": 0},
    )
    await agent.run(
        "Speak the sentence.",
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": "generate_voice"},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one spoken line through MCP.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    note = json.loads((root / "docs" / "voxcpm2" / "smoke.json").read_text(encoding="utf-8"))
    body = {"input": note["input"], "model": note["model"]}
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
        reports: list[str] = []
        print("voice", flush=True)
        asyncio.run(_run(f"http://{HOST}:{port}/v1", voice_tool(port, body, out, reports)))
        if not reports:
            print("voice file was not written", flush=True)
            return 1
        print(reports[-1], end="", flush=True)
        if not report_ok(reports[-1], rate=rate, max_seconds=max_seconds):
            print("voice file failed the format check", flush=True)
            return 1
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
