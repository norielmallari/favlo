"""Agent that generates the 10 second music smoke.

The Microsoft Agent Framework agent calls ``generate_music`` on ``POST /mcp``
and reads ``favlo://music/{id}``. The saved file is
analyzed by `harness/inspect_wav.py`. Exit 0 means the file is 10 seconds,
48 kHz, stereo, and above the silence floor, and the inspect template matches
the request template. The command does not take the machine lock. The judge
does not listen. This command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient

from harness.inspect_wav import inspect_file
from harness.mcp_media import generate, read_link
from harness.music_template import compare_templates, input_template, smoke_body
from harness.server import close_listener, open_listener

JUDGE = "qwen36-35b-a3b:instruct"
HOST = "127.0.0.1"
RMS_FLOOR = 1e-4


def music_tool(port: int, body: dict[str, Any], out: Path, reports: list[str]) -> object:
    @tool(approval_mode="never_require")
    def generate_music() -> str:
        """Generate the 10 second clip. The harness inspects the file."""
        result = generate(port, "generate_music", {**body, "response_format": "wav"})
        payload = read_link(port, result, "audio/wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(out)
        text = inspect_file(out, keyscale=str(body["keyscale"]), bpm=float(body["bpm"]))
        reports.append(text)
        return "music ready"

    return generate_music


def report_ok(text: str, seconds: float) -> bool:
    """True when the inspect report matches the smoke duration and level."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key in {"rate", "channels", "seconds", "rms"} and value:
            fields[key] = value.split()[0]
    try:
        rate = int(fields["rate"])
        channels = int(fields["channels"])
        duration = float(fields["seconds"])
        rms = float(fields["rms"])
    except (KeyError, ValueError):
        return False
    return channels == 2 and rate == 48_000 and abs(duration - seconds) <= 0.05 and rms > RMS_FLOOR


async def _run(base_url: str, music: object) -> None:
    client = OpenAIChatCompletionClient(model=JUDGE, api_key="favlo", base_url=base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="music-smoke-agent",
        instructions=(
            "Call generate_music once. Then stop."
        ),
        tools=[music],
        default_options={"temperature": 0},
    )
    await agent.run(
        "Generate the 10 second music clip.",
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": "generate_music"},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the 10 second music smoke.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    note_path = root / "docs" / "ace-step-15-turbo" / "smoke.json"
    note = json.loads(note_path.read_text(encoding="utf-8"))
    try:
        incoming = input_template(note)
    except (KeyError, TypeError, ValueError) as exc:
        print(str(exc), flush=True)
        return 1
    body = smoke_body(note)
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        out = root / "artifacts" / "music-smoke-agent.wav"
        reports: list[str] = []
        print("music", flush=True)
        asyncio.run(_run(f"http://{HOST}:{port}/v1", music_tool(port, body, out, reports)))
        if not reports:
            print("music file was not written", flush=True)
            return 1
        print(incoming, end="", flush=True)
        print(reports[-1], end="", flush=True)
        if not report_ok(reports[-1], float(note["seconds"])):
            print("music file failed the duration check", flush=True)
            return 1
        mismatches = compare_templates(incoming, reports[-1])
        if mismatches:
            print("template mismatch", flush=True)
            for item in mismatches:
                print(item, flush=True)
            return 1
        print(f"wrote {out.name}", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
