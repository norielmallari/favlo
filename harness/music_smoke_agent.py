"""Agent that generates the music smoke for one pipeline.

The tool names come from the command. The listener already published them.
A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers those names
and no others. The run calls the first name with the body from
``docs/<pipeline>/smoke.json``. The saved file is analyzed by
`harness/inspect_wav.py`. Exit 0 means the file matches the note duration,
48 kHz, stereo, and above the silence floor, and the inspect template matches
the request template. The command does not take the machine lock. This
command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from agent_framework import Agent

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.image_smoke_agent import image_mcp, registered, smoke_note
from harness.inspect_wav import inspect_file
from harness.mcp_media import published_name
from harness.music_template import compare_templates, input_template, smoke_body
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
RMS_FLOOR = 1e-4


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


async def create_music(
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
            name="music-smoke-agent",
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
        message = "the agent did not generate music"
        raise RuntimeError(message)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_bytes(wav[-1])
    temporary.replace(out)
    return inspect_file(out, keyscale=str(body["keyscale"]), bpm=float(body["bpm"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the music smoke for one pipeline.")
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
    try:
        incoming = input_template(note)
    except (KeyError, TypeError, ValueError) as exc:
        print(str(exc), flush=True)
        return 1
    body = smoke_body(note)
    if args.model:
        body["model"] = args.model
    listener = None
    try:
        listener = open_listener()
        port = listener.port
        out = root / "artifacts" / "music-smoke-agent.wav"
        print("music", flush=True)
        model = args.judge or judge_model(root)
        report = asyncio.run(
            create_music(f"http://{HOST}:{port}/v1", port, names, body, model, out)
        )
        print(incoming, end="", flush=True)
        print(report, end="", flush=True)
        if not report_ok(report, float(note["seconds"])):
            print("music file failed the duration check", flush=True)
            return 1
        mismatches = compare_templates(incoming, report)
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


def _ask(tool: str, body: dict[str, Any]) -> str:
    lines = [f"Call {tool} once with these arguments, then stop."]
    for key in (
        "prompt",
        "lyrics",
        "seconds",
        "seed",
        "bpm",
        "keyscale",
        "timesignature",
        "response_format",
    ):
        if key in body:
            lines.append(f"{key}: {body[key]}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
