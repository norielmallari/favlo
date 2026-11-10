"""Agent that generates a long music clip and checks that it holds together.

The tool names come from the command. The listener already published them.
A Microsoft Agent Framework ``MCPStreamableHTTPTool`` registers those names
and no others. The body comes from ``docs/<pipeline>/<card>`` (default
``long.json``). The saved file is analyzed by `harness/inspect_wav.py`.
Exit 0 means the file matches the note duration, 48 kHz, stereo, and above
the silence floor, the inspect template matches the request, the key margin
is settled, the spectrum is tonal, and the one-second tile joins are not
louder than the surrounding samples. The command does not take the machine
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
from agent_framework import Agent

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.image_smoke_agent import image_mcp, registered
from harness.inspect_wav import inspect_file
from harness.mcp_media import published_name
from harness.music_template import compare_templates, input_template, smoke_body
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"
RMS_FLOOR = 1e-4
FLATNESS_CEILING = 0.35
QUIET_SHARE = 0.15
KEY_MARGIN = 0.02
SEAM_RATIO = 15.0
TILE_SECONDS = 1


def report_problems(text: str, seconds: float) -> list[str]:
    """Failures of duration, tone, key margin, and dropouts. An empty list passed."""
    fields = _leading(text)
    problems: list[str] = []
    try:
        rate = int(fields["rate"])
        channels = int(fields["channels"])
        duration = float(fields["seconds"])
        rms = float(fields["rms"])
        flatness = float(fields["spectral_flatness"])
    except (KeyError, ValueError):
        return ["the inspect report is missing a level"]
    if channels != 2 or rate != 48_000 or abs(duration - seconds) > 0.05 or rms <= RMS_FLOOR:
        problems.append("the file failed the duration check")
    if flatness > FLATNESS_CEILING:
        problems.append(f"spectral flatness {flatness:.4f} is noise")
    quiet = _quiet_share(text)
    if quiet is None:
        problems.append("the inspect report is missing quiet slices")
    elif quiet > QUIET_SHARE:
        problems.append(f"quiet slices are {quiet:.2f} of the clip")
    margin = _key_margin(text)
    if margin is None:
        problems.append("the key is unset")
    elif margin < KEY_MARGIN:
        problems.append(f"the key margin is {margin:.4f}")
    requested = _line_value(text, "requested_key")
    if requested is None or not requested.endswith(" match"):
        problems.append("the requested key missed")
    return problems


def boundary_ratio(mono: np.ndarray, rate: int, period: int = TILE_SECONDS) -> float:
    """Largest one-second join, divided by the median step beside that join."""
    if mono.size < 2:
        return 1.0
    diffs = np.abs(np.diff(mono))
    step = period * rate
    ratios: list[float] = []
    for boundary in range(step, int(mono.shape[0]) - 1, step):
        jump = float(diffs[boundary - 1])
        left = diffs[max(0, boundary - 200) : boundary - 1]
        right = diffs[boundary : boundary + 200]
        nearby = np.concatenate((left, right))
        if nearby.size == 0:
            continue
        med = float(np.median(nearby))
        if med < 1e-4:
            if jump > 0.05:
                ratios.append(SEAM_RATIO + 1.0)
            continue
        ratios.append(jump / med)
    return max(ratios) if ratios else 1.0


def wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Channel-mean float audio in ``[-1, 1]`` and the sample rate."""
    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        payload = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    audio = pcm.reshape(-1, channels) / 32768.0
    return audio.mean(axis=1), rate


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
            name="music-long-agent",
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
    parser = argparse.ArgumentParser(description="Generate a long music clip for one pipeline.")
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
        "--card",
        default="long.json",
        help="docs/<pipeline>/<card>; default is long.json",
    )
    parser.add_argument("--judge", default=None, help="chat model id; default is the registry default")
    parser.add_argument(
        "--model",
        default=None,
        help="registry folder; omit for the note's model or the host default",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    note_path = root / "docs" / args.pipeline / args.card
    note = json.loads(note_path.read_text(encoding="utf-8"))
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
        out = root / "artifacts" / "music-long-agent.wav"
        print("music", flush=True)
        model = args.judge or judge_model(root)
        report = asyncio.run(
            create_music(f"http://{HOST}:{port}/v1", port, names, body, model, out)
        )
        print(incoming, end="", flush=True)
        print(report, end="", flush=True)
        problems = report_problems(report, float(note["seconds"]))
        mono, rate = wav_mono(out)
        ratio = boundary_ratio(mono, rate)
        print(f"seam_ratio {ratio:.4f}", flush=True)
        if ratio > SEAM_RATIO:
            problems.append(f"a tile join is {ratio:.1f} times the nearby step")
        mismatches = compare_templates(incoming, report)
        if mismatches:
            problems.append("template mismatch")
            problems.extend(mismatches)
        for item in problems:
            print(item, flush=True)
        if problems:
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


def _leading(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if value:
            fields.setdefault(key, value.split()[0])
    return fields


def _line_value(text: str, name: str) -> str | None:
    prefix = name + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def _key_margin(text: str) -> float | None:
    for line in text.splitlines():
        if not line.startswith("key ") or line == "key none":
            continue
        parts = line.split()
        if "margin" not in parts:
            return None
        token = parts[parts.index("margin") + 1]
        return float(token)
    return None


def _quiet_share(text: str) -> float | None:
    value = _line_value(text, "quiet_slices")
    if value is None or "/" not in value:
        return None
    quiet, _, total = value.partition("/")
    count = int(total)
    if count < 1:
        return None
    return int(quiet) / count


if __name__ == "__main__":
    raise SystemExit(main())
