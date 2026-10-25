"""Agent that generates a 3 minute clip and checks that it holds together.

The Microsoft Agent Framework agent calls ``generate_music`` on ``POST /mcp``
and reads ``favlo://music/{id}.wav``. The saved file is analyzed by
`harness/inspect_wav.py`. Exit 0 means the file is 3 minutes, 48 kHz, stereo,
and above the silence floor, the inspect template matches the request, the
key margin is settled, the spectrum is tonal, and the one-second tile joins
are not louder than the surrounding samples. The command does not take the
machine lock. The judge does not listen. This command is not part of pytest.
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

from harness.inspect_wav import inspect_file
from harness.mcp_media import generate, read_link
from harness.music_template import compare_templates, input_template, smoke_body
from harness.server import close_listener, open_listener

JUDGE = "qwen36-35b-a3b:instruct"
HOST = "127.0.0.1"
RMS_FLOOR = 1e-4
FLATNESS_CEILING = 0.35
QUIET_SHARE = 0.15
KEY_MARGIN = 0.02
SEAM_RATIO = 15.0
TILE_SECONDS = 1


def music_tool(port: int, body: dict[str, Any], out: Path, reports: list[str]) -> object:
    @tool(approval_mode="never_require")
    def generate_music() -> str:
        """Generate the 3 minute clip. The harness inspects the file."""
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


async def _run(base_url: str, music: object) -> None:
    client = OpenAIChatCompletionClient(model=JUDGE, api_key="favlo", base_url=base_url)
    client.function_invocation_configuration["max_iterations"] = 2
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="music-long-agent",
        instructions="Call generate_music once. Then stop.",
        tools=[music],
        default_options={"temperature": 0},
    )
    await agent.run(
        "Generate the 3 minute music clip.",
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": "generate_music"},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a 3 minute music clip.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    note_path = root / "docs" / "ace-step-15-turbo" / "long.json"
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
        out = root / "artifacts" / "music-long-agent.wav"
        reports: list[str] = []
        print("music", flush=True)
        asyncio.run(_run(f"http://{HOST}:{port}/v1", music_tool(port, body, out, reports)))
        if not reports:
            print("music file was not written", flush=True)
            return 1
        print(incoming, end="", flush=True)
        print(reports[-1], end="", flush=True)
        problems = report_problems(reports[-1], float(note["seconds"]))
        mono, rate = wav_mono(out)
        ratio = boundary_ratio(mono, rate)
        print(f"seam_ratio {ratio:.4f}", flush=True)
        if ratio > SEAM_RATIO:
            problems.append(f"a tile join is {ratio:.1f} times the nearby step")
        mismatches = compare_templates(incoming, reports[-1])
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


if __name__ == "__main__":
    raise SystemExit(main())
