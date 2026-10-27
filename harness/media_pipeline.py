"""Preempt each served media pipeline and require the same task to finish.

One Microsoft Agent Framework agent, on the registry default chat model,
starts a task with the matching MCP tool. The next turn is a chat
completion, which preempts that task. The tool polls ``media_status`` and
``tasks/get``, then reads the file. The cases are an image, the Wan 2.2
I2V clip, the Wan 2.2 TI2V text clip, a music clip, and a spoken line.
Exit 0 means each selected task was paused with ``task is ongoing`` and
the same task finished at that case's shape. Names after the command run
that subset, in catalog order. An omitted list runs every probe. This
command is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import threading
import time
import wave
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from agent_framework import Agent, tool
from openai import AsyncOpenAI

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.mcp_media import call_tool, read_link, start_task, task_view
from harness.music_template import smoke_body
from harness.server import close_listener, open_listener
from harness.video_mp4 import save_mp4
from harness.video_smoke_agent import _image_block, _png_size, post_json, smoke_paths

HOST = "127.0.0.1"
FRAMES = 5
WIDTH = 576
HEIGHT = 320
CLIP_SECONDS = 0.5
IMAGE_MODEL = "Z-Image-Turbo-fp4"
TI2V_FRAMES = 5
TI2V_WIDTH = 1280
TI2V_HEIGHT = 704
TI2V_SECONDS = 0.2
MUSIC_RATE = 48000
MUSIC_CHANNELS = 2
MUSIC_TOLERANCE = 0.05
RMS_FLOOR = 1e-4
_WAIT = 3600.0
_PROBES = ("image", "i2v", "t2v", "music", "voice")


class Probe(NamedTuple):
    """One media task the agent starts and then requires to recover."""

    name: str
    tool: str
    arguments: dict[str, Any]
    mime: str
    filename: str
    start_prompt: str


def recovered(samples: list[dict[str, Any]]) -> bool:
    """True when a working pause is followed by completion of that task."""
    paused = False
    for row in samples:
        status = row.get("status")
        if status in {"failed", "cancelled"}:
            return False
        if status == "working" and row.get("stage") == "paused":
            paused = True
            continue
        if paused and status == "completed":
            return True
    return False


def status_record(result: dict[str, Any]) -> dict[str, Any]:
    """The object from a ``media_status`` tool result."""
    if result.get("isError"):
        message = json.dumps(result)
        raise RuntimeError(message)
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        inner = structured.get("result")
        if isinstance(inner, dict) and "status" in inner:
            return inner
        if "status" in structured:
            return structured
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "status" in parsed:
                return parsed
    message = "media_status returned no record"
    raise RuntimeError(message)


def wav_fields(blob: bytes) -> tuple[int, int, float, float]:
    """Sample rate, channels, seconds, and RMS of one WAV payload."""
    with wave.open(io.BytesIO(blob), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        pcm = np.frombuffer(handle.readframes(frames), dtype="<i2").astype(np.float32) / 32768.0
    seconds = frames / rate if rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
    return rate, channels, seconds, rms


def select_probes(cases: list[Probe], names: list[str] | None) -> list[Probe]:
    """The named probes in catalog order. An empty list is every probe."""
    if tuple(case.name for case in cases) != _PROBES:
        message = "the probe list is not the served media pipelines"
        raise ValueError(message)
    wanted = list(_PROBES if not names else names)
    unknown = [name for name in wanted if name not in _PROBES]
    if unknown:
        message = f"unknown media pipeline {unknown[0]}"
        raise ValueError(message)
    chosen = set(wanted)
    by_name = {case.name: case for case in cases}
    return [by_name[name] for name in _PROBES if name in chosen]


def probes(root: Path, still: Path) -> list[Probe]:
    """The five served media tasks, shortest settings first within each kind."""
    image = json.loads((root / "docs" / "z-image-turbo" / "smoke.json").read_text(encoding="utf-8"))
    ti2v = json.loads((root / "docs" / "wan22-ti2v-5b" / "smoke.json").read_text(encoding="utf-8"))
    music = json.loads((root / "docs" / "ace-step-15-turbo" / "smoke.json").read_text(encoding="utf-8"))
    voice = json.loads((root / "docs" / "voxcpm2" / "smoke.json").read_text(encoding="utf-8"))
    video = smoke_paths(root)[1]
    music_body = smoke_body(music)
    voice_body: dict[str, Any] = {
        "input": voice["input"],
        "model": voice["model"],
        "response_format": "wav",
    }
    instructions = voice.get("instructions") or ""
    if instructions:
        voice_body["instructions"] = instructions
    return [
        Probe(
            "image",
            "generate_image",
            {
                "model": IMAGE_MODEL,
                "prompt": image["prompt"],
                "size": f"{image['width']}x{image['height']}",
                "n": 1,
                "quality": "low",
                "seed": image["seed"],
            },
            "image/png",
            "media-preempt-image.png",
            "Start the image.",
        ),
        Probe(
            "i2v",
            "generate_video_i2v",
            {
                "model": video["model"],
                "prompt": video["video_prompt"],
                "image": _image_block(still),
                "size": f"{WIDTH}x{HEIGHT}",
                "duration": CLIP_SECONDS,
                "seed": video["seed"],
            },
            "video/mp4",
            "media-preempt-i2v.mp4",
            "Start the 5-frame 320p clip.",
        ),
        Probe(
            "t2v",
            "generate_video_t2v",
            {
                "model": ti2v["model"],
                "prompt": ti2v["prompt"],
                "size": f"{TI2V_WIDTH}x{TI2V_HEIGHT}",
                "duration": TI2V_SECONDS,
                "seed": ti2v["seed"],
            },
            "video/mp4",
            "media-preempt-t2v.mp4",
            "Start the 5-frame text clip.",
        ),
        Probe(
            "music",
            "generate_music",
            music_body,
            "audio/wav",
            "media-preempt-music.wav",
            "Start the 10 second clip.",
        ),
        Probe(
            "voice",
            "generate_voice",
            voice_body,
            "audio/wav",
            "media-preempt-voice.wav",
            "Start the spoken line.",
        ),
    ]


def accept(probe: Probe, blob: bytes, dest: Path, *, voice_rate: int, voice_max: float) -> str | None:
    """Write ``blob`` and return a reason when its shape is wrong."""
    if probe.name in {"i2v", "t2v"}:
        frames, width, height = save_mp4(blob, dest)
        expect = (FRAMES, WIDTH, HEIGHT) if probe.name == "i2v" else (TI2V_FRAMES, TI2V_WIDTH, TI2V_HEIGHT)
        print(f"wrote {dest.name} frames {frames} size {width}x{height}", flush=True)
        if (frames, width, height) != expect:
            return f"{probe.name} is {frames} frames at {width}x{height}"
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_bytes(blob)
    temporary.replace(dest)
    if probe.name == "image":
        width, height = _png_size(blob)
        print(f"wrote {dest.name} size {width}x{height}", flush=True)
        raw = str(probe.arguments["size"])
        want_w, _, want_h = raw.partition("x")
        if (width, height) != (int(want_w), int(want_h)):
            return f"image is {width}x{height}"
        return None
    rate, channels, seconds, rms = wav_fields(blob)
    print(
        f"wrote {dest.name} rate {rate} channels {channels} seconds {seconds:.3f} rms {rms:.6f}",
        flush=True,
    )
    if rms <= RMS_FLOOR:
        return f"{probe.name} is silent"
    if probe.name == "music":
        if (rate, channels) != (MUSIC_RATE, MUSIC_CHANNELS):
            return f"music is {rate} Hz {channels} ch"
        target = float(probe.arguments["seconds"])
        if abs(seconds - target) > MUSIC_TOLERANCE:
            return f"music is {seconds:.3f}s"
        return None
    if (rate, channels) != (voice_rate, 1):
        return f"voice is {rate} Hz {channels} ch"
    if not 0.0 < seconds <= voice_max:
        return f"voice is {seconds:.3f}s"
    return None


class _Watch:
    """Samples taken while one media task is open."""

    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.media_paused: dict[str, Any] | None = None
        self.task_id = ""
        self._stage = ""
        self._lock = threading.Lock()
        self.stop = threading.Event()

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()
            self.media_paused = None
            self.task_id = ""
            self._stage = ""

    def add(self, row: dict[str, Any]) -> None:
        with self._lock:
            if row.get("_task") != self.task_id:
                return
            self.samples.append(row)
            stage = row.get("stage")
            if isinstance(stage, str) and stage != self._stage:
                self._stage = stage
                progress = row.get("progress")
                print(f"stage {stage} progress {progress}", flush=True)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.samples)

    def note_paused(self, row: dict[str, Any]) -> None:
        with self._lock:
            if self.media_paused is None:
                self.media_paused = row


def _holding(view: dict[str, Any]) -> bool:
    """True once the task is working and is not queued or paused."""
    if view.get("status") != "working":
        return False
    if view.get("position") is not None:
        return False
    if view.get("stage") == "paused":
        return False
    return True


def _chat(port: int, model: str) -> None:
    payload = post_json(
        port,
        "/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 8,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": "Reply with one word: ready."}],
        },
    )
    if "error" in payload:
        message = json.dumps(payload["error"])
        raise RuntimeError(message)


def _sample(port: int, task_id: str, watch: _Watch) -> dict[str, Any]:
    view = task_view(port, task_id)
    stamped = dict(view)
    stamped["_task"] = task_id
    watch.add(stamped)
    if view.get("status") == "working" and view.get("stage") == "paused" and watch.media_paused is None:
        record = status_record(call_tool(port, "media_status", {"taskId": task_id}))
        watch.note_paused(record)
        print(
            f"paused stage {record.get('stage')} message {record.get('statusMessage')}",
            flush=True,
        )
    return view


def _watch_loop(port: int, watch: _Watch) -> None:
    while not watch.stop.is_set():
        task_id = watch.task_id
        if not task_id:
            watch.stop.wait(0.2)
            continue
        try:
            view = _sample(port, task_id, watch)
        except RuntimeError as exc:
            print(f"status {exc}", flush=True)
            watch.stop.wait(0.5)
            continue
        if view.get("status") in {"completed", "failed", "cancelled"}:
            watch.stop.wait(0.25)
            continue
        watch.stop.wait(0.25)


def _wait_until_holding(port: int, task_id: str, watch: _Watch, deadline: float) -> None:
    while time.monotonic() < deadline:
        view = _sample(port, task_id, watch)
        status = view.get("status")
        if status in {"failed", "cancelled"}:
            message = json.dumps(view)
            raise RuntimeError(message)
        if status == "completed":
            message = "the task finished before the chat"
            raise RuntimeError(message)
        if _holding(view):
            return
        time.sleep(0.25)
    message = "the task did not start"
    raise RuntimeError(message)


def _preempt_chat(port: int, model: str, task_id: str, watch: _Watch, deadline: float) -> None:
    """Post one short chat and sample the task until that chat returns."""
    error: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        try:
            _chat(port, model)
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    print("preempt chat", flush=True)
    threading.Thread(target=run, name="favlo-preempt-chat", daemon=True).start()
    while not done.is_set():
        if time.monotonic() >= deadline:
            message = "the preempt chat did not return"
            raise RuntimeError(message)
        try:
            _sample(port, task_id, watch)
        except RuntimeError as exc:
            print(f"status {exc}", flush=True)
        done.wait(0.2)
    if error:
        raise error[0]


def _wait_result(port: int, task_id: str, watch: _Watch, deadline: float) -> dict[str, Any]:
    while time.monotonic() < deadline:
        view = _sample(port, task_id, watch)
        status = view.get("status")
        if status == "completed":
            result = view.get("result")
            if isinstance(result, dict) and result.get("isError"):
                message = json.dumps(result)
                raise RuntimeError(message)
            if not isinstance(result, dict):
                message = "the task completed without a result"
                raise RuntimeError(message)
            watch.add({"status": "completed", "taskId": task_id, "_task": task_id})
            return result
        if status in {"failed", "cancelled"}:
            message = json.dumps(view)
            raise RuntimeError(message)
        time.sleep(2)
    message = f"task {task_id} did not finish"
    raise RuntimeError(message)


def _tools(port: int, model: str, probe: Probe, watch: _Watch, media: list[bytes]) -> list[object]:
    @tool(approval_mode="never_require")
    def start_media() -> str:
        """Start the named media task and return when it is running."""
        deadline = time.monotonic() + _WAIT
        task_id = start_task(port, probe.tool, probe.arguments)
        watch.task_id = task_id
        print(f"task {task_id}", flush=True)
        _wait_until_holding(port, task_id, watch, deadline)
        print(f"{probe.name} running", flush=True)
        time.sleep(1.0)
        return task_id

    @tool(approval_mode="never_require")
    def confirm_recovery() -> str:
        """Require the running task to pause, then finish as the same task."""
        task_id = watch.task_id
        if not task_id:
            message = "the task has not started"
            raise RuntimeError(message)
        deadline = time.monotonic() + _WAIT
        attempts = 0
        while watch.media_paused is None and attempts < 3 and time.monotonic() < deadline:
            view = _sample(port, task_id, watch)
            status = view.get("status")
            if status in {"failed", "cancelled"}:
                message = json.dumps(view)
                raise RuntimeError(message)
            if status == "completed":
                message = "the task finished before it paused"
                raise RuntimeError(message)
            if view.get("position") is not None:
                time.sleep(0.5)
                continue
            time.sleep(1.0)
            if watch.media_paused is not None:
                break
            attempts += 1
            _preempt_chat(port, model, task_id, watch, deadline)
        paused = watch.media_paused
        if paused is None or paused.get("stage") != "paused":
            message = "media_status did not report paused"
            raise RuntimeError(message)
        if paused.get("statusMessage") != "task is ongoing":
            message = "media_status did not keep the task ongoing"
            raise RuntimeError(message)
        if paused.get("taskId") not in (None, task_id):
            message = "media_status reported a different task"
            raise RuntimeError(message)
        result = _wait_result(port, task_id, watch, deadline)
        if not recovered(watch.snapshot()):
            message = "the task did not recover"
            raise RuntimeError(message)
        media.append(read_link(port, result, probe.mime))
        return "recovered"

    return [start_media, confirm_recovery]


def _agent(base_url: str, tools: list[object], name: str, text: str, model: str) -> int:
    try:
        asyncio.run(_run(base_url, tools, name, text, model))
    except Exception as exc:
        print(exc, flush=True)
        return 1
    return 0


async def _run(base_url: str, tools: list[object], name: str, text: str, model: str) -> None:
    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=None, max_retries=0)
    client = open_chat_client(model, base_url, async_client=http)
    client.function_invocation_configuration["max_iterations"] = 1
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="media-pipeline",
        instructions="Call the named tool once. Then stop.",
        tools=tools,
        default_options={
            "temperature": 0,
            "extra_body": {"enable_thinking": False},
        },
    )
    await agent.run(
        text,
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": name},
            "extra_body": {"enable_thinking": False},
        },
    )


def _voice_bounds(root: Path) -> tuple[int, float]:
    voice = json.loads((root / "docs" / "voxcpm2" / "smoke.json").read_text(encoding="utf-8"))
    return int(voice["sample_rate"]), float(voice["max_seconds"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preempt served media pipelines and require recovery.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--judge", default=None, help="chat model id; default is the registry default")
    parser.add_argument(
        "pipelines",
        nargs="*",
        metavar="pipeline",
        help="probe names to run, in catalog order; omitted runs every served media pipeline",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    judge_id = args.judge or judge_model(root)
    still = root / "artifacts" / "video-smoke-agent.png"
    if not args.pipelines or "i2v" in args.pipelines:
        if not still.is_file() or _png_size(still.read_bytes()) != (WIDTH, HEIGHT):
            print(f"still is not {WIDTH}x{HEIGHT}", flush=True)
            return 1
    try:
        cases = select_probes(probes(root, still), args.pipelines)
    except ValueError as exc:
        print(exc, flush=True)
        return 1
    voice_rate, voice_max = _voice_bounds(root)
    listener = None
    watch = _Watch()
    watcher: threading.Thread | None = None
    try:
        listener = open_listener()
        port = listener.port
        watcher = threading.Thread(
            target=_watch_loop,
            args=(port, watch),
            name="favlo-preempt-watch",
            daemon=True,
        )
        watcher.start()
        base = f"http://{HOST}:{port}/v1"
        for probe in cases:
            watch.reset()
            media: list[bytes] = []
            tools = _tools(port, judge_id, probe, watch, media)
            print(f"start {probe.name}", flush=True)
            failed = _agent(base, [tools[0]], "start_media", probe.start_prompt, judge_id)
            if failed != 0 or not watch.task_id:
                print(f"{probe.name} did not start", flush=True)
                return 1
            print(f"preempt {probe.name}", flush=True)
            failed = _agent(
                base,
                [tools[1]],
                "confirm_recovery",
                "The task is running. Confirm that it recovers.",
                judge_id,
            )
            if failed != 0 or not media:
                print(f"{probe.name} did not recover", flush=True)
                return 1
            reason = accept(
                probe,
                media[-1],
                root / "artifacts" / probe.filename,
                voice_rate=voice_rate,
                voice_max=voice_max,
            )
            if reason is not None:
                print(reason, flush=True)
                return 1
            if not recovered(watch.snapshot()):
                print(f"{probe.name} did not resume", flush=True)
                return 1
            print(f"recovered {probe.name}", flush=True)
        return 0
    finally:
        watch.stop.set()
        if watcher is not None and watcher.ident is not None:
            watcher.join(timeout=2)
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
