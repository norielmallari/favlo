"""Media tools on the same process as the listener.

FastMCP serves MCP 2026-07-28 at ``/mcp``. ``generate_image``, ``generate_video``, ``generate_music``, and
``generate_voice`` are optional Tasks. A client that
advertises ``io.modelcontextprotocol/tasks`` gets a ``CreateTaskResult`` and
the in-memory worker runs one task at a time. A client that does not gets the
tool result on that call. ``list_media_models``, ``media_status``, and
``media_cancel`` run in the request. A generate tool calls the pipeline
handler through the same lock as ``POST /v1``. Chat completion is not a tool.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import timedelta
from pathlib import Path
from typing import Any

import mcp_types
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.resources import ResourceContent, ResourceResult
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.tasks import TaskConfig
from fastmcp_tasks import TasksExtension
from fastmcp_tasks.context import get_task_context
from fastmcp_tasks.handlers import tasks_cancel, tasks_get
from fastmcp_tasks.input_store import is_cancelled
from fastmcp_tasks.models import CreateTaskResult, GetTaskParams, GetTaskResult
from mcp.shared.exceptions import MCPError

from favlo import __version__
from favlo.images import load_image
from favlo.music import load_music, music_dir
from favlo.videos import load_video
from favlo.voice import load_voice, voice_dir

_MEDIA = (
    ("image", "handle_image"),
    ("video", "handle_video"),
    ("music", "handle_music"),
    ("voice", "handle_voice"),
)
_POLL = timedelta(seconds=2)
_TASK = TaskConfig(mode="optional", poll_interval=_POLL)
_INLINE = 1024 * 1024
_ANNOTATIONS = mcp_types.ToolAnnotations(
    read_only_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_AUDIO = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
_IMAGE_EXT = frozenset({"png"})
_VIDEO_EXT = frozenset({"mp4"})


def build_mcp(dispatch: Any, state: dict[str, Any]) -> FastMCP:
    """Tools close over ``dispatch(kind, body, task_id)``.

    ``kind`` is ``list``, ``image``, ``video``, or ``music``. A generate call
    passes the task id the worker is running. ``list`` ignores it. Task
    records are stored on ``state``.
    """
    mcp = FastMCP("Favlo", version=__version__)
    tasks = FavloTasks(state)
    mcp.add_extension(tasks)

    @mcp.tool(task=False)
    def list_media_models() -> list[dict[str, Any]]:
        """Registry rows whose pipeline generates an image, a video, music, or voice."""
        return dispatch("list", {})

    @mcp.tool(task=False)
    async def media_status(taskId: str) -> dict[str, Any]:
        """The queue record for one task. An unknown id is an error."""
        return await tasks.status(taskId)

    @mcp.tool(task=False)
    async def media_cancel(taskId: str) -> dict[str, Any]:
        """Cancel one task. A second call returns the same status."""
        return await tasks.cancel(taskId)

    @mcp.tool(task=_TASK, annotations=_ANNOTATIONS)
    async def generate_image(
        prompt: str,
        model: str,
        size: str = "1280x720",
        n: int = 1,
        quality: str = "high",
        seed: int = 0,
        include_image: bool = False,
    ) -> ToolResult:
        """Generate one or more PNGs. ``n`` is 1 to 10 and is one task.

        ``size`` defaults to ``1280x720``. ``auto`` is that same 16:9 frame.
        """
        body = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "n": n,
            "quality": quality,
            "seed": seed,
        }
        payload = await _run_task(dispatch, "image", body)
        return _image_result(payload, include_image, tasks.state)

    @mcp.tool(task=_TASK, annotations=_ANNOTATIONS)
    async def generate_video(
        prompt: str,
        model: str,
        image: str | dict[str, Any],
        size: str = "576x320",
        seconds: str = "",
        frames: int = 0,
        seed: int = 0,
        quality: str = "high",
    ) -> ToolResult:
        """Generate one clip. The still is a stored image id or a PNG block."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "seed": seed,
            "quality": quality,
            "image": _still(image, tasks.state),
        }
        if seconds:
            body["seconds"] = seconds
        if frames:
            body["frames"] = frames
        payload = await _run_task(dispatch, "video", body)
        return _video_result(payload)

    @mcp.tool(task=_TASK, annotations=_ANNOTATIONS)
    async def generate_music(
        prompt: str,
        model: str,
        response_format: str = "mp3",
        lyrics: str = "",
        seconds: float | None = None,
        seed: int | None = None,
        vocal_language: str = "",
        bpm: int | None = None,
        keyscale: str = "",
        timesignature: str = "",
        include_audio: bool = False,
    ) -> ToolResult:
        """Generate one clip. ``response_format`` is mp3, wav, or flac."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "response_format": response_format,
        }
        if lyrics:
            body["lyrics"] = lyrics
        if seconds is not None:
            body["seconds"] = seconds
        if seed is not None:
            body["seed"] = seed
        if vocal_language:
            body["vocal_language"] = vocal_language
        if bpm is not None:
            body["bpm"] = bpm
        if keyscale:
            body["keyscale"] = keyscale
        if timesignature:
            body["timesignature"] = timesignature
        payload = await _run_task(dispatch, "music", body)
        return _music_result(payload, include_audio, tasks.state)

    @mcp.tool(task=_TASK, annotations=_ANNOTATIONS)
    async def generate_voice(
        input: str,
        model: str,
        voice: str = "",
        instructions: str = "",
        response_format: str = "mp3",
        include_audio: bool = False,
    ) -> ToolResult:
        """Speak one line.

        ``voice`` is a speaker id from ``list_media_models``.
        ``response_format`` is mp3, wav, or flac.
        """
        body: dict[str, Any] = {
            "input": input,
            "model": model,
            "response_format": response_format,
        }
        if voice:
            body["voice"] = voice
        if instructions:
            body["instructions"] = instructions
        payload = await _run_task(dispatch, "voice", body)
        return _voice_result(payload, include_audio, tasks.state)

    @mcp.resource("favlo://images/{image_id}", mime_type="image/png")
    def read_image(image_id: str) -> bytes:
        """The stored PNG. A ``.png`` suffix is the same file."""
        stem, _ext = _split_suffix(image_id, _IMAGE_EXT)
        raw = load_image(stem, home=_home(tasks.state, "image_home"))
        if raw is None:
            message = "stored image is missing"
            raise ToolError(message)
        return raw

    @mcp.resource("favlo://videos/{video_id}", mime_type="video/mp4")
    def read_video(video_id: str) -> bytes:
        """The stored MP4. A ``.mp4`` suffix is the same file."""
        stem, _ext = _split_suffix(video_id, _VIDEO_EXT)
        raw = load_video(stem, home=_home(tasks.state, "video_home"))
        if raw is None:
            message = "stored video is missing"
            raise ToolError(message)
        return raw

    @mcp.resource("favlo://voice/{voice_id}")
    def read_voice(voice_id: str) -> ResourceResult:
        """The stored spoken audio. A suffix selects that encoding."""
        return _read_audio(tasks.state, voice_id, "voice", stored_voice, load_voice)

    @mcp.resource("favlo://music/{music_id}")
    def read_music(music_id: str) -> ResourceResult:
        """The stored audio. A suffix selects that encoding."""
        return _read_audio(tasks.state, music_id, "music", stored_music, load_music)

    return mcp


class FavloTasks(TasksExtension):
    """One in-memory Docket worker. Records live on the listener state."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        super().__init__(
            url="memory://",
            name=f"favlo-{id(state)}",
            concurrency=1,
            redelivery_timeout=timedelta(hours=24),
        )

    async def intercept_tool_call(self, params: Any, context: Any, call_next: Any) -> Any:
        outcome = await super().intercept_tool_call(params, context, call_next)
        if isinstance(outcome, CreateTaskResult):
            tasks = self.state.setdefault("media_tasks", {})
            order = self.state.setdefault("media_order", [])
            tasks[outcome.task_id] = {
                "taskId": outcome.task_id,
                "tool": params.name,
                "arguments": dict(params.arguments or {}),
                "started": False,
                "paused": False,
                "job": None,
            }
            order.append(outcome.task_id)
        return outcome

    async def _handle_get(self, ctx: Any, params: GetTaskParams) -> GetTaskResult | dict[str, Any]:
        result = await super()._handle_get(ctx, params)
        if result.status != "working":
            return result
        dumped = result.model_dump(by_alias=True, mode="json", exclude_none=True)
        dumped.update(_working_fields(self.state, params.task_id))
        return dumped

    async def _handle_cancel(self, ctx: Any, params: GetTaskParams) -> Any:
        result = await super()._handle_cancel(ctx, params)
        _flag_cancel(self.state, params.task_id)
        return result

    async def status(self, task_id: str) -> dict[str, Any]:
        try:
            got = await tasks_get(self.server, task_id)
        except MCPError as exc:
            raise ToolError(exc.error.message) from exc
        view = _stored(self.state, task_id)
        view["status"] = got.status
        if got.status == "working":
            view.update(_working_fields(self.state, task_id))
        if got.status == "failed" and got.error:
            view["error"] = got.error
        return view

    async def cancel(self, task_id: str) -> dict[str, Any]:
        try:
            await tasks_cancel(self.server, task_id)
        except MCPError as exc:
            raise ToolError(exc.error.message) from exc
        _flag_cancel(self.state, task_id)
        return await self.status(task_id)


def media_kinds(module: Any) -> list[str]:
    found = [name for name, attr in _MEDIA if callable(getattr(module, attr, None))]
    return found


def _home(state: dict[str, Any], key: str) -> Path | None:
    home = state.get(key)
    if isinstance(home, Path):
        return home
    return None


async def _run_task(dispatch: Any, kind: str, body: dict[str, Any]) -> dict[str, Any]:
    info = get_task_context()
    task_id = info.task_id if info is not None else ""
    if task_id and await _already_cancelled(task_id):
        return {"status": "cancelled"}
    payload = await asyncio.to_thread(dispatch, kind, body, task_id)
    if not isinstance(payload, dict):
        message = "the pipeline wrote no reply"
        raise ToolError(message)
    return payload


async def _already_cancelled(task_id: str) -> bool:
    from fastmcp_tasks.dependencies import _current_docket

    docket = _current_docket.get()
    if docket is None:
        return False
    from fastmcp_tasks.context import get_task_scope

    return await is_cancelled(docket, get_task_scope(), task_id)


def _still(image: Any, state: dict[str, Any]) -> str:
    if isinstance(image, dict):
        if image.get("type") != "image":
            raise RuntimeError("the still is a stored image id or a PNG image block")
        data = image.get("data")
        if not isinstance(data, str) or not data:
            raise RuntimeError("the still is a stored image id or a PNG image block")
        return data
    if not isinstance(image, str) or not image or _is_path(image):
        raise RuntimeError("the still is a stored image id or a PNG image block")
    if image.startswith("image_"):
        raw = load_image(image, home=_home(state, "image_home"))
        if raw is None:
            raise RuntimeError("stored image is missing")
        return base64.b64encode(raw).decode("ascii")
    return image


def _is_path(value: str) -> bool:
    if value.startswith(("/", "~", "./", "../")) or "\\" in value:
        return True
    return "/" in value and not value.startswith("data:")


def _image_result(
    payload: dict[str, Any], include_image: bool, state: dict[str, Any]
) -> ToolResult:
    if payload.get("status") == "cancelled":
        return ToolResult([mcp_types.TextContent(type="text", text="cancelled")])
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        message = "the pipeline wrote no image"
        raise RuntimeError(message)
    blocks: list[mcp_types.ContentBlock] = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        image_id = item["id"]
        blocks.append(mcp_types.TextContent(type="text", text=image_id))
        blocks.append(_media_link("images", image_id, "png", "image/png"))
        if not include_image:
            continue
        raw = _image_bytes(item, state)
        if raw is not None and len(raw) <= _INLINE:
            blocks.append(
                mcp_types.ImageContent(
                    type="image",
                    data=base64.b64encode(raw).decode("ascii"),
                    mime_type="image/png",
                )
            )
    if not blocks:
        message = "the pipeline wrote no image"
        raise RuntimeError(message)
    return ToolResult(blocks)


def _image_bytes(item: dict[str, Any], state: dict[str, Any]) -> bytes | None:
    image_id = item.get("id")
    if isinstance(image_id, str):
        raw = load_image(image_id, home=_home(state, "image_home"))
        if raw is not None:
            return raw
    encoded = item.get("b64_json")
    if not isinstance(encoded, str):
        return None
    try:
        return base64.b64decode(encoded)
    except (ValueError, TypeError):
        return None


def _video_result(payload: dict[str, Any]) -> ToolResult:
    if payload.get("status") == "cancelled":
        return ToolResult([mcp_types.TextContent(type="text", text="cancelled")])
    video_id = payload.get("id")
    if not isinstance(video_id, str) or not video_id:
        message = "the pipeline wrote no video"
        raise RuntimeError(message)
    size = payload.get("size") if isinstance(payload.get("size"), str) else ""
    seconds = payload.get("seconds") if isinstance(payload.get("seconds"), str) else ""
    text = " ".join(part for part in (video_id, size, seconds) if part)
    return ToolResult(
        [
            mcp_types.TextContent(type="text", text=text),
            _media_link("videos", video_id, "mp4", "video/mp4"),
        ]
    )


def _music_result(
    payload: dict[str, Any], include_audio: bool, state: dict[str, Any]
) -> ToolResult:
    if payload.get("status") == "cancelled":
        return ToolResult([mcp_types.TextContent(type="text", text="cancelled")])
    music_id = payload.get("id")
    if not isinstance(music_id, str) or not music_id:
        message = "the pipeline wrote no music"
        raise RuntimeError(message)
    fmt = payload.get("response_format")
    if not isinstance(fmt, str) or fmt not in _AUDIO:
        fmt = "mp3"
    mime = _AUDIO[fmt]
    blocks: list[mcp_types.ContentBlock] = [
        mcp_types.TextContent(type="text", text=music_id),
        _media_link("music", music_id, fmt, mime),
    ]
    if include_audio:
        loaded = load_music(music_id, fmt, home=_home(state, "music_home"))
        if loaded is not None and len(loaded[0]) <= _INLINE:
            blocks.append(
                mcp_types.AudioContent(
                    type="audio",
                    data=base64.b64encode(loaded[0]).decode("ascii"),
                    mime_type=mime,
                )
            )
    return ToolResult(blocks)


def _voice_result(
    payload: dict[str, Any], include_audio: bool, state: dict[str, Any]
) -> ToolResult:
    if payload.get("status") == "cancelled":
        return ToolResult([mcp_types.TextContent(type="text", text="cancelled")])
    voice_id = payload.get("id")
    if not isinstance(voice_id, str) or not voice_id:
        message = "the pipeline wrote no voice"
        raise RuntimeError(message)
    fmt = payload.get("response_format")
    if not isinstance(fmt, str) or fmt not in _AUDIO:
        fmt = "mp3"
    mime = _AUDIO[fmt]
    blocks: list[mcp_types.ContentBlock] = [
        mcp_types.TextContent(type="text", text=voice_id),
        _media_link("voice", voice_id, fmt, mime),
    ]
    if include_audio:
        loaded = load_voice(voice_id, fmt, home=_home(state, "voice_home"))
        if loaded is not None and len(loaded[0]) <= _INLINE:
            blocks.append(
                mcp_types.AudioContent(
                    type="audio",
                    data=base64.b64encode(loaded[0]).decode("ascii"),
                    mime_type=mime,
                )
            )
    return ToolResult(blocks)


def _media_link(kind: str, item_id: str, ext: str, mime: str) -> mcp_types.ResourceLink:
    """A link whose URI path ends in the file extension."""
    filename = f"{item_id}.{ext}"
    return mcp_types.ResourceLink(
        type="resource_link",
        uri=f"favlo://{kind}/{filename}",
        name=filename,
        mime_type=mime,
    )


def _split_suffix(value: str, suffixes: frozenset[str]) -> tuple[str, str | None]:
    """Split a known suffix from an id. A bare id stays whole."""
    stem, dot, ext = value.rpartition(".")
    if dot and ext in suffixes and stem and "." not in stem:
        return stem, ext
    return value, None


def _read_audio(
    state: dict[str, Any],
    value: str,
    kind: str,
    stored: Any,
    load: Any,
) -> ResourceResult:
    """Bytes for one clip. A suffix names the encoding. A bare id uses the job."""
    stem, ext = _split_suffix(value, frozenset(_AUDIO))
    if ext is not None:
        try:
            loaded = load(stem, ext, home=_home(state, f"{kind}_home"))
        except ValueError:
            loaded = None
        if loaded is None:
            message = f"stored {kind} is missing"
            raise ToolError(message)
        return ResourceResult([ResourceContent(loaded[0], mime_type=_AUDIO[ext])])
    found = stored(state, value)
    if found is None:
        message = f"stored {kind} is missing"
        raise ToolError(message)
    raw, mime = found
    return ResourceResult([ResourceContent(raw, mime_type=mime)])


def stored_voice(state: dict[str, Any], voice_id: str) -> tuple[bytes, str] | None:
    """Bytes and MIME type for one stored spoken clip."""
    home = _home(state, "voice_home")
    chosen = _voice_format(state, voice_id)
    if chosen is not None:
        try:
            loaded = load_voice(voice_id, chosen, home=home)
        except ValueError:
            return None
        if loaded is not None:
            return loaded[0], _AUDIO[chosen]
    try:
        wav = load_voice(voice_id, "wav", home=home)
    except ValueError:
        return None
    if wav is not None:
        return wav[0], _AUDIO["wav"]
    root = home if isinstance(home, Path) else None
    if root is None:
        from favlo.logs import favlo_home

        root = favlo_home()
    folder = voice_dir(root)
    for fmt, mime in (("wav", "audio/wav"), ("flac", "audio/flac"), ("mp3", "audio/mpeg")):
        path = folder / f"{voice_id}.{fmt}"
        try:
            if path.is_file():
                return path.read_bytes(), mime
        except OSError:
            continue
    return None


def _voice_format(state: dict[str, Any], voice_id: str) -> str | None:
    jobs = state.get("voice")
    if not isinstance(jobs, dict):
        return None
    job = jobs.get(voice_id)
    if not isinstance(job, dict):
        return None
    fmt = job.get("response_format")
    if isinstance(fmt, str) and fmt in _AUDIO:
        return fmt
    return None


def stored_music(state: dict[str, Any], music_id: str) -> tuple[bytes, str] | None:
    """Bytes and MIME type for one stored clip.

    The job's ``response_format`` wins. Otherwise the WAV already on disk is
    read, then FLAC, then MP3. A missing encoded file is not created here.
    """
    home = _home(state, "music_home")
    chosen = _music_format(state, music_id)
    if chosen is not None:
        try:
            loaded = load_music(music_id, chosen, home=home)
        except ValueError:
            return None
        if loaded is not None:
            return loaded[0], _AUDIO[chosen]
    try:
        wav = load_music(music_id, "wav", home=home)
    except ValueError:
        return None
    if wav is not None:
        return wav[0], _AUDIO["wav"]
    root = home if isinstance(home, Path) else None
    if root is None:
        from favlo.logs import favlo_home

        root = favlo_home()
    folder = music_dir(root)
    for fmt, mime in (("wav", "audio/wav"), ("flac", "audio/flac"), ("mp3", "audio/mpeg")):
        path = folder / f"{music_id}.{fmt}"
        try:
            if path.is_file():
                return path.read_bytes(), mime
        except OSError:
            continue
    return None


def _music_format(state: dict[str, Any], music_id: str) -> str | None:
    jobs = state.get("music")
    if not isinstance(jobs, dict):
        return None
    job = jobs.get(music_id)
    if not isinstance(job, dict):
        return None
    fmt = job.get("response_format")
    if isinstance(fmt, str) and fmt in _AUDIO:
        return fmt
    return None


def _stored(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    record = (state.get("media_tasks") or {}).get(task_id)
    if not isinstance(record, dict):
        return {"taskId": task_id, "tool": None, "arguments": {}}
    return {
        "taskId": task_id,
        "tool": record.get("tool"),
        "arguments": record.get("arguments") or {},
    }


def _working_fields(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    record = (state.get("media_tasks") or {}).get(task_id)
    fields: dict[str, Any] = {}
    position = _position(state, task_id)
    if position is not None:
        fields["position"] = position
    if isinstance(record, dict) and record.get("paused"):
        fields["stage"] = "paused"
        return fields
    job = record.get("job") if isinstance(record, dict) else None
    if not isinstance(job, dict):
        job = _live_job(state)
    if not isinstance(job, dict):
        return fields
    if job.get("progress") is not None:
        fields["progress"] = job.get("progress")
    if job.get("eta") is not None:
        fields["eta"] = job.get("eta")
    if isinstance(job.get("stage"), str):
        fields["stage"] = job.get("stage")
    return fields


def _position(state: dict[str, Any], task_id: str) -> int | None:
    record = (state.get("media_tasks") or {}).get(task_id)
    if not isinstance(record, dict) or record.get("paused") or record.get("started"):
        return None
    ahead = 0
    for other in state.get("media_order") or []:
        if other == task_id:
            return ahead
        other_record = (state.get("media_tasks") or {}).get(other) or {}
        if other_record.get("paused") or not other_record.get("started"):
            ahead += 1
    return None


def _live_job(state: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("videos", "music"):
        jobs = state.get(key)
        if not isinstance(jobs, dict):
            continue
        for job in jobs.values():
            if isinstance(job, dict) and job.get("status") == "in_progress":
                return job
    return None


def _flag_cancel(state: dict[str, Any], task_id: str) -> None:
    if state.get("media_task") == task_id:
        cancel = state.get("cancel")
        if cancel is not None:
            cancel.set()
