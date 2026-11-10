"""Media tools on the same process as the listener.

FastMCP serves MCP 2026-07-28 at ``/mcp``. Listener tools are
``media_status``, ``media_cancel``, ``remember``, ``recall``,
and ``forget``. A registered module's ``mcp_tools()`` adds the other tools.
The published name is ``{pipeline}__{local name}``. A tool marked ``task`` is
an optional Task. A client that advertises ``io.modelcontextprotocol/tasks``
gets a ``CreateTaskResult`` and the in-memory worker runs one task at a time.
A client that does not gets the tool result on that call. Chat completion is
not a tool.
"""

from __future__ import annotations

import base64
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import mcp_types
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.resources import ResourceContent, ResourceResult
from fastmcp.tools.base import Tool, ToolResult
from pydantic import PrivateAttr
from fastmcp.utilities.tasks import TaskConfig
from fastmcp_tasks import TasksExtension
from fastmcp_tasks.context import get_task_context
from fastmcp_tasks.handlers import tasks_cancel, tasks_get
from fastmcp_tasks.input_store import is_cancelled
from fastmcp_tasks.models import CreateTaskResult, GetTaskParams, GetTaskResult
from mcp.shared.exceptions import MCPError

from favlo import __version__
from favlo.images import image_path, load_image
from favlo.json_model import dumps_for_model
from favlo.local_uri import file_uri, is_file_uri, materialize
from favlo.media_catalog import (
    PipelineRecord,
    ToolSpec,
    build_catalog,
    default_folder,
)
from favlo.memory_mcp import (
    FORGET_DESCRIPTION,
    RECALL_DESCRIPTION,
    REMEMBER_DESCRIPTION,
    forget_text,
    open_store,
    recall_text,
    remember_text,
)
from favlo.music import load_music, music_dir
from favlo.videos import load_video, video_path
from favlo.voice import load_voice, voice_dir

# Favlo tool descriptions stay at or under this length (stricter than OpenAI's 1024).
# The MCP spec does not cap description length.
TOOL_DESCRIPTION_MAX_CHARS = 256
# Parameter property descriptions in tools/list stay at or under this length.
TOOL_PARAM_DESCRIPTION_MAX_CHARS = 48
_POLL = timedelta(seconds=2)
_TASK = TaskConfig(mode="optional", poll_interval=_POLL)
_ANNOTATIONS = mcp_types.ToolAnnotations(
    read_only_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_AUDIO = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
_IMAGE_EXT = frozenset({"png"})
_VIDEO_EXT = frozenset({"mp4"})
_STILL = (
    "image is an id starting with image_, "
    '{"type": "image", "data": "<base64 png>"}, or an absolute file:// URI. '
    "A bare path and a favlo:// URI are refused."
)
# Only these keys are stills. A prompt with a slash (f/1.8) is not a path.
_STILL_KEYS = frozenset({"image", "source", "target", "subject", "background"})
_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def build_mcp(
    dispatch: Any,
    state: dict[str, Any],
    embed: Any,
) -> FastMCP:
    """Listener tools, then one tool per registered ``mcp_tools()`` row.

    A task tool dispatches ``image``, ``video``, ``music``, or ``voice``.
    The body carries the handler attribute and the registry folder.
    """
    mcp = FastMCP("Favlo", version=__version__)
    tasks = FavloTasks(state)
    mcp.add_extension(tasks)

    @mcp.tool(task=False)
    async def media_status(taskId: str) -> dict[str, Any]:
        """The queue record for one task.

        A finished task includes the media name and the favlo:// link.
        An unknown id is an error.
        """
        return await tasks.status(taskId)

    @mcp.tool(task=False)
    async def media_cancel(taskId: str) -> dict[str, Any]:
        """Cancel one task. A second call returns the same status."""
        return await tasks.cancel(taskId)

    store = open_store(state.get("memory_home"))

    @mcp.tool(task=False, description=REMEMBER_DESCRIPTION)
    def remember(text: str, client: str = "") -> str:
        who = client.strip() or None
        return remember_text(store, embed, text, client=who)

    @mcp.tool(task=False, description=RECALL_DESCRIPTION)
    def recall(query: str, client: str = "") -> Any:
        who = client.strip() or None
        return recall_text(store, embed, query, client=who)

    @mcp.tool(task=False, description=FORGET_DESCRIPTION)
    def forget(id: str, client: str = "") -> str:  # noqa: A002
        who = client.strip() or None
        return forget_text(store, id, client=who)

    mount_plugins(mcp, dispatch, state)

    @mcp.resource("favlo://images/{image_id}", mime_type="image/png")
    def read_image(image_id: str) -> ResourceResult:
        """The stored PNG plus its standardized metadata JSON when present."""
        stem, _ext = _split_suffix(image_id, _IMAGE_EXT)
        raw = load_image(stem, home=_home(tasks.state, "image_home"))
        if raw is None:
            message = "stored image is missing"
            raise ToolError(message)
        contents: list[ResourceContent] = [
            ResourceContent(raw, mime_type="image/png"),
        ]
        from favlo.media_meta import load_media

        record = load_media(stem, home=_home(tasks.state, "image_home"))
        if record is not None:
            contents.append(
                ResourceContent(
                    dumps_for_model(record),
                    mime_type="application/json",
                )
            )
        return ResourceResult(contents)

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
        else:
            payload = getattr(got, "result", None)
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            view.update(_finished_handles(payload))
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
    """Kinds published by ``mcp_tools()``. A module without tools is not media."""
    declare = getattr(module, "mcp_tools", None)
    if not callable(declare):
        return []
    try:
        raw = declare()
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    found: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("handler")
        if not isinstance(name, str):
            continue
        kind = getattr(getattr(module, name, None), "kind", None)
        if kind in {"image", "video", "music", "voice"} and kind not in found:
            found.append(kind)
    return found


def _home(state: dict[str, Any], key: str) -> Path | None:
    home = state.get(key)
    if isinstance(home, Path):
        return home
    return None


class CatalogTool(Tool):
    """One published plugin tool. The schema is the module's, not a signature."""

    _spec: ToolSpec = PrivateAttr()
    _record: PipelineRecord = PrivateAttr()
    _dispatch: Any = PrivateAttr()
    _state: dict[str, Any] = PrivateAttr()

    def __init__(
        self,
        name: str,
        spec: ToolSpec,
        record: PipelineRecord,
        dispatch: Any,
        state: dict[str, Any],
    ) -> None:
        task = _TASK if spec.task else TaskConfig(mode="forbidden")
        super().__init__(
            name=name,
            description=spec.description,
            parameters=spec.parameters,
            task_config=task,
            annotations=_ANNOTATIONS if spec.task else None,
            meta={"pipeline": record.pipeline, "result": spec.result},
        )
        self._spec = spec
        self._record = record
        self._dispatch = dispatch
        self._state = state

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        sent = arguments or {}
        _check_schema(self._spec.parameters, sent)
        body = _request_body(sent, self._record, self._spec, self._state)
        if not self._spec.task:
            return _pack(self._spec.result, {"result": body}, sent, self._state)
        kind = self._spec.kind
        payload = await _run_task(self._dispatch, kind, body)
        return _pack(self._spec.result, payload, sent, self._state)


def mount_plugins(mcp: FastMCP, dispatch: Any, state: dict[str, Any]) -> None:
    """Replace plugin tools with the current registry catalog."""
    previous = state.get("plugin_tools")
    if isinstance(previous, list):
        for name in previous:
            try:
                mcp._local_provider.remove_tool(str(name))
            except KeyError:
                continue
    catalog = build_catalog(
        description_max=TOOL_DESCRIPTION_MAX_CHARS,
        param_max=TOOL_PARAM_DESCRIPTION_MAX_CHARS,
    )
    names: list[str] = []
    for name, record, spec in catalog.published():
        mcp.add_tool(CatalogTool(name, spec, record, dispatch, state))
        names.append(name)
    state["plugin_tools"] = names
    state["media_catalog"] = catalog
    state["remount_media"] = lambda: mount_plugins(mcp, dispatch, state)


def _check_schema(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be an object")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required") or []
    for key in required:
        if key not in arguments:
            raise ToolError(f"{key} is required")
    for key, value in arguments.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            if schema.get("additionalProperties") is False:
                raise ToolError(f"unknown argument {key}")
            continue
        expected = prop.get("type")
        if not isinstance(expected, str):
            continue
        if expected == "integer" and isinstance(value, bool):
            raise ToolError(f"{key} must be an integer")
        if expected == "number" and isinstance(value, bool):
            raise ToolError(f"{key} must be a number")
        kind = _TYPES.get(expected)
        if kind is not None and not isinstance(value, kind):
            raise ToolError(f"{key} must be a {expected}")


def _request_body(
    arguments: dict[str, Any],
    record: PipelineRecord,
    spec: ToolSpec,
    state: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"include_image", "include_audio", "session"}:
            continue
        body[key] = _argument(value, state, still=key in _STILL_KEYS)
    named = body.get("model")
    try:
        body["model"] = _media_model_id(named, record, spec)
    except LookupError as exc:
        raise ToolError(str(exc) or "unknown model") from exc
    body["pipeline"] = record.pipeline
    body["handler"] = spec.handler
    return body


def _media_model_id(named: Any, record: PipelineRecord, spec: ToolSpec) -> str:
    """Public model id for ``resolve``. Accepts the folder name or its alias."""
    from favlo.registry import load_registry

    data = load_registry()
    if isinstance(named, str) and named.strip():
        for folder in record.folders:
            entry = data["models"].get(folder)
            if not isinstance(entry, dict):
                continue
            alias = entry.get("alias")
            public = alias if isinstance(alias, str) and alias else folder
            if named == folder or named == public:
                return public
        raise LookupError("unknown model")
    if named in (None, ""):
        folder = default_folder(record, spec)
        entry = data["models"].get(folder)
        if isinstance(entry, dict):
            alias = entry.get("alias")
            if isinstance(alias, str) and alias:
                return alias
        return folder
    raise LookupError("unknown model")


def _argument(value: Any, state: dict[str, Any], *, still: bool = False) -> Any:
    """Pass still fields through ``_still``. Do not materialize first: base64 often
    contains ``/``, and a second path check would refuse a valid ``file://`` read.
    """
    if not still:
        return value
    try:
        return _still(value, state)
    except (RuntimeError, ValueError, OSError) as exc:
        raise ToolError(str(exc)) from exc


def _pack(
    kind: str,
    payload: dict[str, Any],
    arguments: dict[str, Any],
    state: dict[str, Any],
) -> ToolResult:
    if kind in {"image", "video", "audio"}:
        return _media_tool_result(payload, state)
    if kind == "json":
        text = payload if isinstance(payload, str) else dumps_for_model(payload)
        return ToolResult([mcp_types.TextContent(type="text", text=text)])
    raise ToolError("unknown result")


async def _run_task(dispatch: Any, kind: str, body: dict[str, Any]) -> dict[str, Any]:
    info = get_task_context()
    task_id = info.task_id if info is not None else ""
    if task_id and await _already_cancelled(task_id):
        return {"status": "cancelled"}
    from favlo.pipelines import execute_job

    payload = await execute_job(task_id, lambda: dispatch(kind, body, task_id))
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
    if isinstance(image, str) and is_file_uri(image):
        loaded = materialize(image)
        if not isinstance(loaded, str) or not loaded:
            raise RuntimeError(_STILL)
        return loaded
    if isinstance(image, dict):
        if image.get("type") != "image":
            raise RuntimeError(_STILL)
        data = image.get("data")
        if isinstance(data, str) and is_file_uri(data):
            data = materialize(data)
        if not isinstance(data, str) or not data:
            raise RuntimeError(_STILL)
        return data
    if not isinstance(image, str) or not image or _is_path(image):
        raise RuntimeError(_STILL)
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


def _media_tool_result(payload: dict[str, Any], state: dict[str, Any]) -> ToolResult:
    """Text JSON of ``metadata`` and ``mediaUri``. No inline bytes."""
    if payload.get("status") == "cancelled":
        return ToolResult([mcp_types.TextContent(type="text", text="cancelled")])
    from favlo.media_result import pack_media

    packed = pack_media(payload, state)
    text = dumps_for_model(packed, drop_empty=False)
    return ToolResult([mcp_types.TextContent(type="text", text=text)])


def _load_local(value: Any) -> Any:
    """Read a file:// argument. Other values pass through."""
    try:
        return materialize(value)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except OSError as exc:
        raise ToolError("local file is missing") from exc


def _local_link(
    state: dict[str, Any], kind: str, item_id: str, ext: str, mime: str
) -> mcp_types.ResourceLink | None:
    """A file:// link when the stored file is on this machine."""
    path = _stored_path(state, kind, item_id, ext)
    if path is None or not path.is_file():
        return None
    filename = f"{item_id}.{ext}"
    return mcp_types.ResourceLink(
        type="resource_link",
        uri=file_uri(path),
        name=filename,
        mime_type=mime,
    )


def _stored_path(state: dict[str, Any], kind: str, item_id: str, ext: str) -> Path | None:
    homes = {
        "images": "image_home",
        "videos": "video_home",
        "music": "music_home",
        "voice": "voice_home",
    }
    home = _home(state, homes[kind])
    if home is None:
        from favlo.logs import favlo_home

        home = favlo_home()
    try:
        if kind == "images":
            return image_path(home, item_id)
        if kind == "videos":
            return video_path(home, item_id)
        folder = music_dir(home) if kind == "music" else voice_dir(home)
    except ValueError:
        return None
    if "/" in item_id or "\\" in item_id or item_id in {".", ".."}:
        return None
    return folder / f"{item_id}.{ext}"


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


_HANDLE_LINK = re.compile(
    r"(?:favlo://(?:images|videos|voice|music)/[A-Za-z0-9_-]+\.(?:png|mp4|mp3|wav|flac)"
    r"|file://[^\s\"'\\]+\.(?:png|mp4|mp3|wav|flac))"
)
_HANDLE_NAME = re.compile(r"\b(?:image|video|music|voice)_[A-Za-z0-9_-]+\b")


def _finished_handles(result: Any) -> dict[str, list[str]]:
    """Media names and file or favlo links in a finished task result."""
    links: list[str] = []
    names: list[str] = []
    _collect_handles(result, links, names)
    found: dict[str, list[str]] = {}
    if links:
        found["links"] = links
    if names:
        found["names"] = names
    return found


def _collect_handles(value: Any, links: list[str], names: list[str]) -> None:
    if isinstance(value, str):
        for match in _HANDLE_LINK.findall(value):
            if match not in links:
                links.append(match)
        for match in _HANDLE_NAME.findall(value):
            if match not in names:
                names.append(match)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_handles(item, links, names)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_handles(item, links, names)


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
    paused = isinstance(record, dict) and bool(record.get("paused"))
    if position is not None:
        fields["position"] = position
    if position is not None or paused:
        fields["statusMessage"] = "task is ongoing"
    if paused:
        fields["stage"] = "paused"
    job = record.get("job") if isinstance(record, dict) else None
    if not isinstance(job, dict) and position is None and not paused:
        job = _live_job(state)
    if not isinstance(job, dict):
        return fields
    if job.get("progress") is not None:
        fields["progress"] = job.get("progress")
    if job.get("eta") is not None:
        fields["eta"] = job.get("eta")
    if not paused and isinstance(job.get("stage"), str):
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
