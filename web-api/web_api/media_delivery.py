"""Show a finished generate in the chat after the turn has ended.

``Client.call_tool`` would poll ``tasks/get`` inside the AG-UI run, so the UI
stays on Thinking until the file exists. AG-UI has no message after
``RUN_FINISHED``. A claimed generate returns its task id and later tools in
that turn stay available. This module then polls ``tasks/get`` on its own client.

When a task reaches a terminal status the host starts a silent AG-UI turn.
The file is copied into the session root under the name the agent passed.
``adapt_media_cue`` turns that task result into ``<media_completed>``,
``<media_cancelled>``, or ``<media_error>`` around the file name. The chat's
tools decide what to do with it. When a later run is open on that thread,
the cue waits until the thread is idle.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from agent_framework import ChatMiddleware

from web_api.ids import guid

_LOG = logging.getLogger(__name__)
_SCOPE = "favlo"
_LINK = re.compile(
    r"favlo://(?:images|videos|voice|music)/[A-Za-z0-9_-]+\.(?:png|mp4|mp3|wav|flac)"
)
_FILE_URI = re.compile(r"file://[^\s\"'\\]+\.(?:png|mp4|mp3|wav|flac)")
_URI = re.compile(
    r"^favlo://(images|videos|voice|music)/([A-Za-z0-9_-]+)\.(png|mp4|mp3|wav|flac)$"
)
_TASKS: ContextVar[list[tuple[str, str, str, str]] | None] = ContextVar(
    "favlo_media_tasks", default=None
)
_QUEUED: dict[str, list[str]] = {}
_QUEUE_LOCK = threading.Lock()
def kind_for_tool(tool_name: str) -> str:
    """Chat and filmstrip kind for one generate tool name."""
    local = tool_name.split("__")[-1]
    if "video" in local:
        return "video"
    if "music" in local or "voice" in local:
        return "audio"
    return "image"


class TrackGenerateTasks(ChatMiddleware):
    """Remember generate task ids from this turn. Later tool calls stay available."""

    async def process(self, context: Any, call_next: Any) -> None:
        _arm()
        await call_next()


def media_middleware() -> list[TrackGenerateTasks]:
    """The middleware the web chat client installs."""
    return [TrackGenerateTasks()]


def resolve_pending_task(kind: str, hint: str) -> str:
    """The in-flight task for this tile.

    A generate id is long, and a copied id can miss by a character. The file
    name, or the only open task of this kind, is the tile the host will clear.
    """
    cleaned = hint.strip()
    current = _TASKS.get() or []
    media = kind.strip().lower()
    if any(task_id == cleaned for task_id, _url, _task_kind, _name in current):
        return cleaned
    named = [
        task_id
        for task_id, _url, task_kind, name in current
        if name == cleaned and task_kind == media
    ]
    if len(named) == 1:
        return named[0]
    same = [task_id for task_id, _url, task_kind, _name in current if task_kind == media]
    if len(same) == 1:
        return same[0]
    return cleaned


def bound_task(kind: str, hint: str) -> str | None:
    """The open task this hint names. A finished task is not open."""
    resolved = resolve_pending_task(kind, hint)
    current = _TASKS.get() or []
    media = kind.strip().lower()
    open_ids = {
        task_id
        for task_id, _url, task_kind, _name in current
        if task_id == resolved and task_kind == media
    }
    if open_ids:
        return resolved
    return None


def note_task(
    task_id: str,
    target: str,
    *,
    kind: str = "image",
    filename: str = "",
) -> None:
    """Remember one claimed generate so the turn can finish and the file can follow."""
    current = _TASKS.get()
    if current is None:
        return
    cleaned = task_id.strip()
    url = target.strip()
    media = kind.strip() or "image"
    if not cleaned or not url:
        return
    current.append((cleaned, url, media, filename.strip()))


def take_tasks() -> list[tuple[str, str, str, str]]:
    """The claimed generates for this turn. A second call returns none."""
    current = _TASKS.get()
    if not current:
        return []
    found = list(current)
    current.clear()
    return found


def schedule(app: Any, thread_id: str) -> None:
    """Poll claimed generates after the run, and flush a deferred cue when idle."""
    found = take_tasks()
    cleaned = thread_id.strip()
    if found and cleaned:
        job = asyncio.create_task(deliver(app, cleaned, found))
        pending = getattr(getattr(app, "state", None), "tasks", None)
        if isinstance(pending, set):
            pending.add(job)
            job.add_done_callback(pending.discard)
    if cleaned:
        flush = asyncio.create_task(flush_queued(app, cleaned))
        pending = getattr(getattr(app, "state", None), "tasks", None)
        if isinstance(pending, set):
            pending.add(flush)
            flush.add_done_callback(pending.discard)


async def deliver(
    app: Any, thread_id: str, tasks: list[tuple[str, str, str, str]]
) -> None:
    """Cue the model when each task reaches a terminal status."""
    from web_api.workspace_files import release_output

    try:
        finished: list[tuple[str, str, str]] = []
        store = app.state.store
        root = await asyncio.to_thread(_session_root, store, thread_id)
        for task_id, target, kind, filename in tasks:
            try:
                status, result = await _wait_status(app, thread_id, target, task_id)
                if not status:
                    continue
                name = filename.strip()
                if name and status == "completed":
                    events = await asyncio.to_thread(
                        _place_named, store, thread_id, name, favlo_links(result)
                    )
                    if events is not None:
                        if events:
                            await _publish(app, events)
                        await _drop_tile(app, store, thread_id, task_id, kind, "completed", name)
                        finished.append((task_id, "completed", name))
                        continue
                    status = "failed"
                if name:
                    await _drop_tile(
                        app,
                        store,
                        thread_id,
                        task_id,
                        kind,
                        "cancelled" if status != "failed" else "failed",
                        name,
                    )
                finished.append((task_id, status, name))
            finally:
                if root is not None and filename.strip():
                    release_output(root, filename)
        text = task_cue(finished)
        if text:
            await _cue(app, thread_id, text)
    except Exception:
        _LOG.exception("media delivery failed")


async def flush_queued(app: Any, thread_id: str) -> None:
    """Send a deferred task cue once no run is open on this thread."""
    try:
        store = app.state.store
        if await asyncio.to_thread(_busy, store, thread_id):
            return
        texts = _take_queued(thread_id)
        for text in texts:
            cleaned = text.strip()
            if cleaned:
                await _cue(app, thread_id, cleaned)
    except Exception:
        _LOG.exception("media flush failed")


def store_links(
    store: Any,
    thread_id: str,
    text: str,
    *,
    task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Append one assistant row when that text is not already in the snapshot.

    When a run is still open on this thread, queue the text for that run's
    ``finally``.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if _busy(store, thread_id):
        _queue(thread_id, cleaned)
        return []
    found = favlo_links(cleaned)
    if found and _links_present(store, thread_id, found):
        return []
    if not found:
        snapshot = store.thread_snapshot(_SCOPE, thread_id) or {}
        messages = snapshot.get("messages") if isinstance(snapshot, dict) else None
        if isinstance(messages, list) and cleaned in _blob(messages):
            return []
    store.append_thread_snapshot_message(
        _SCOPE,
        thread_id,
        {"id": guid(), "role": "assistant", "content": cleaned},
    )
    return list(store.record_assistant(thread_id, cleaned, task_ids=task_ids))


def adapt_media_cue(filename: str, status: str) -> str:
    """The cue for one finished generate. The body is the agent's file name."""
    from web_api.host_cues import fence_media_status

    return fence_media_status(filename, status)


def media_turn_messages(text: str, *, message_id: str) -> list[dict[str, Any]] | None:
    """The only messages the model receives after a task finishes.

    A media fence is that turn. The task result is not included. None when
    ``text`` is not a media fence.
    """
    from web_api.host_cues import host_cue_message, is_media_fence

    if not is_media_fence(text):
        return None
    return [host_cue_message(text, message_id=message_id)]


def task_cue(finished: list[tuple[str, str, str]]) -> str:
    """One media fence per finished task. A missing file name is left out."""
    blocks: list[str] = []
    for _task_id, status, filename in finished:
        block = adapt_media_cue(filename, status)
        if block:
            blocks.append(block)
    return "\n".join(blocks)


async def _cue(app: Any, thread_id: str, text: str) -> None:
    """Silent AG-UI turn. The cue fences the task id."""
    from web_api.runs import perform
    from web_api.store import Conflict, NotFound, Rejected

    store = app.state.store
    bus = app.state.bus
    cleaned = text.strip()
    if not cleaned:
        return
    if not await asyncio.to_thread(store.session_exists, thread_id):
        return
    if await asyncio.to_thread(_busy, store, thread_id):
        _queue(thread_id, cleaned)
        return
    run_id = guid()
    try:
        started = await asyncio.to_thread(
            store.begin_greeting,
            thread_id,
            run_id,
            cleaned,
        )
    except Conflict:
        _queue(thread_id, cleaned)
        job = asyncio.create_task(_flush_later(app, thread_id))
        pending = getattr(getattr(app, "state", None), "tasks", None)
        if isinstance(pending, set):
            pending.add(job)
            job.add_done_callback(pending.discard)
        return
    except (NotFound, Rejected):
        return
    for event in started.get("events") or []:
        await bus.publish(event)
    cancel = asyncio.Event()
    cancels = getattr(getattr(app, "state", None), "cancels", None)
    if isinstance(cancels, dict):
        cancels[run_id] = cancel
    try:
        await perform(app, cancel, run_id)
    except Exception:
        _LOG.exception("media cue failed for %s", thread_id)
    finally:
        if isinstance(cancels, dict):
            cancels.pop(run_id, None)


async def _flush_later(app: Any, thread_id: str) -> None:
    await asyncio.sleep(1.0)
    await flush_queued(app, thread_id)


def _links_present(store: Any, thread_id: str, links: list[str]) -> bool:
    """True when every link appears in a visible assistant row (not a host cue)."""
    from web_api.host_cues import is_host_cue

    snapshot = store.thread_snapshot(_SCOPE, thread_id) or {}
    messages = snapshot.get("messages") if isinstance(snapshot, dict) else None
    if not isinstance(messages, list):
        return False
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if is_host_cue(message):
            continue
        role = str(message.get("role") or "")
        if role not in {"assistant", "AI", "ai"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    blob = "\n".join(parts)
    return all(link in blob for link in links)


def place_gallery_media(store: Any, thread_id: str, links: list[str]) -> list[dict[str, Any]]:
    """Copy finished media into a gallery folder and publish ``gallery.updated``."""
    workspace_id = store.gallery_for_session(thread_id)
    if not workspace_id or not store.has_local_folder(workspace_id):
        return []
    from web_api.folder_port import folder_for
    from web_api.gallery_metadata import write_sidecar
    from web_api.workspace_tools import append_media_placement

    folder = folder_for(store, workspace_id)
    events: list[dict[str, Any]] = []
    for link in links:
        written = _write_link(folder, link)
        if written is None:
            continue
        if written.get("kind") == "image" and written.get("has_sidecar") is False:
            meta = _image_meta(link, written["path"])
            if meta is not None:
                write_sidecar(folder.root, written["path"], meta)
                written = folder.stat(written["path"])
        state = append_media_placement(store, workspace_id, written["path"])
        payload = {**written, "state": state} if state is not None else written
        events.extend(store.note_gallery(workspace_id, payload))
    return events


def favlo_links(payload: Any) -> list[str]:
    """``favlo://`` and ``file://`` media URIs in a ``tasks/get`` result, in first-seen order."""
    found: list[str] = []
    _walk(payload, found)
    return found


def _arm() -> list[tuple[str, str, str]]:
    current = _TASKS.get()
    if current is None:
        current = []
        _TASKS.set(current)
    return current


def _busy(store: Any, thread_id: str) -> bool:
    try:
        return bool(store.running_run_ids(thread_id))
    except Exception:
        return False


def _queue(thread_id: str, text: str) -> None:
    with _QUEUE_LOCK:
        _QUEUED.setdefault(thread_id, []).append(text)


def _take_queued(thread_id: str) -> list[str]:
    with _QUEUE_LOCK:
        return _QUEUED.pop(thread_id, [])


def _walk(payload: Any, found: list[str]) -> None:
    if isinstance(payload, str):
        for pattern in (_LINK, _FILE_URI):
            for match in pattern.findall(payload):
                if match not in found:
                    found.append(match)
        return
    if isinstance(payload, dict):
        for value in payload.values():
            _walk(value, found)
        return
    if isinstance(payload, list):
        for item in payload:
            _walk(item, found)


def _blob(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def _write_link(folder: Any, link: str) -> dict[str, Any] | None:
    local = _file_bytes(link)
    if local is not None:
        name, payload = local
        try:
            if folder.resolve(name, create=True).exists():
                return folder.stat(name)
            return folder.write(name, payload)
        except Exception:
            _LOG.exception("gallery write failed for %s", link)
            return None
    match = _URI.match(link.strip())
    if match is None:
        return None
    folder_name, stem, ext = match.group(1), match.group(2), match.group(3)
    relative = f"{stem}.{ext}"
    try:
        payload = _media_bytes(folder_name, stem, ext)
    except Exception:
        _LOG.exception("read failed for %s", link)
        return None
    if payload is None:
        return None
    try:
        if folder.resolve(relative, create=True).exists():
            return folder.stat(relative)
        return folder.write(relative, payload)
    except Exception:
        _LOG.exception("gallery write failed for %s", link)
        return None


def _file_bytes(link: str) -> tuple[str, bytes] | None:
    if not link.startswith("file://"):
        return None
    from urllib.parse import unquote, urlsplit

    from favlo.local_uri import read_file_uri

    path = Path(unquote(urlsplit(link.strip()).path))
    if path.suffix.lower() not in {".png", ".mp4", ".mp3", ".wav", ".flac"}:
        return None
    try:
        raw = read_file_uri(link.strip())
    except (OSError, ValueError):
        return None
    return path.name, raw


def _media_bytes(folder_name: str, stem: str, ext: str) -> bytes | None:
    from favlo.logs import favlo_home

    home = favlo_home()
    if folder_name == "images" and ext == "png":
        from favlo.images import image_path

        path = image_path(home, stem)
        return path.read_bytes() if path.is_file() else None
    if folder_name == "videos" and ext == "mp4":
        from favlo.videos import video_path

        path = video_path(home, stem)
        return path.read_bytes() if path.is_file() else None
    if folder_name == "music":
        from favlo.music import load_music

        loaded = load_music(stem, ext, home=home)
        return None if loaded is None else loaded[0]
    if folder_name == "voice":
        from favlo.voice import load_voice

        loaded = load_voice(stem, ext, home=home)
        return None if loaded is None else loaded[0]
    return None


def _image_meta(link: str, relative: str) -> dict[str, Any] | None:
    local = _file_bytes(link)
    if local is not None:
        stem = Path(local[0]).stem
        if not stem.startswith("image_"):
            return None
        from favlo.media_meta import load_media

        meta = load_media(stem)
        if meta is None:
            return None
        meta = dict(meta)
        meta["filename"] = Path(relative).name
        return meta
    match = _URI.match(link.strip())
    if match is None or match.group(1) != "images":
        return None
    from favlo.media_meta import load_media

    meta = load_media(match.group(2))
    if meta is None:
        return None
    meta = dict(meta)
    meta["filename"] = Path(relative).name
    return meta


def _session_root(store: Any, thread_id: str) -> Path | None:
    from web_api.workspace_files import local_root

    return local_root(store, thread_id)


def _place_named(
    store: Any, thread_id: str, filename: str, links: list[str]
) -> list[dict[str, Any]] | None:
    """Copy one finished file onto the session root under the chosen name.

    ``None`` means the file was not written. An empty list is a home write
    with nothing to publish. A workspace write publishes ``gallery.updated``.
    """
    from web_api.folder_port import LocalFolder
    from web_api.gallery_metadata import write_sidecar
    from web_api.workspace_files import local_root, remember_source

    root = local_root(store, thread_id)
    if root is None or not links:
        return None
    link = links[0]
    local = _file_bytes(link)
    if local is not None:
        payload = local[1]
    else:
        match = _URI.match(link.strip())
        if match is None:
            return None
        payload = _media_bytes(match.group(1), match.group(2), match.group(3))
    if payload is None:
        return None
    folder = LocalFolder(str(root))
    try:
        written = folder.write(filename, payload)
    except Exception:
        _LOG.exception("session write failed for %s", filename)
        return None
    remember_source(root, filename, link)
    if written.get("kind") == "image" and written.get("has_sidecar") is False:
        meta = _image_meta(link, written["path"])
        if meta is not None:
            write_sidecar(folder.root, written["path"], meta)
            written = folder.stat(written["path"])
    workspace = store.workspace_for_session(thread_id)
    if workspace and store.has_local_folder(workspace["id"]):
        from web_api.workspace_tools import append_media_placement

        path = str(written.get("path") or filename)
        state = append_media_placement(store, workspace["id"], path)
        payload = {**written, "state": state} if state is not None else written
        return list(store.note_gallery(workspace["id"], payload))
    return []


async def _drop_tile(
    app: Any,
    store: Any,
    thread_id: str,
    task_id: str,
    kind: str,
    status: str,
    filename: str,
) -> None:
    """Clear the generating tile, including one keyed by the file name."""
    ids = [task_id]
    name = filename.strip()
    if name and name != task_id:
        ids.append(name)
    for item in ids:
        await _publish(
            app,
            await asyncio.to_thread(
                store.note_media_pending,
                thread_id,
                item,
                kind,
                status,
            ),
        )


async def _publish(app: Any, events: list[dict[str, Any]]) -> None:
    bus = app.state.bus
    for event in events:
        await bus.publish(event)


async def _wait_status(
    app: Any, thread_id: str, target: str, task_id: str
) -> tuple[str, Any]:
    """Poll until the task is completed, failed, or cancelled."""
    from fastmcp_tasks.client import TasksClientExtension
    from fastmcp_tasks.client_models import (
        ClientGetTaskResult,
        GetTaskRequest,
        GetTaskRequestParams,
    )
    from mcp import Client

    delay = 0.5
    async with Client(target, extensions=[TasksClientExtension()]) as client:
        while True:
            if not await asyncio.to_thread(app.state.store.session_exists, thread_id):
                return "", None
            try:
                raw = await client.session.send_request(
                    GetTaskRequest(params=GetTaskRequestParams(task_id=task_id)),
                    ClientGetTaskResult,
                )
            except Exception:
                _LOG.exception("tasks/get failed for %s", task_id)
                return "", None
            status = str(getattr(raw, "status", "") or "")
            if status in {"completed", "failed", "cancelled"}:
                return status, getattr(raw, "result", None)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)
