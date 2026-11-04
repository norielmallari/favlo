"""Show a finished generate in the chat after the turn has ended.

``Client.call_tool`` would poll ``tasks/get`` inside the AG-UI run, so the UI
stays on Thinking until the file exists. AG-UI has no message after
``RUN_FINISHED``. This module ends the tool loop once a generate task is
claimed, then polls ``tasks/get`` on its own client.

When the file is ready the host places gallery copies, then starts a silent
AG-UI turn (fenced host cue) so the model writes one short message that
includes the ``favlo://`` links. When a later run is open on that thread,
finished links wait in a queue and announce in that run's ``finally``.
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
_URI = re.compile(
    r"^favlo://(images|videos|voice|music)/([A-Za-z0-9_-]+)\.(png|mp4|mp3|wav|flac)$"
)
_TASKS: ContextVar[list[tuple[str, str, str]] | None] = ContextVar(
    "favlo_media_tasks", default=None
)
_QUEUED: dict[str, list[str]] = {}
_QUEUE_LOCK = threading.Lock()
_KIND_BY_TOOL = {
    "generate_image": "image",
    "generate_video_i2v": "video",
    "generate_video_t2v": "video",
    "generate_music": "audio",
    "generate_voice": "audio",
}


class ReplyAfterGenerate(ChatMiddleware):
    """After a generate task is claimed, the next model call is text."""

    async def process(self, context: Any, call_next: Any) -> None:
        if _arm():
            options = getattr(context, "options", None)
            if isinstance(options, dict):
                options["tool_choice"] = "none"
        await call_next()


def media_middleware() -> list[ReplyAfterGenerate]:
    """The middleware the web chat client installs."""
    return [ReplyAfterGenerate()]


def note_task(task_id: str, target: str, *, kind: str = "image") -> None:
    """Remember one claimed generate so the turn can finish and the file can follow."""
    current = _TASKS.get()
    if current is None:
        return
    cleaned = task_id.strip()
    url = target.strip()
    media = kind.strip() or "image"
    if not cleaned or not url:
        return
    current.append((cleaned, url, media))


def take_tasks() -> list[tuple[str, str, str]]:
    """The claimed generates for this turn. A second call returns none."""
    current = _TASKS.get()
    if not current:
        return []
    found = list(current)
    current.clear()
    return found


def schedule(app: Any, thread_id: str) -> None:
    """Poll claimed generates after the run, and flush deferred links when idle."""
    found = take_tasks()
    cleaned = thread_id.strip()
    if found and cleaned:
        for task_id, _target, kind in found:
            _publish_pending(app, cleaned, task_id, kind, "working")
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


async def deliver(app: Any, thread_id: str, tasks: list[tuple[str, str, str]]) -> None:
    """Place finished files, then ask the model for a short message with the links."""
    try:
        links: list[str] = []
        task_ids: list[str] = []
        for task_id, target, kind in tasks:
            for link in await _wait_links(app, thread_id, target, task_id, kind):
                if link not in links:
                    links.append(link)
            task_ids.append(task_id)
        if not links:
            return
        await _place_and_announce(app, thread_id, links, task_ids=task_ids)
    except Exception:
        _LOG.exception("media delivery failed")


async def flush_queued(app: Any, thread_id: str) -> None:
    """Announce deferred links once no run is open on this thread."""
    try:
        store = app.state.store
        if await asyncio.to_thread(_busy, store, thread_id):
            return
        texts = _take_queued(thread_id)
        if not texts:
            return
        for text in texts:
            found = favlo_links(text)
            if not found:
                continue
            await _place_and_announce(app, thread_id, found)
    except Exception:
        _LOG.exception("media flush failed")


def store_links(
    store: Any,
    thread_id: str,
    text: str,
    *,
    task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Append one assistant row when those links are not already in the snapshot.

    Fallback when a silent announce cannot run. When a run is still open on
    this thread, queue the text for that run's ``finally``.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if _busy(store, thread_id):
        _queue(thread_id, cleaned)
        return []
    # Host cues also list the links; only a visible assistant row counts as delivered.
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
    events = list(store.record_assistant(thread_id, cleaned, task_ids=task_ids))
    events.extend(place_gallery_media(store, thread_id, found))
    return events


def announce_cue(links: list[str]) -> str:
    """Host cue for the silent follow-up that shows the finished media."""
    listed = "\n".join(link.strip() for link in links if link.strip())
    return (
        "These media files are ready. Write one short message to the user, then "
        "repeat every favlo:// link below on its own line exactly as written so "
        "the chat shows the files. Do not call tools.\n"
        f"{listed}"
    )


async def _place_and_announce(
    app: Any,
    thread_id: str,
    links: list[str],
    *,
    task_ids: list[str] | None = None,
) -> None:
    store = app.state.store
    bus = app.state.bus
    cleaned = [link.strip() for link in links if link.strip()]
    if not cleaned:
        return
    events = await asyncio.to_thread(place_gallery_media, store, thread_id, cleaned)
    for event in events:
        await bus.publish(event)
    if await asyncio.to_thread(_busy, store, thread_id):
        _queue(thread_id, "\n".join(cleaned))
        return
    if await _announce(app, thread_id, cleaned):
        return
    fallback = await asyncio.to_thread(
        store_links,
        store,
        thread_id,
        "\n".join(cleaned),
        task_ids=task_ids,
    )
    for event in fallback:
        await bus.publish(event)


async def _announce(app: Any, thread_id: str, links: list[str]) -> bool:
    """Silent AG-UI turn that asks the model to speak with the favlo:// links.

    Returns True when the announce ran (even if the model omitted a link; the
    caller still checks the snapshot and may fall back).
    """
    from web_api.runs import perform
    from web_api.store import Conflict, NotFound, Rejected

    store = app.state.store
    bus = app.state.bus
    if not await asyncio.to_thread(store.session_exists, thread_id):
        return True
    if await asyncio.to_thread(_links_present, store, thread_id, links):
        return True
    run_id = guid()
    try:
        started = await asyncio.to_thread(
            store.begin_greeting,
            thread_id,
            run_id,
            announce_cue(links),
        )
    except Conflict:
        _queue(thread_id, "\n".join(links))
        job = asyncio.create_task(_flush_later(app, thread_id))
        pending = getattr(getattr(app, "state", None), "tasks", None)
        if isinstance(pending, set):
            pending.add(job)
            job.add_done_callback(pending.discard)
        return True
    except (NotFound, Rejected):
        return False
    for event in started.get("events") or []:
        await bus.publish(event)
    cancel = asyncio.Event()
    cancels = getattr(getattr(app, "state", None), "cancels", None)
    if isinstance(cancels, dict):
        cancels[run_id] = cancel
    try:
        await perform(app, cancel, run_id)
    except Exception:
        _LOG.exception("media announce failed for %s", thread_id)
        return False
    finally:
        if isinstance(cancels, dict):
            cancels.pop(run_id, None)
    if await asyncio.to_thread(_links_present, store, thread_id, links):
        return True
    return False


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
    """``favlo://`` media URIs in a ``tasks/get`` result, in first-seen order."""
    found: list[str] = []
    _walk(payload, found)
    return found


def kind_for_tool(tool_name: str) -> str:
    """Chat/filmstrip kind for one generate tool name."""
    return _KIND_BY_TOOL.get(tool_name, "image")


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
        for match in _LINK.findall(payload):
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


def _publish_pending(app: Any, thread_id: str, task_id: str, kind: str, status: str) -> None:
    bus = getattr(getattr(app, "state", None), "bus", None)
    store = getattr(getattr(app, "state", None), "store", None)
    if bus is None or store is None:
        return
    try:
        events = store.note_media_pending(thread_id, task_id, kind, status)
    except Exception:
        _LOG.exception("media.pending failed for %s", task_id)
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for event in events:
        loop.create_task(bus.publish(event))


def _write_link(folder: Any, link: str) -> dict[str, Any] | None:
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


async def _wait_links(
    app: Any,
    thread_id: str,
    target: str,
    task_id: str,
    kind: str,
) -> list[str]:
    from fastmcp_tasks.client import TasksClientExtension
    from fastmcp_tasks.client_models import ClientGetTaskResult, GetTaskRequest, GetTaskRequestParams
    from mcp import Client

    delay = 0.5
    last_status = ""
    async with Client(target, extensions=[TasksClientExtension()]) as client:
        while True:
            if not await asyncio.to_thread(app.state.store.session_exists, thread_id):
                return []
            try:
                raw = await client.session.send_request(
                    GetTaskRequest(params=GetTaskRequestParams(task_id=task_id)),
                    ClientGetTaskResult,
                )
            except Exception:
                _LOG.exception("tasks/get failed for %s", task_id)
                return []
            status = str(getattr(raw, "status", "") or "")
            message = str(getattr(raw, "status_message", "") or "").strip()
            shown = message or status or "working"
            if shown != last_status:
                last_status = shown
                _publish_pending(app, thread_id, task_id, kind, shown)
            if status == "completed":
                return favlo_links(getattr(raw, "result", None))
            if status in {"failed", "cancelled"}:
                return []
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)
