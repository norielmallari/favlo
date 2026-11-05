"""Background job: one concurrent index pass per gallery workspace."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from pathlib import Path
from typing import Any

from web_api.db import LOCAL_BASE_URL
from web_api.folder_port import LocalFolder
from web_api.gallery_index import GalleryIndex
from web_api.gallery_metadata import (
    build_index_sidecar,
    prompt_text,
    read_sidecar,
    sidecar_path,
    write_sidecar,
)
from web_api.gallery_vision import analyze_image

_PASS_CAP = 40
_VISION_CAP = 8
_SLEEP_SECONDS = 30.0
# A generate sidecar is indexed from its prompt first. Vision waits so it
# does not take the chat model while the file is still new.
_VISION_DEBOUNCE_SECONDS = 120.0


def _app_of(target: Any) -> Any | None:
    state = getattr(target, "state", None)
    if state is not None and hasattr(state, "gallery_index_running"):
        return target
    app = getattr(target, "gallery_index_app", None)
    if app is not None and getattr(app, "state", None) is not None:
        return app
    return None


def _ensure_index_state(state: Any) -> tuple[threading.Lock, dict[str, bool]] | None:
    """Return the per-workspace lock and running map, creating them when missing."""
    lock = getattr(state, "gallery_index_lock", None)
    running = getattr(state, "gallery_index_running", None)
    if not isinstance(lock, threading.Lock):
        lock = threading.Lock()
        state.gallery_index_lock = lock
    if not isinstance(running, dict):
        running = {}
        state.gallery_index_running = running
    return lock, running


def request_wake(
    target: Any, workspace_id: str | None, *, vision: bool = True
) -> bool:
    """Start one index pass for ``workspace_id`` unless that workspace is already running.

    Accepts the FastAPI app or a Store that ``watch`` wired to the app.
    ``vision=False`` writes path sidecars and embeds without calling the chat model,
    so a bind or folder mutate does not contend with an open chat turn.
    Returns True when a pass was scheduled, False when skipped or unavailable.
    Never raises: a wake failure must not fail the HTTP request that asked for it.
    """
    try:
        if not workspace_id or not str(workspace_id).strip():
            return False
        workspace_id = str(workspace_id).strip()
        app = _app_of(target)
        if app is None:
            return False
        state = app.state
        store = getattr(state, "store", None)
        if store is None:
            return False
        try:
            row = store.workspace(workspace_id)
        except Exception:
            return False
        if row.get("type") != "gallery":
            return False
        if not store.has_local_folder(workspace_id):
            return False
        ensured = _ensure_index_state(state)
        if ensured is None:
            return False
        lock, running = ensured
        with lock:
            if running.get(workspace_id):
                return False
            running[workspace_id] = True
            state.gallery_index_running = running
            store.gallery_index_running = dict(running)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with lock:
                running.pop(workspace_id, None)
                store.gallery_index_running = dict(running)
            return False

        index = getattr(state, "gallery_index", None)

        async def job() -> None:
            try:
                await asyncio.to_thread(_pass, store, index, workspace_id, vision)
            except Exception:
                pass
            finally:
                with lock:
                    running.pop(workspace_id, None)
                    store.gallery_index_running = dict(running)

        task = loop.create_task(job())
        tasks = getattr(state, "tasks", None)
        if tasks is not None:
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        return True
    except Exception:
        return False


def gallery_index_running(target: Any, workspace_id: str | None = None) -> bool:
    """True when ``workspace_id`` has a pass in flight, or when any pass is in flight if omitted."""
    app = _app_of(target)
    if app is None:
        running = getattr(target, "gallery_index_running", None)
        if isinstance(running, dict):
            if workspace_id:
                return bool(running.get(str(workspace_id).strip()))
            return bool(running)
        return bool(running)
    running = getattr(app.state, "gallery_index_running", {}) or {}
    if workspace_id:
        return bool(running.get(str(workspace_id).strip()))
    return bool(running)


async def watch(app: Any) -> None:
    """Hold the shared index and periodically wake each idle gallery workspace."""
    stop: asyncio.Event = app.state.heartbeat_stop
    store = app.state.store
    store.gallery_index_app = app
    _ensure_index_state(app.state)
    store.gallery_index_running = dict(app.state.gallery_index_running)
    index_path = Path(store.path).with_name("gallery_index.db")
    index = GalleryIndex(index_path)
    app.state.gallery_index = index
    store.gallery_index = index
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=_SLEEP_SECONDS)
            break
        except TimeoutError:
            for row in store.list_workspaces():
                if row.get("type") == "gallery":
                    request_wake(app, row["id"])
    lock, running = _ensure_index_state(app.state) or (threading.Lock(), {})
    with lock:
        running.clear()
        store.gallery_index_running = {}


def run_once(
    store: Any,
    index: GalleryIndex | None = None,
    workspace_id: str | None = None,
    *,
    vision: bool = True,
) -> int:
    """Process one pass for ``workspace_id``, or every gallery when omitted (tests)."""
    if index is None:
        index = GalleryIndex(Path(store.path).with_name("gallery_index.db"))
    if workspace_id:
        return _pass(store, index, workspace_id, vision)
    total = 0
    for row in store.list_workspaces():
        if row.get("type") == "gallery":
            total += _pass(store, index, row["id"], vision)
    return total


def hold_index(target: Any, workspace_id: str) -> bool:
    """Mark one workspace's index as running so the watcher skips it."""
    app = _app_of(target)
    if app is None:
        return False
    ensured = _ensure_index_state(app.state)
    if ensured is None:
        return False
    lock, running = ensured
    for _ in range(40):
        with lock:
            if not running.get(workspace_id):
                running[workspace_id] = True
                store = getattr(app.state, "store", None)
                if store is not None:
                    store.gallery_index_running = dict(running)
                return True
        time.sleep(0.25)
    return False


def release_index(target: Any, workspace_id: str) -> None:
    app = _app_of(target)
    if app is None:
        return
    ensured = _ensure_index_state(app.state)
    if ensured is None:
        return
    lock, running = ensured
    with lock:
        running.pop(workspace_id, None)
        store = getattr(app.state, "store", None)
        if store is not None:
            store.gallery_index_running = dict(running)


def write_path_sidecars(store: Any, workspace_id: str) -> int:
    """Write a path sidecar for every media file. No vision and no pass cap."""
    return _pass(store, None, workspace_id, False, cap=None)


def write_vision_sidecars(store: Any, workspace_id: str) -> int:
    """Caption every image that still lacks a vision description. No vision cap.

    Provisioning does not wait out the generate debounce.
    """
    return _pass(store, None, workspace_id, True, cap=None, vision_cap=None, debounce=False)


def uncaptioned_images(store: Any, workspace_id: str) -> list[str]:
    """Image paths that still have no vision caption."""
    folder = LocalFolder(store.local_location(workspace_id))
    listing = folder.walk_media(depth_limit=0, record_limit=5000)
    pending: list[str] = []
    for item in listing.get("media") or []:
        if str(item.get("kind") or "") != "image":
            continue
        relative = str(item["path"])
        if _needs_vision(read_sidecar(folder.root, relative), relative):
            pending.append(relative)
    return pending


def _pass(
    store: Any,
    index: GalleryIndex | None,
    workspace_id: str,
    vision: bool = True,
    cap: int | None = _PASS_CAP,
    vision_cap: int | None = _VISION_CAP,
    debounce: bool = True,
) -> int:
    if index is None:
        index = GalleryIndex(Path(store.path).with_name("gallery_index.db"))
    try:
        workspace = store.workspace(workspace_id)
    except Exception:
        return 0
    if workspace.get("type") != "gallery":
        return 0
    if not store.has_local_folder(workspace_id):
        return 0
    try:
        folder = LocalFolder(store.local_location(workspace_id))
    except Exception:
        return 0
    worked = 0
    vision_used = 0
    model, base_url, effort = _chat_endpoint(store, workspace_id) if vision else (None, None, None)
    listing = folder.walk_media(depth_limit=0, record_limit=5000 if cap is None else 80)
    for item in listing.get("media") or []:
        if cap is not None and worked >= cap:
            break
        relative = str(item["path"])
        absolute = folder.resolve(relative, file=True)
        kind = str(item.get("kind") or "image")
        allow_vision = False
        under_cap = vision_cap is None or vision_used < vision_cap
        if vision and kind == "image" and model and base_url and under_cap:
            existing = read_sidecar(folder.root, relative)
            fresh = debounce and _generated_too_fresh(absolute, existing)
            if _needs_vision(existing, relative) and not fresh:
                allow_vision = True
        status = _load_and_embed(
            folder=folder,
            index=index,
            workspace_id=workspace_id,
            relative=relative,
            absolute=absolute,
            kind=kind,
            model=model,
            base_url=base_url,
            allow_vision=allow_vision,
            reasoning_effort=effort,
        )
        if status == "deferred":
            continue
        if allow_vision and status != "unchanged":
            vision_used += 1
        if status == "unchanged":
            continue
        worked += 1
    return worked


def _load_and_embed(
    *,
    folder: LocalFolder,
    index: GalleryIndex,
    workspace_id: str,
    relative: str,
    absolute: Path,
    kind: str,
    model: str | None,
    base_url: str | None,
    allow_vision: bool,
    reasoning_effort: str | None = None,
) -> str:
    """Ensure metadata exists, then embed when the prompt is new.

    Returns ``indexed``, ``unchanged``, ``deferred``, or ``failed``.
    """
    sidecar = read_sidecar(folder.root, relative)
    if sidecar is None:
        stamped = _mtime_iso(absolute)
        if kind == "image" and allow_vision and model and base_url:
            sidecar = analyze_image(
                absolute,
                relative=relative,
                model=model,
                base_url=base_url,
                reasoning_effort=reasoning_effort,
            )
            if sidecar is None:
                return "failed"
        elif kind == "image" and model and base_url:
            # Vision available but this pass is over budget; wait for a later wake.
            return "deferred"
        else:
            sidecar = build_index_sidecar(
                absolute,
                relative=relative,
                mtime_iso=stamped,
            )
        write_sidecar(folder.root, relative, sidecar)
    elif (
        kind == "image"
        and allow_vision
        and model
        and base_url
        and _needs_vision(sidecar, relative)
    ):
        # Replace a path or generation sidecar with a vision caption.
        enriched = analyze_image(
            absolute,
            relative=relative,
            model=model,
            base_url=base_url,
            existing=sidecar,
            reasoning_effort=reasoning_effort,
        )
        if enriched is not None:
            sidecar = enriched
            write_sidecar(folder.root, relative, sidecar)

    prompt = prompt_text(sidecar)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    prior = index.row(workspace_id, relative)
    if prior is not None and prior.get("prompt_hash") == digest:
        return "unchanged"
    vector = _embed(prompt)
    if vector is None:
        return "failed"
    side = sidecar_path(folder.root, relative)
    index.upsert(
        workspace_id,
        relative,
        kind=str(sidecar.get("kind") or kind),
        sidecar_mtime=_mtime_iso(side),
        prompt_text=prompt,
        vector=vector,
    )
    return "indexed"


def _chat_endpoint(store: Any, workspace_id: str) -> tuple[str, str, str | None]:
    """The workspace agent's model and the effort that agent would send.

    An unset effort stays off the request, the same as chat.
    """
    model, base = store.gallery_vision_model(workspace_id)
    effort = store.gallery_vision_effort(workspace_id)
    return model, base, effort


def index_sidecar(store: Any, workspace_id: str, root: str | Path, relative: str, sidecar: dict[str, Any]) -> bool:
    """Embed one sidecar into this gallery's index. False when the vector is missing."""
    prompt = prompt_text(sidecar)
    vector = _embed(prompt)
    if vector is None:
        return False
    try:
        index = GalleryIndex(Path(store.path).with_name("gallery_index.db"))
        index.upsert(
            workspace_id,
            relative,
            kind=str(sidecar.get("kind") or "image"),
            sidecar_mtime=_mtime_iso(sidecar_path(root, relative)),
            prompt_text=prompt,
            vector=vector,
        )
    except Exception:
        return False
    return True


def _embed(text: str) -> list[float] | None:
    from favlo.memory_mcp import embed_http

    try:
        result = embed_http(LOCAL_BASE_URL, text)
    except Exception:
        return None
    if isinstance(result, str):
        return None
    return result


def _path_only(sidecar: dict[str, Any], relative: str) -> bool:
    """True when the sidecar is the path sentence and has not been captioned."""
    kind = str(sidecar.get("kind") or "image")
    description = str(sidecar.get("description") or "")
    return description.startswith(f"A {kind} file at {relative}.")


def _needs_vision(sidecar: dict[str, Any] | None, relative: str) -> bool:
    """True when this image still needs a vision caption."""
    if sidecar is None:
        return True
    if _needs_generation_enrichment(sidecar):
        return True
    return _path_only(sidecar, relative)


def _generated_too_fresh(absolute: Path, sidecar: dict[str, Any] | None) -> bool:
    """True when a generated image was written less than 120 seconds ago."""
    if sidecar is None or sidecar.get("source") != "generate":
        return False
    if not isinstance(sidecar.get("generation"), dict):
        return False
    try:
        age = time.time() - absolute.stat().st_mtime
    except OSError:
        return False
    return age < _VISION_DEBOUNCE_SECONDS


def _needs_generation_enrichment(sidecar: dict[str, Any]) -> bool:
    """True when a generate sidecar still lacks vision-filled semantic fields."""
    if sidecar.get("source") != "generate":
        return False
    if not isinstance(sidecar.get("generation"), dict):
        return False
    if str(sidecar.get("style") or "").strip():
        return False
    objects = sidecar.get("objects")
    return not (isinstance(objects, list) and objects)


def _mtime_iso(path: Path) -> str:
    from datetime import UTC, datetime

    try:
        stamp = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(stamp, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = ["gallery_index_running", "request_wake", "run_once", "watch"]
