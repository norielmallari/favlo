"""Gallery folder tools. Injected only on that workspace session."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from favlo.images import image_path
from favlo.logs import favlo_home
from web_api.folder_port import folder_for
from web_api.store import Rejected
from web_api.tool_names import qualified

SOURCE = "gallery"


def _root_media_path(path: str) -> str:
    """Reject nested paths so media stays at the local folder root."""
    cleaned = path.strip().strip("/").replace("\\", "/")
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned:
        raise Rejected("Media paths stay at the local folder root.")
    return cleaned


_SELECTED_LINE = re.compile(
    r"^The selected media on this turn is (.+)\. This image means that path\.$"
)
_NONE_SELECTED = "No media is selected on this turn."


def selection_cue(path: str | None, description: str = "") -> str:
    """Path cue and sidecar metadata text for a newly selected image."""
    cleaned = (path or "").strip()
    if not cleaned:
        return _NONE_SELECTED
    lines = [f"The selected media on this turn is {cleaned}. This image means that path."]
    text = description.strip()
    if text:
        lines.append(text)
    return "\n".join(lines)


def announced_selection(text: str) -> str | None:
    """Selection path from one cue body.

    Returns the path, an empty string when that cue cleared selection, or None
    when the text had no selection cue.
    """
    if not text:
        return None
    found: str | None = None
    saw = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == _NONE_SELECTED:
            found = ""
            saw = True
            continue
        match = _SELECTED_LINE.match(line)
        if match:
            found = match.group(1).strip()
            saw = True
    return found if saw else None


def announced_selection_from_history(messages: list[Any] | None) -> str | None:
    """Most recent selection cue in durable thread history.

    Prefers fenced host ``user`` cues (see ``host_cues``). Legacy ``developer``
    and ``system`` rows still count for older threads. Returns the path, an
    empty string when the latest cue cleared selection, or None when history
    has no cue.
    """
    from web_api.host_cues import host_cue_text

    if not messages:
        return None
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        text: str | None = None
        host = host_cue_text(item)
        if host is not None:
            text = host
        elif role in {"user", "developer", "system"}:
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {None, "text", "input_text"}
                )
        if text is None:
            continue
        found = announced_selection(text)
        if found is not None:
            return found
    return None


def selection_announcement(
    current: str | None,
    previous: str | None,
    description: str = "",
) -> str:
    """Cue when selection changed since the previous announced path. Empty when unchanged."""
    cur = (current or "").strip()
    if previous is None:
        if not cur:
            return ""
        return selection_cue(cur, description)
    prev = previous.strip()
    if cur == prev:
        return ""
    if not cur:
        return _NONE_SELECTED
    return selection_cue(cur, description)


_TOKEN = re.compile(
    r"(?<![\w./:\\-])((?:[\w.-]+/)*[\w.-]+\.(?:png|jpe?g))(?![\w])",
    re.IGNORECASE,
)
_SPACED_LINE = re.compile(
    r"(?im)^[ \t]*((?:[^/:\n]+/)*[^/:\n]* [^/:\n]*\.(?:png|jpe?g))[ \t]*$"
)


def named_images(text: str) -> list[str]:
    """Png and jpg names in the message, in reading order.

    A line that is itself a path may contain spaces. A token inside that line
    stays in the list until a search shows the path exists.
    """
    return [name for _start, name, _spaced in _mentions(text)]


def _mentions(text: str) -> list[tuple[int, str, bool]]:
    """Start offset, name, and whether the name is a whole line that contains a space."""
    found: list[tuple[int, str, bool]] = []
    for match in _SPACED_LINE.finditer(text):
        found.append((match.start(), match.group(1).strip(), True))
    for match in _TOKEN.finditer(text):
        found.append((match.start(1), match.group(1), False))
    found.sort(key=lambda item: (item[0], not item[2]))
    return found


def images_in_message(store: Any, session_id: str, text: str) -> list[tuple[bytes, str]]:
    """Png and jpg bytes for filenames in the message that the gallery tools can read.

    The search root is the local folder those tools list. A name that is not
    there is skipped. A session without that folder returns nothing.
    """
    workspace_id = store.gallery_for_session(session_id)
    if not workspace_id:
        return []
    try:
        folder = folder_for(store, workspace_id)
        relative_paths = folder.find_images(_names_to_open(folder, text))
    except Rejected:
        return []
    found: list[tuple[bytes, str]] = []
    for relative in relative_paths:
        try:
            payload = folder.read(relative)
            media = str(folder.stat(relative)["media_type"])
        except Rejected:
            continue
        if media not in {"image/png", "image/jpeg"}:
            continue
        found.append((payload, media))
    return found


def _names_to_open(folder: Any, text: str) -> list[str]:
    """Filenames to read. A spaced path that exists hides tokens inside that line."""
    mentions = _mentions(text)
    spaced_hits = {
        name
        for name, _path in folder.locate([name for _start, name, spaced in mentions if spaced])
    }
    hidden: list[tuple[int, int]] = []
    for start, name, spaced in mentions:
        if not spaced or name not in spaced_hits:
            continue
        line_end = text.find("\n", start)
        hidden.append((start, len(text) if line_end < 0 else line_end))
    names: list[str] = []
    seen: set[str] = set()
    for start, name, spaced in mentions:
        if spaced and name not in spaced_hits:
            continue
        if not spaced and any(left <= start < right for left, right in hidden):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def preview_file(store: Any, session_id: str, path: str) -> tuple[bytes, str] | None:
    """Bytes and media type for one image path in the local folder."""
    workspace_id = store.gallery_for_session(session_id)
    if not workspace_id:
        return None
    try:
        folder = folder_for(store, workspace_id)
        payload = folder.read(path)
        media = str(folder.stat(path)["media_type"])
    except Rejected:
        return None
    return payload, media

_IMAGE_URI = re.compile(r"^favlo://images/([A-Za-z0-9_-]+)\.png$")


def tools(store: Any, workspace_id: str, bus: Any | None = None) -> list[object]:
    """Every gallery tool is eager so the tool schema stays a stable prompt prefix."""
    return [
        _find_tool(store, workspace_id),
        _search_tool(store, workspace_id),
        _get_metadata_tool(store, workspace_id),
        _load_image_tool(store, workspace_id),
        _write_tool(store, workspace_id, bus),
        _update_metadata_tool(store, workspace_id),
        _search_unlike_tool(store, workspace_id),
        _move_image_tool(store, workspace_id, bus),
        _delete_image_tool(store, workspace_id, bus),
        _theme_tool(store, workspace_id, bus),
        _select_media_tool(store, workspace_id, bus),
        _show_generating_tool(store, workspace_id, bus),
        _update_generating_tool(store, workspace_id, bus),
    ]


def _theme_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "set_theme"))
    async def gallery_set_theme(theme: str) -> str:
        """Set this gallery's color scheme.

        theme is favlo, rose-pine, or catppuccin. Light, dark, and system stay
        the user's appearance setting.
        """
        try:
            result, events = store.set_workspace_theme(workspace_id, theme)
        except Rejected as exc:
            return _dump({"error": str(exc)})
        if bus is not None:
            for event in events:
                await bus.publish(event)
        return _dump(result)

    gallery_set_theme.__name__ = "gallery_set_theme"
    return gallery_set_theme


def _show_generating_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "show_generating_media"))
    async def show_generating_media(task_id: str, kind: str = "image") -> str:
        """Show a generating tile on the filmstrip for one task.

        task_id is the file name passed to generate, or the id that call
        returned. kind is image, video, or audio.
        """
        from web_api.media_delivery import bound_task

        session_id = store.workspace_session_id(workspace_id)
        if not session_id:
            return "This gallery has no chat."
        media = kind.strip().lower()
        if media not in {"image", "video", "audio"}:
            media = "image"
        bound = bound_task(media, task_id)
        if bound is None:
            return "That file is not being made."
        events = store.note_media_pending(session_id, bound, media, "working")
        await _publish(bus, events)
        return "The filmstrip shows a tile for that task."

    return show_generating_media


def _update_generating_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    from web_api.media_delivery import place_gallery_media

    @tool(approval_mode="never_require", name=qualified(SOURCE, "update_generating_media"))
    async def update_generating_media(task_id: str, link: str = "") -> str:
        """Save a finished file and replace its generating tile.

        task_id is the id from the host cue. link is the mediaUri from the
        generate result. Omit link to clear the tile when the task failed
        or was cancelled.
        """
        session_id = store.workspace_session_id(workspace_id)
        if not session_id:
            return "This gallery has no chat."
        cleaned = task_id.strip()
        uri = link.strip()
        if not cleaned:
            return "Pass the task id."
        if not uri:
            events = store.note_media_pending(session_id, cleaned, "image", "cancelled")
            await _publish(bus, events)
            return "The generating tile is cleared."
        events = place_gallery_media(store, session_id, [uri])
        await _publish(bus, events)
        if not events:
            return "That link could not be saved."
        done = store.note_media_pending(session_id, cleaned, "image", "completed")
        await _publish(bus, done)
        return "The file is on the filmstrip."

    return update_generating_media


async def _publish(bus: Any | None, events: list[dict[str, Any]]) -> None:
    if bus is None:
        return
    for event in events:
        await bus.publish(event)


def _bind_or_message(store: Any, workspace_id: str) -> Any | str:
    if not store.has_local_folder(workspace_id):
        return "Bind a local folder first."
    try:
        return folder_for(store, workspace_id)
    except Rejected as exc:
        return str(exc)


def _file_names(rows: list[Any]) -> list[str]:
    """File names only, in the order given."""
    names: list[str] = []
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("name") or "").strip()
        name = path.rsplit("/", 1)[-1]
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _slim_media_rows(listing: dict[str, Any]) -> dict[str, Any]:
    """File names only. Drop kind, captions, and the duplicate images array."""
    return {
        "names": _file_names(list(listing.get("media") or [])),
        "truncated": bool(listing.get("truncated")),
    }


def _select_media_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    from web_api.workspace_tools import select_media_placement

    @tool(approval_mode="never_require", name=qualified(SOURCE, "select_media"))
    async def select_media(name: str) -> str:
        """Select one image on the filmstrip.

        name is a file name at the folder root. The strip already shows the folder.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            relative = _root_media_path(name)
            written = folder.stat(relative)
        except Rejected as exc:
            return str(exc)
        path = str(written.get("path") or relative)
        state = select_media_placement(store, workspace_id, path)
        if state is None:
            return "This gallery has no chat."
        await _notify(store, workspace_id, bus, {"path": path, "state": state})
        return "That file is selected."

    return select_media


def _find_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "find_media"))
    def find_media(
        path: str = "",
        name_contains: str = "",
        kinds: list[str] | None = None,
    ) -> str:
        """List file names at the local folder root.

        The result is names only. Call gallery_get_metadata for sidecar fields.
        kinds may be image, audio, and video. Prefer gallery_search_media for meaning.
        path is ignored; listing is always the folder root.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        listing = folder.walk_media("", name_contains=name_contains, kinds=kinds, depth_limit=0)
        if isinstance(listing, dict):
            listing = _slim_media_rows(listing)
        return _dump(listing)

    # Alias kept for prompts that still say find_images.
    find_media.__name__ = "gallery_find_media"
    return find_media


def _search_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "search_media"))
    def search_media(query: str, limit: int = 12, kinds: list[str] | None = None) -> str:
        """Semantic search over indexed media metadata prompts for this workspace."""
        return _search_hits(store, workspace_id, query, limit=limit, kinds=kinds, unlike=False)

    return search_media


def _search_unlike_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "search_unlike_media"))
    def search_unlike_media(query: str, limit: int = 12, kinds: list[str] | None = None) -> str:
        """Semantic anti-search: indexed media farthest from the query vector.

        Use when the request wants results that do not match the meaning.
        """
        return _search_hits(store, workspace_id, query, limit=limit, kinds=kinds, unlike=True)

    return search_unlike_media


def _search_hits(
    store: Any,
    workspace_id: str,
    query: str,
    *,
    limit: int,
    kinds: list[str] | None,
    unlike: bool,
) -> str:
    folder = _bind_or_message(store, workspace_id)
    if isinstance(folder, str):
        return folder
    from favlo.memory_mcp import embed_http
    from web_api.gallery_index import GalleryIndex

    vector = embed_http("http://127.0.0.1:8129/v1", query)
    if isinstance(vector, str):
        return vector
    index = GalleryIndex(store.path.with_name("gallery_index.db"))
    hits = index.search(workspace_id, vector, limit=limit, kinds=kinds, unlike=unlike)
    return _dump(
        {
            "names": _file_names(hits),
            "indexed": index.count(workspace_id),
            "unlike": unlike,
        }
    )


def _get_metadata_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "get_metadata"))
    def get_metadata(path: str) -> str:
        """Return the clamped sidecar for one media path under the local folder.

        path is relative to the local folder. Call when the user asks about
        captions or other fields, or before gallery_update_metadata.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            relative = str(folder.stat(path)["path"])
        except Rejected as exc:
            return _dump({"error": "missing", "message": str(exc)})
        from web_api.gallery_metadata import read_sidecar

        stored = read_sidecar(folder.root, relative)
        if stored is None:
            return _dump({"error": "missing_sidecar", "path": relative, "message": "No sidecar yet."})
        return _dump(
            {
                "path": relative,
                "title": stored.get("title") or "",
                "description": stored.get("description") or "",
                "tags": stored.get("tags") or [],
                "place": stored.get("place") or "",
                "people": stored.get("people") or [],
                "time": stored.get("time") or "",
                "objects": stored.get("objects") or [],
                "style": stored.get("style") or "",
                "colors": stored.get("colors") or [],
                "text_in_image": stored.get("text_in_image") or "",
                "nsfw": bool(stored.get("nsfw")),
            }
        )

    return get_metadata


def _load_image_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import Content, tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "load_image"))
    def load_image(path: str) -> str | Content:
        """Load one png or jpeg so the model can see its pixels.

        path is relative to the local folder. Use when the sidecar description
        is not enough. Do not call only because the path is selected.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            relative = str(folder.stat(path)["path"])
            payload = folder.read(relative)
            media = str(folder.stat(relative)["media_type"])
        except Rejected as exc:
            return str(exc)
        if media not in {"image/png", "image/jpeg"}:
            return "Only png and jpeg can be loaded for the model."
        return Content.from_data(data=payload, media_type=media, name=Path(relative).name)

    return load_image


def _update_metadata_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "update_metadata"))
    def update_metadata(
        path: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        place: str | None = None,
        people: list[str] | None = None,
        time: str | None = None,
        objects: list[str] | None = None,
        style: str | None = None,
        colors: list[str] | None = None,
        text_in_image: str | None = None,
        nsfw: bool | None = None,
    ) -> str:
        """Update one image's sidecar. Omitted fields stay as they are.

        path is relative to the local folder. tags, people, objects, and colors
        replace that whole list when sent. Returns the stored fields.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        changes = {
            "title": title,
            "description": description,
            "tags": tags,
            "place": place,
            "people": people,
            "time": time,
            "objects": objects,
            "style": style,
            "colors": colors,
            "text_in_image": text_in_image,
            "nsfw": nsfw,
        }
        if all(value is None for value in changes.values()):
            return _dump(
                {
                    "error": "missing_fields",
                    "message": "Name at least one metadata field to update.",
                }
            )
        try:
            relative = str(folder.stat(path)["path"])
        except Rejected as exc:
            return _dump({"error": "missing", "message": str(exc)})
        from web_api.gallery_metadata import build_index_sidecar, clamp, kind_for, read_sidecar, write_sidecar

        if kind_for(relative) != "image":
            return _dump({"error": "not_an_image", "message": "Metadata updates are for images."})
        existing = read_sidecar(folder.root, relative)
        if existing is None:
            existing = build_index_sidecar(folder.resolve(relative, file=True), relative=relative)
        merged = dict(existing)
        for key, value in changes.items():
            if value is not None:
                merged[key] = value
        stored = clamp(merged)
        write_sidecar(folder.root, relative, stored)
        from web_api.gallery_embed import index_sidecar

        indexed = index_sidecar(store, workspace_id, folder.root, relative, stored)
        return _dump(
            {
                "path": relative,
                "title": stored.get("title") or "",
                "description": stored.get("description") or "",
                "tags": stored.get("tags") or [],
                "place": stored.get("place") or "",
                "people": stored.get("people") or [],
                "time": stored.get("time") or "",
                "objects": stored.get("objects") or [],
                "style": stored.get("style") or "",
                "colors": stored.get("colors") or [],
                "text_in_image": stored.get("text_in_image") or "",
                "nsfw": bool(stored.get("nsfw")),
                "indexed": indexed,
            }
        )

    return update_metadata


def _write_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "write_image"))
    async def write_image(
        path: str,
        source: str = "",
        data: str = "",
        expected_revision: str = "",
    ) -> str:
        """Write an image under the local folder.

        Pass source as a favlo://images/{id}.png URI from an image tool, or data
        as base64. path is relative to the local folder.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            relative = _root_media_path(path)
            payload, meta = image_payload(source=source or None, data=data or None)
            written = folder.write(
                relative,
                payload,
                expected_revision=expected_revision or None,
            )
            if meta is not None:
                from web_api.gallery_metadata import write_sidecar

                meta = dict(meta)
                meta["filename"] = Path(written["path"]).name
                write_sidecar(folder.root, written["path"], meta)
                written = folder.stat(written["path"])
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, written)
        return _dump(written)

    return write_image


def _move_image_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "move_image"))
    async def move_image(path: str, destination: str) -> str:
        """Rename an image at the local folder root."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            source = _root_media_path(path)
            target = _root_media_path(destination)
            moved = folder.move(source, target)
        except Rejected as exc:
            return str(exc)
        state = _sync_media(store, workspace_id, source=source, destination=str(moved.get("path") or ""))
        await _notify(store, workspace_id, bus, _with_state(moved, state))
        return _stated(_dump(moved), state)

    return move_image


def _delete_image_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "delete_image"))
    async def delete_image(path: str) -> str:
        """Move an image to the Trash. path is a file at the local folder root."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            relative = _root_media_path(path)
            deleted = folder.delete_image(relative)
        except Rejected as exc:
            return str(exc)
        state = _sync_media(store, workspace_id, source=relative, destination=None)
        await _notify(store, workspace_id, bus, _with_state(deleted, state))
        return _stated(_dump(deleted), state)

    return delete_image


def image_bytes(*, source: str | None, data: str | None) -> bytes:
    payload, _meta = image_payload(source=source, data=data)
    return payload


def image_payload(
    *, source: str | None, data: str | None
) -> tuple[bytes, dict[str, Any] | None]:
    """PNG bytes and the standardized sidecar when ``source`` is a generated image."""
    if data:
        try:
            return base64.b64decode(data, validate=True), None
        except (ValueError, TypeError) as exc:
            raise Rejected("Image bytes must be base64.") from exc
    if source:
        match = _IMAGE_URI.match(source.strip())
        if match is None:
            raise Rejected("source must be a favlo image URI.")
        image_id = match.group(1)
        path = image_path(favlo_home(), image_id)
        if not path.is_file():
            raise Rejected("That image is missing.")
        from favlo.media_meta import load_media

        meta = load_media(image_id)
        if meta is not None:
            meta = dict(meta)
            meta["filename"] = Path(path.name).name
        return path.read_bytes(), meta
    raise Rejected("Provide bytes or source.")


def _sync_media(
    store: Any,
    workspace_id: str,
    *,
    source: str,
    destination: str | None,
) -> dict[str, Any] | None:
    from web_api.workspace_tools import sync_media_placement

    return sync_media_placement(
        store,
        workspace_id,
        source=source,
        destination=destination,
    )


def _with_state(payload: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return payload
    return {**payload, "state": state}


def _stated(text: str, state: dict[str, Any] | None) -> str | Any:
    if state is None:
        return text
    from agent_framework_ag_ui import state_update

    return state_update(text, state=state)


def _dump(payload: dict[str, Any]) -> str:
    from favlo.json_model import dumps_for_model

    return dumps_for_model(payload)


async def _notify(store: Any, workspace_id: str, bus: Any | None, payload: dict[str, Any]) -> None:
    from web_api.gallery_embed import request_wake

    events = store.note_gallery(workspace_id, payload)
    request_wake(store, workspace_id, vision=False)
    if bus is None:
        return
    for event in events:
        await bus.publish(event)
