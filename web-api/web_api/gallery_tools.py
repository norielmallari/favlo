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

INSTRUCTIONS = (
    "These tools belong to this gallery workspace session. "
    "Media files (images, mp3, mp4) live in the local folder for this gallery. "
    "Paths are relative to that folder. Do not use a host path or another workspace id. "
    "Search with "
    f"{qualified(SOURCE, 'search_media')} when the request is semantic. "
    "Find with "
    f"{qualified(SOURCE, 'find_media')} once for path or name coverage, then "
    "workspace_place once in the same turn. Do not call find twice. "
    "Always pass selection for one filmstrip path when the strip has items. "
    "Do not invent media descriptions; Favlo's index job writes metadata under "
    ".favlo_metadata beside the media (for example Albums/.favlo_metadata/media.mp3.json). "
    "This image or this clip means the selected media path on the turn, when one is named. "
    "After an image generate tool finishes, call "
    f"{qualified(SOURCE, 'write_image')} with the favlo:// URI as source and a path "
    "beside the selected media or at the folder root. "
    "Before folder mutate tools or "
    f"{qualified(SOURCE, 'search_unlike_media')}, call "
    f"{qualified(SOURCE, 'load_folder_tools')} once."
)


def turn_note(folder: str, image: str | None, *, strip_count: int | None = None) -> str:
    """Selected media and filmstrip size for one turn. No open-folder lines."""
    lines: list[str] = []
    if image:
        lines.append(f"The selected media on this turn is {image}. This image means that path.")
    else:
        lines.append("No media is selected on this turn.")
    if strip_count is not None:
        lines.append(f"The filmstrip shows {strip_count} images you placed.")
    return "\n".join(lines)


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
    cold = [
        _search_unlike_tool(store, workspace_id),
        _create_tool(store, workspace_id, bus),
        _rename_tool(store, workspace_id, bus),
        _move_tool(store, workspace_id, bus),
        _move_image_tool(store, workspace_id, bus),
        _delete_image_tool(store, workspace_id, bus),
        _delete_tool(store, workspace_id, bus),
    ]
    return [
        _find_tool(store, workspace_id),
        _search_tool(store, workspace_id),
        _write_tool(store, workspace_id, bus),
        _load_folder_tools(cold),
    ]


def _load_folder_tools(cold: list[object]) -> object:
    from agent_framework import FunctionInvocationContext, tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "load_folder_tools"))
    def load_folder_tools(ctx: FunctionInvocationContext) -> str:
        """Load folder mutate tools and search_unlike_media for later calls in this run."""
        ctx.add_tools(cold)
        return (
            "Loaded gallery_search_unlike_media, gallery_create_folder, gallery_rename_folder, "
            "gallery_move_folder, gallery_move_image, gallery_delete_image, and gallery_delete_folder."
        )

    return load_folder_tools


def _bind_or_message(store: Any, workspace_id: str) -> Any | str:
    if not store.has_local_folder(workspace_id):
        return "Bind a local folder first."
    try:
        return folder_for(store, workspace_id)
    except Rejected as exc:
        return str(exc)


def _find_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "find_media"))
    def find_media(
        path: str = "",
        name_contains: str = "",
        kinds: list[str] | None = None,
    ) -> str:
        """List media under the local folder in one capped recursive walk.

        kinds may be image, audio, and video. Prefer gallery_search_media for meaning.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        listing = folder.walk_media(path, name_contains=name_contains, kinds=kinds)
        if isinstance(listing, dict):
            listing = {
                **listing,
                "next": (
                    "Call workspace_place once with preview on stage and filmstrip on rail "
                    "using these paths; pass selection for one path."
                ),
            }
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
    enriched = []
    for hit in hits:
        row: dict[str, Any] = {
            "path": hit["path"],
            "kind": hit["kind"],
            "score": hit["score"],
        }
        try:
            from web_api.gallery_metadata import read_sidecar

            side = read_sidecar(folder.root, hit["path"]) or {}
            title = str(side.get("title") or "").strip()
            description = str(side.get("description") or "").strip()
            tags = side.get("tags") if isinstance(side.get("tags"), list) else []
            if title:
                row["title"] = title
            if description:
                row["description"] = description
            if tags:
                row["tags"] = tags
        except Rejected:
            pass
        enriched.append(row)
    return _dump({"media": enriched, "indexed": index.count(workspace_id), "unlike": unlike})


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

        Pass source as a favlo://images/{id}.png URI from generate_image, or data
        as base64. path is relative to the local folder.
        """
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            payload, meta = image_payload(source=source or None, data=data or None)
            written = folder.write(
                path,
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


def _create_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "create_folder"))
    async def create_folder(path: str) -> str:
        """Create a directory inside the local folder."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            created = folder.create_folder(path)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, created)
        return _dump(created)

    return create_folder


def _rename_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "rename_folder"))
    async def rename_folder(path: str, name: str) -> str:
        """Rename a directory inside the local folder."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            renamed = folder.rename_folder(path, name)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, renamed)
        return _dump(renamed)

    return rename_folder


def _move_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "move_folder"))
    async def move_folder(path: str, destination: str) -> str:
        """Move a directory to another path inside the local folder."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            moved = folder.move_folder(path, destination)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, moved)
        return _dump(moved)

    return move_folder


def _move_image_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "move_image"))
    async def move_image(path: str, destination: str) -> str:
        """Move an image to another path inside the local folder."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            moved = folder.move(path, destination)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, moved)
        return _dump(moved)

    return move_image


def _delete_image_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "delete_image"))
    async def delete_image(path: str) -> str:
        """Delete an image inside the local folder. path is relative to that folder."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            deleted = folder.delete_image(path)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, deleted)
        return _dump(deleted)

    return delete_image


def _delete_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "delete_folder"))
    async def delete_folder(path: str) -> str:
        """Delete an empty directory inside the local folder. The local folder itself cannot be deleted."""
        folder = _bind_or_message(store, workspace_id)
        if isinstance(folder, str):
            return folder
        try:
            deleted = folder.delete_folder(path)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, deleted)
        return _dump(deleted)

    return delete_folder


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
