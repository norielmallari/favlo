"""Gallery folder tools. Injected only on that workspace session."""

from __future__ import annotations

import base64
import re
from typing import Any

from favlo.images import image_path
from favlo.logs import favlo_home
from web_api.folder_port import folder_for
from web_api.store import Rejected
from web_api.tool_names import qualified

SOURCE = "gallery"

INSTRUCTIONS = (
    "Image files live under the gallery root the user set. Paths are relative "
    "to that root. Do not use a host path or another workspace id. After an "
    "image generate tool finishes, call "
    f"{qualified(SOURCE, 'write_image')} with the favlo:// URI as source and a path "
    "under the root. Use "
    f"{qualified(SOURCE, 'create_folder')}, {qualified(SOURCE, 'rename_folder')}, "
    f"{qualified(SOURCE, 'move_folder')}, and {qualified(SOURCE, 'delete_folder')} "
    "for directories inside the root. Those tools cannot change the root itself."
)

_IMAGE_URI = re.compile(r"^favlo://images/([A-Za-z0-9_-]+)\.png$")


def tools(store: Any, workspace_id: str, bus: Any | None = None) -> list[object]:
    return [
        _list_tool(store, workspace_id),
        _write_tool(store, workspace_id, bus),
        _create_tool(store, workspace_id, bus),
        _rename_tool(store, workspace_id, bus),
        _move_tool(store, workspace_id, bus),
        _delete_tool(store, workspace_id, bus),
    ]


def _list_tool(store: Any, workspace_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "list_folder"))
    def list_folder(path: str = "") -> str:
        """List images and subfolders under a path relative to the gallery root."""
        try:
            listing = folder_for(store, workspace_id).list(path)
        except Rejected as exc:
            return str(exc)
        return _dump(listing)

    return list_folder


def _write_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "write_image"))
    async def write_image(
        path: str,
        source: str = "",
        data: str = "",
        expected_revision: str = "",
    ) -> str:
        """Write an image under the gallery root.

        Pass source as a favlo://images/{id}.png URI from generate_image, or data
        as base64. path is relative to the gallery root.
        """
        try:
            payload = image_bytes(source=source or None, data=data or None)
            written = folder_for(store, workspace_id).write(
                path,
                payload,
                expected_revision=expected_revision or None,
            )
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, written)
        return _dump(written)

    return write_image


def _create_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "create_folder"))
    async def create_folder(path: str) -> str:
        """Create a directory inside the gallery root."""
        try:
            created = folder_for(store, workspace_id).create_folder(path)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, created)
        return _dump(created)

    return create_folder


def _rename_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "rename_folder"))
    async def rename_folder(path: str, name: str) -> str:
        """Rename a directory inside the gallery root."""
        try:
            renamed = folder_for(store, workspace_id).rename_folder(path, name)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, renamed)
        return _dump(renamed)

    return rename_folder


def _move_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "move_folder"))
    async def move_folder(path: str, destination: str) -> str:
        """Move a directory to another path inside the gallery root."""
        try:
            moved = folder_for(store, workspace_id).move_folder(path, destination)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, moved)
        return _dump(moved)

    return move_folder


def _delete_tool(store: Any, workspace_id: str, bus: Any | None) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "delete_folder"))
    async def delete_folder(path: str) -> str:
        """Delete a directory inside the gallery root. The root itself cannot be deleted."""
        try:
            deleted = folder_for(store, workspace_id).delete_folder(path)
        except Rejected as exc:
            return str(exc)
        await _notify(store, workspace_id, bus, deleted)
        return _dump(deleted)

    return delete_folder


def image_bytes(*, source: str | None, data: str | None) -> bytes:
    if data:
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise Rejected("Image bytes must be base64.") from exc
    if source:
        match = _IMAGE_URI.match(source.strip())
        if match is None:
            raise Rejected("source must be a favlo image URI.")
        path = image_path(favlo_home(), match.group(1))
        if not path.is_file():
            raise Rejected("That image is missing.")
        return path.read_bytes()
    raise Rejected("Provide bytes or source.")


def _dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload)


async def _notify(store: Any, workspace_id: str, bus: Any | None, payload: dict[str, Any]) -> None:
    events = store.note_gallery(workspace_id, payload)
    if bus is None:
        return
    for event in events:
        await bus.publish(event)
