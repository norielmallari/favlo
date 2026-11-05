"""Show a favlo:// file in the chat transcript. Gallery sessions do not register it."""

from __future__ import annotations

from typing import Any

from web_api.ids import guid
from web_api.media_delivery import favlo_links

INSTRUCTIONS = (
    "display_media shows one file from this chat's folder. "
    "When you receive <media_completed>, call display_media with that file name. "
    "Do not paste the file name or a link. "
    "When you receive <media_error> or <media_cancelled>, tell the user in one short sentence."
)

_SCOPE = "favlo"


def tools(store: Any, session_id: str, bus: Any | None = None) -> list[object]:
    from agent_framework import tool

    @tool(approval_mode="never_require", name="display_media")
    async def display_media(filename: str) -> str:
        """Show one file from this chat's folder.

        filename is the name from <media_completed>.
        """
        from web_api.store import Rejected
        from web_api.workspace_files import file_name, local_root, source_link

        root = local_root(store, session_id)
        if root is None:
            return "This chat has no folder."
        try:
            name = file_name(filename)
        except Rejected as exc:
            return str(exc)
        if not (root / name).is_file():
            return "That file is not in this folder."
        cleaned = source_link(root, name)
        found = favlo_links(cleaned)
        if found != [cleaned]:
            return "That file has no media link."
        events = _append(store, session_id, cleaned)
        if bus is not None:
            for event in events:
                await bus.publish(event)
        return "Showing that file in the chat."

    return [display_media]


def _append(store: Any, session_id: str, link: str) -> list[dict[str, Any]]:
    snapshot = store.thread_snapshot(_SCOPE, session_id) or {}
    messages = snapshot.get("messages") if isinstance(snapshot, dict) else None
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and link in str(message.get("content") or ""):
                return []
    store.append_thread_snapshot_message(
        _SCOPE,
        session_id,
        {"id": guid(), "role": "assistant", "content": link},
    )
    return list(store.record_assistant(session_id, link))
