"""Workspace placement tool for the shell catalog."""

from __future__ import annotations

from typing import Any

from favlo.json_model import dumps_for_model
from web_api.folder_port import folder_for
from web_api.gallery_metadata import is_media
from web_api.placement import selection_for_elements, validate
from web_api.store import Rejected
from web_api.tool_names import qualified

SOURCE = "workspace"


def tools(store: Any, workspace_id: str, guide_id: str = "gallery") -> list[object]:
    return [_place_tool(store, workspace_id, guide_id)]


def _place_tool(store: Any, workspace_id: str, guide_id: str) -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "place"))
    def workspace_place(elements: list[dict[str, Any]], selection: dict[str, Any] | None = None) -> str:
        """Place only catalog elements on the workspace shell.

        Pass the full elements list. Each element needs key, element, slot, and
        props. Gallery uses preview on slot stage and filmstrip on slot rail.
        Example filmstrip props: {"title": "", "items": [{"path": "Albums/zebra.png"}]}.
        When the filmstrip has items, selection is required:
        {"kind": "file", "path": "<one of those paths>"} so preview shows that file.
        If selection is omitted or not on the strip, Favlo selects the first path.
        Filmstrip paths must come from gallery_find_media or gallery_search_media.
        Do not describe the layout in prose.
        """
        prior = None
        session_id = store.workspace_session_id(workspace_id)
        if session_id:
            prior = store.agui_state(session_id)
        if not isinstance(elements, list) or not elements:
            return dumps_for_model(
                {
                    "error": "elements must be a non-empty list of catalog placements",
                    "ignored": [],
                }
            )
        payload = {
            "favlo_workspace": {
                "version": 1,
                "guide": guide_id,
                "elements": elements,
                "selection": selection,
            }
        }
        state, ignored = validate(payload, guide_id=guide_id, prior=prior)
        workspace = state["favlo_workspace"]
        try:
            folder = folder_for(store, workspace_id)
        except Rejected as exc:
            return dumps_for_model({"error": str(exc), "ignored": ignored})
        missing: list[str] = []
        for element in workspace.get("elements") or []:
            if element.get("element") != "filmstrip":
                continue
            items = []
            for item in (element.get("props") or {}).get("items") or []:
                path = str(item.get("path") or "")
                resolved = _resolve_media_path(folder, path)
                if resolved is None:
                    missing.append(path)
                    continue
                row = dict(item)
                row["path"] = resolved
                items.append(row)
            element["props"]["items"] = items
        selection_obj = workspace.get("selection")
        if isinstance(selection_obj, dict):
            path = str(selection_obj.get("path") or "")
            resolved = _resolve_media_path(folder, path) if path else None
            if path and resolved is None:
                missing.append(path)
                selection_obj = None
            elif resolved is not None:
                selection_obj = {"kind": "file", "path": resolved}
        workspace["selection"] = selection_for_elements(
            list(workspace.get("elements") or []),
            selection_obj if isinstance(selection_obj, dict) else None,
        )
        if session_id:
            store.save_agui_state(session_id, state)
        if missing:
            ignored.extend(f"missing path {path}" for path in missing)
        return dumps_for_model({"state": state, "ignored": ignored})

    return workspace_place


def _media_exists(folder: Any, path: str) -> bool:
    return _resolve_media_path(folder, path) is not None


def _resolve_media_path(folder: Any, path: str) -> str | None:
    """Exact relative path, or a unique basename match under the gallery folder."""
    if not path or not is_media(path):
        return None
    try:
        folder.resolve(path, file=True)
        return path
    except Rejected:
        pass
    name = path.rsplit("/", 1)[-1]
    try:
        found = folder.find_images([name])
    except Rejected:
        return None
    if len(found) == 1:
        return found[0]
    return None
