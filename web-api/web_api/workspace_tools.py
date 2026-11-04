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


def ensure_opening_filmstrip(store: Any, workspace_id: str) -> None:
    """Place every image, with its sidecar caption, through the filmstrip clamps.

    The model does not call this. ``apply_placement`` keeps the same clamps as
    ``workspace_place``: 60 items, unique paths, and an 80-character caption.
    """
    from web_api.gallery_metadata import read_sidecar

    try:
        folder = folder_for(store, workspace_id)
    except Rejected:
        return
    listing = folder.walk_media(kinds=["image"], record_limit=5000)
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in listing.get("media") or []:
        path = str(row.get("path") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        item: dict[str, str] = {"path": path}
        side = read_sidecar(folder.root, path) or {}
        caption = str(side.get("title") or side.get("description") or "").strip()
        if caption:
            item["caption"] = caption
        items.append(item)
    if not items:
        return
    # A longer list is rejected by the 64 KiB state clamp before the filmstrip
    # clamp runs. The first 60 are what that clamp keeps.
    items = items[:60]
    apply_placement(
        store,
        workspace_id,
        [
            {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
            {
                "key": "filmstrip",
                "element": "filmstrip",
                "slot": "rail",
                "props": {"title": "", "items": items},
            },
        ],
        {"kind": "file", "path": items[0]["path"]},
        guide_id="gallery",
    )


def _place_tool(store: Any, workspace_id: str, guide_id: str) -> object:
    from agent_framework import Content, tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "place"))
    def workspace_place(
        elements: list[dict[str, Any]], selection: dict[str, Any] | None = None
    ) -> str | Content:
        """Place only catalog elements on the workspace shell.

        Pass the full elements list. Each element needs key, element, slot, and
        props. Gallery uses preview on slot stage and filmstrip on slot rail.
        Example filmstrip props: {"title": "", "items": [{"path": "Albums/zebra.png"}]}.
        When the filmstrip has items, selection is required:
        {"kind": "file", "path": "<one of those paths>"} so preview shows that file.
        If selection is omitted or not on the strip, Favlo selects the first path.
        Filmstrip paths must come from gallery_find_media or gallery_search_media.
        Do not describe the layout in prose.
        A path string in items is the same as {"path": that string}.
        """
        result = apply_placement(store, workspace_id, elements, selection, guide_id=guide_id)
        text = dumps_for_model(result)
        state = result.get("state") if isinstance(result, dict) else None
        if isinstance(state, dict):
            from agent_framework_ag_ui import state_update

            return state_update(text, state=state)
        return text

    return workspace_place


def sync_media_placement(
    store: Any,
    workspace_id: str,
    *,
    source: str,
    destination: str | None,
) -> dict[str, Any] | None:
    """Point the filmstrip and selection at a moved image, or drop a deleted one.

    ``destination`` is the new relative path. Omit it to remove ``source``.
    The session snapshot and the stored placement both take the result.
    """
    from web_api.placement import retarget_media

    session_id = store.workspace_session_id(workspace_id)
    if not session_id:
        return None
    current = _placement_state(store, session_id)
    if current is None:
        return None
    updated = retarget_media(current, source, destination)
    if updated is None:
        return None
    store.save_agui_state(session_id, updated)
    _write_snapshot_state(store, session_id, updated)
    return updated


def apply_placement(
    store: Any,
    workspace_id: str,
    elements: list[dict[str, Any]] | None,
    selection: dict[str, Any] | None,
    *,
    guide_id: str,
) -> dict[str, Any]:
    """Validate a placement, keep paths that exist, and store it on the workspace chat."""
    session_id = store.workspace_session_id(workspace_id)
    prior = store.agui_state(session_id) if session_id else None
    if not isinstance(elements, list) or not elements:
        return {"error": "elements must be a non-empty list of catalog placements", "ignored": []}
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
        return {"error": str(exc), "ignored": ignored}
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
    return {"state": state, "ignored": ignored}


def _placement_state(store: Any, session_id: str) -> dict[str, Any] | None:
    """The placement the gallery is showing: snapshot state, else the stored copy."""
    from web_api.agui import _SNAPSHOT_SCOPE

    document = store.thread_snapshot(_SNAPSHOT_SCOPE, session_id)
    if isinstance(document, dict):
        shared = document.get("state")
        if isinstance(shared, dict) and isinstance(shared.get("favlo_workspace"), dict):
            return shared
    stored = store.agui_state(session_id)
    if isinstance(stored, dict) and isinstance(stored.get("favlo_workspace"), dict):
        return stored
    return None


def _write_snapshot_state(store: Any, session_id: str, state: dict[str, Any]) -> None:
    from web_api.agui import _SNAPSHOT_SCOPE

    document = store.thread_snapshot(_SNAPSHOT_SCOPE, session_id)
    if not isinstance(document, dict):
        return
    store.save_thread_snapshot(_SNAPSHOT_SCOPE, session_id, {**document, "state": state})


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
