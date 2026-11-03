"""Validate and normalize favlo_workspace placement state."""

from __future__ import annotations

import json
from typing import Any

from web_api.catalog import ELEMENTS, get as catalog_get
from web_api.guides import GUIDES

_MAX_BYTES = 64 * 1024


def initial_for(guide_id: str) -> dict[str, Any]:
    guide = GUIDES.get(guide_id) or GUIDES["gallery"]
    return {
        "favlo_workspace": {
            "version": 1,
            "guide": guide["id"],
            "elements": [
                {"key": "preview", "element": "preview", "slot": "stage", "props": {"fit": "contain"}},
                {
                    "key": "filmstrip",
                    "element": "filmstrip",
                    "slot": "rail",
                    "props": {"title": "", "items": []},
                },
            ],
            "selection": None,
        }
    }


def validate(
    payload: dict[str, Any] | None,
    *,
    guide_id: str,
    prior: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return validated state and ignored reasons."""
    ignored: list[str] = []
    if payload is None or not isinstance(payload, dict):
        return initial_for(guide_id), ["missing state"]
    try:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        return prior or initial_for(guide_id), ["state is not JSON"]
    if len(encoded.encode()) > _MAX_BYTES:
        return prior or initial_for(guide_id), ["state exceeds 64 KiB"]
    workspace = payload.get("favlo_workspace")
    if not isinstance(workspace, dict):
        return initial_for(guide_id), ["missing favlo_workspace"]
    if int(workspace.get("version") or 0) != 1:
        return initial_for(guide_id), ["unknown state version"]
    if str(workspace.get("guide") or "") != guide_id:
        return initial_for(guide_id), ["guide mismatch"]
    guide = GUIDES.get(guide_id) or GUIDES["gallery"]
    allowed = set(guide["catalog"])
    raw_elements = workspace.get("elements")
    if not isinstance(raw_elements, list):
        return initial_for(guide_id), ["elements must be a list"]
    cleaned: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    counts: dict[str, int] = {}
    for item in raw_elements:
        if not isinstance(item, dict):
            ignored.append("dropped non-object element")
            continue
        name = str(item.get("element") or "")
        key = str(item.get("key") or name)
        slot = str(item.get("slot") or "")
        spec = catalog_get(name)
        if name not in allowed or spec is None:
            ignored.append(f"unknown element {name}")
            continue
        if slot not in spec["slots"]:
            ignored.append(f"wrong slot for {name}")
            continue
        if key in seen_keys:
            ignored.append(f"duplicate key {key}")
            continue
        count = counts.get(name, 0) + 1
        if count > int(spec["max"]):
            ignored.append(f"too many {name}")
            continue
        props = item.get("props") if isinstance(item.get("props"), dict) else {}
        props = _clamp_props(name, props, ignored)
        seen_keys.add(key)
        counts[name] = count
        cleaned.append({"key": key, "element": name, "slot": slot, "props": props})
    selection = workspace.get("selection")
    if selection is not None:
        if not isinstance(selection, dict) or selection.get("kind") != "file":
            ignored.append("invalid selection")
            selection = None
        else:
            path = str(selection.get("path") or "").strip()
            selection = {"kind": "file", "path": path} if path else None
    # Keep local selection from prior when agent omitted it.
    if selection is None and isinstance(prior, dict):
        prior_ws = prior.get("favlo_workspace")
        if isinstance(prior_ws, dict) and prior_ws.get("selection"):
            selection = prior_ws.get("selection")
    if not cleaned:
        ignored.append("empty elements")
        if isinstance(prior, dict) and isinstance(prior.get("favlo_workspace"), dict):
            prior_elements = prior["favlo_workspace"].get("elements")
            if isinstance(prior_elements, list) and prior_elements:
                selection = selection_for_elements(prior_elements, selection)
                return {
                    "favlo_workspace": {
                        "version": 1,
                        "guide": guide_id,
                        "elements": prior_elements,
                        "selection": selection,
                    }
                }, ignored
        return initial_for(guide_id), ignored
    cleaned = _ensure_gallery_preview(guide_id, cleaned, ignored, prior)
    selection = selection_for_elements(cleaned, selection)
    return {
        "favlo_workspace": {
            "version": 1,
            "guide": guide_id,
            "elements": cleaned,
            "selection": selection,
        }
    }, ignored


def _ensure_gallery_preview(
    guide_id: str,
    elements: list[dict[str, Any]],
    ignored: list[str],
    prior: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """On first fill, restore preview when the agent omits it.

    If the strip already had items and the agent omits preview, keep that remove.
    """
    if guide_id != "gallery":
        return elements
    if not filmstrip_paths(elements):
        return elements
    if any(item.get("element") == "preview" for item in elements):
        return elements
    prior_paths: list[str] = []
    if isinstance(prior, dict):
        prior_ws = prior.get("favlo_workspace")
        if isinstance(prior_ws, dict):
            prior_paths = filmstrip_paths(list(prior_ws.get("elements") or []))
    if prior_paths:
        return elements
    ignored.append("restored preview")
    preview = {
        "key": "preview",
        "element": "preview",
        "slot": "stage",
        "props": {"fit": "contain"},
    }
    return [preview, *elements]


def filmstrip_paths(elements: list[dict[str, Any]]) -> list[str]:
    """Paths on filmstrip elements, in order."""
    paths: list[str] = []
    for element in elements:
        if element.get("element") != "filmstrip":
            continue
        for item in (element.get("props") or {}).get("items") or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if path:
                paths.append(path)
    return paths


def selection_for_elements(
    elements: list[dict[str, Any]],
    selection: dict[str, Any] | None,
) -> dict[str, str] | None:
    """Require a strip path when the filmstrip has items; otherwise clear selection."""
    paths = filmstrip_paths(elements)
    if not paths:
        return None
    if isinstance(selection, dict):
        path = str(selection.get("path") or "").strip()
        if path in paths:
            return {"kind": "file", "path": path}
    return {"kind": "file", "path": paths[0]}


def _clamp_props(name: str, props: dict[str, Any], ignored: list[str]) -> dict[str, Any]:
    spec = ELEMENTS[name]
    out: dict[str, Any] = {}
    if name == "preview":
        fit = props.get("fit", "contain")
        out["fit"] = fit if fit in {"contain", "cover"} else "contain"
        return out
    if name == "filmstrip":
        title = str(props.get("title") or "")[:80]
        items_raw = props.get("items") if isinstance(props.get("items"), list) else []
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in items_raw[:60]:
            if not isinstance(row, dict):
                ignored.append("dropped filmstrip item")
                continue
            path = str(row.get("path") or "").strip()
            if not path or path in seen:
                ignored.append("dropped filmstrip path")
                continue
            seen.add(path)
            item: dict[str, str] = {"path": path}
            caption = str(row.get("caption") or "").strip()
            if caption:
                item["caption"] = caption[:80]
            items.append(item)
        out["title"] = title
        out["items"] = items
        return out
    return out
