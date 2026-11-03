"""Frozen agent definition for one chat GUID."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from web_api.chat import sampling_options, turn_setup
from web_api.events import now


def build(
    *,
    store: Any,
    agent: dict[str, Any],
    session_id: str,
    base_url: str,
    model: str,
    mcp_text: str = "",
    mcp_tools: list[object] | None = None,
    bus: Any | None = None,
    pending: set[Any] | None = None,
) -> dict[str, Any]:
    """Snapshot instructions, model, and tool identity from live configuration."""
    instructions, tool_list = turn_setup(
        str(agent.get("persona") or ""),
        str(agent.get("name") or "Agent"),
        list(agent.get("tool_groups") or []),
        base_url,
        store,
        agent.get("id"),
        session_id,
        bus,
        pending,
        mcp_text=mcp_text,
        mcp_tools=mcp_tools,
    )
    options = sampling_options(
        agent.get("enable_thinking"),
        agent.get("preserve_thinking"),
        agent.get("reasoning_effort"),
    ) or {}
    options = dict(options)
    options["store"] = False
    tools = [_tool_identity(item) for item in tool_list]
    mcp = _mcp_rows(store, str(agent["id"])) if agent.get("id") else []
    workspace = store.workspace_for_session(session_id)
    workspace_payload: dict[str, Any] | None = None
    if workspace is not None:
        workspace_payload = {
            "id": workspace["id"],
            "type": workspace["type"],
            "guide": workspace["type"],
            "guide_version": 1,
        }
    payload: dict[str, Any] = {
        "version": 1,
        "created_at": now(),
        "agent": {"id": agent.get("id"), "name": agent.get("name")},
        "model": {
            "provider_id": agent.get("default_model_id"),
            "model_id": model,
            "base_url": base_url,
        },
        "options": options,
        "max_iterations": 6 if tools else None,
        "instructions": instructions,
        "tools": tools,
        "mcp": mcp,
        "workspace": workspace_payload,
        "tool_groups": list(agent.get("tool_groups") or []),
    }
    payload["hash"] = hash_of(payload)
    return payload


def hash_of(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _tool_identity(item: object) -> dict[str, Any]:
    name = getattr(item, "name", None) or getattr(item, "__name__", None) or ""
    if not isinstance(name, str):
        name = str(name)
    description = getattr(item, "description", None) or getattr(item, "__doc__", None) or ""
    if not isinstance(description, str):
        description = str(description)
    prefix = getattr(item, "tool_name_prefix", None)
    source = "mcp" if prefix else "function"
    if name.startswith("gallery_"):
        source = "gallery"
    elif name.startswith("conversation_"):
        source = "conversation"
    elif name.startswith("heartbeat_"):
        source = "heartbeat"
    elif name.startswith("config_"):
        source = "config"
    return {
        "name": name,
        "source": source,
        "description": description.strip(),
        "prefix": prefix if isinstance(prefix, str) else None,
    }


def _mcp_rows(store: Any, agent_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for server in store.enabled_mcp_servers(agent_id):
        switches = server.get("switches") or {}
        groups = [str(key) for key, on in switches.items() if on]
        rows.append(
            {
                "server_id": server["server_id"],
                "name": server["server_name"],
                "transport": server["transport"],
                "url": server.get("url") or "",
                "command": server.get("command") or "",
                "groups": groups,
            }
        )
    return rows
