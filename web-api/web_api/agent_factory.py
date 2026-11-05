"""Build an Agent Framework agent from a frozen chat definition."""

from __future__ import annotations

from typing import Any

from favlo.chat_client import open_chat_client
from favlo.config_tools import GROUP_ID
from favlo.config_tools import tools as configuration_tools
from web_api.compaction import percent_window
from web_api.conversation import GROUP_ID as CONVERSATION_ID
from web_api.conversation import tools as conversation_tools
from web_api.heartbeat_tools import GROUP_ID as HEARTBEAT_ID
from web_api.heartbeat_tools import tools as heartbeat_tools


def from_definition(
    *,
    definition: dict[str, Any],
    store: Any,
    session_id: str,
    bus: Any | None = None,
    mcp_tools: list[object] | None = None,
) -> Any:
    """Recreate the agent from the snapshot only. Instructions are verbatim."""
    from agent_framework import Agent, InMemoryHistoryProvider

    model = str((definition.get("model") or {}).get("model_id") or "")
    base_url = str((definition.get("model") or {}).get("base_url") or "")
    agent_meta = definition.get("agent") or {}
    agent_name = str(agent_meta.get("name") or "Agent")
    agent_id = agent_meta.get("id")
    instructions = str(definition.get("instructions") or "")
    options = dict(definition.get("options") or {})
    options["store"] = True
    tool_list = rebuild_tools(
        definition,
        store=store,
        session_id=session_id,
        bus=bus,
        base_url=base_url,
        agent_id=str(agent_id) if agent_id else None,
        mcp_tools=mcp_tools or [],
    )
    from web_api.media_delivery import media_middleware

    client = open_chat_client(model, base_url, middleware=media_middleware())
    if tool_list:
        client.function_invocation_configuration["max_iterations"] = int(
            definition.get("max_iterations") or 6
        )
    providers: list[Any] = [
        InMemoryHistoryProvider(skip_excluded=True, load_messages=False),
        percent_window(store.compaction_percent(session_id)),
    ]
    return Agent(
        client=client,
        name=agent_name,
        instructions=instructions,
        tools=tool_list,
        default_options=options,
        require_per_service_call_history_persistence=True,
        context_providers=providers,
    )


def rebuild_tools(
    definition: dict[str, Any],
    *,
    store: Any,
    session_id: str,
    bus: Any | None,
    base_url: str,
    agent_id: str | None,
    mcp_tools: list[object],
) -> list[object]:
    """Bind live callables that match the frozen tool section."""
    groups = list(definition.get("tool_groups") or [])
    tool_list: list[object] = []
    if GROUP_ID in groups:
        tool_list.extend(configuration_tools(base_url, None))
    if CONVERSATION_ID in groups:
        tool_list.extend(conversation_tools(store))
    if HEARTBEAT_ID in groups and agent_id:
        tool_list.extend(heartbeat_tools(store, agent_id))
    if mcp_tools:
        tool_list.extend(mcp_tools)
    workspace = definition.get("workspace")
    workspace_id = None
    guide_id = ""
    if isinstance(workspace, dict) and workspace.get("id"):
        workspace_id = str(workspace["id"])
        guide_id = str(workspace.get("guide") or workspace.get("type") or "")
    else:
        workspace_id = store.gallery_for_session(session_id)
        if workspace_id:
            guide_id = "gallery"
    if workspace_id and guide_id == "gallery":
        from web_api.gallery_tools import tools as gallery_tools

        tool_list.extend(gallery_tools(store, workspace_id, bus))
    elif session_id:
        from web_api.display_tools import tools as display_tools

        tool_list.extend(display_tools(store, session_id, bus))
    return tool_list
