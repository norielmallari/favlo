"""One Agent Framework turn against the local listener."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


def assistant_piece(update: Any) -> str:
    role = getattr(update, "role", None)
    if role not in (None, "assistant"):
        return ""
    text = getattr(update, "text", "")
    return text if isinstance(text, str) else ""


def turn_setup(
    persona: str,
    agent_name: str,
    tool_groups: list[str],
    base_url: str,
    store: Any | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    bus: Any | None = None,
    pending: set[Any] | None = None,
) -> tuple[str, list[object]]:
    """Instructions and tools for one turn. Each enabled group adds its own text."""
    from favlo.config_tools import GROUP_ID
    from favlo.config_tools import INSTRUCTIONS as CONFIG_INSTRUCTIONS
    from favlo.config_tools import tools as configuration_tools
    from web_api.conversation import GROUP_ID as CONVERSATION_ID
    from web_api.conversation import INSTRUCTIONS as CONVERSATION_INSTRUCTIONS
    from web_api.conversation import tools as conversation_tools
    from web_api.heartbeat_tools import GROUP_ID as HEARTBEAT_ID
    from web_api.heartbeat_tools import INSTRUCTIONS as HEARTBEAT_INSTRUCTIONS
    from web_api.heartbeat_tools import tools as heartbeat_tools

    instructions = persona.strip() or f"You are {agent_name}."
    tool_list: list[object] = []
    if GROUP_ID in tool_groups:
        if CONFIG_INSTRUCTIONS not in instructions:
            instructions = f"{instructions}\n\n{CONFIG_INSTRUCTIONS}"
        tool_list.extend(configuration_tools(base_url, None))
    if CONVERSATION_ID in tool_groups and store is not None:
        if CONVERSATION_INSTRUCTIONS not in instructions:
            instructions = f"{instructions}\n\n{CONVERSATION_INSTRUCTIONS}"
        tool_list.extend(conversation_tools(store))
    if HEARTBEAT_ID in tool_groups and store is not None and agent_id:
        if HEARTBEAT_INSTRUCTIONS not in instructions:
            instructions = f"{instructions}\n\n{HEARTBEAT_INSTRUCTIONS}"
        tool_list.extend(heartbeat_tools(store, agent_id))
    if store is not None and agent_id and session_id and bus is not None and pending is not None:
        from web_api.mcp_tools import INSTRUCTIONS as MCP_INSTRUCTIONS
        from web_api.mcp_tools import bundle

        mcp_text, mcp_tools = bundle(store, agent_id, session_id, bus, pending)
        if mcp_tools:
            if MCP_INSTRUCTIONS not in instructions:
                instructions = f"{instructions}\n\n{mcp_text}"
            tool_list.extend(mcp_tools)
    return instructions, tool_list


async def stream_reply(
    *,
    base_url: str,
    model: str,
    persona: str,
    agent_name: str,
    history: list[dict[str, str]],
    user_text: str,
    tool_groups: list[str] | None = None,
    store: Any | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    bus: Any | None = None,
    pending: set[Any] | None = None,
) -> AsyncIterator[str]:
    from agent_framework import Agent, InMemoryHistoryProvider, Message
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=3600.0)
    client = OpenAIChatCompletionClient(
        model=model,
        api_key="favlo",
        base_url=base_url,
        async_client=http,
    )
    instructions, tool_list = turn_setup(
        persona,
        agent_name,
        tool_groups or [],
        base_url,
        store,
        agent_id,
        session_id,
        bus,
        pending,
    )
    if tool_list:
        client.function_invocation_configuration["max_iterations"] = 6
    agent = Agent(
        client=client,
        name=agent_name,
        instructions=instructions,
        tools=tool_list,
        default_options={"max_tokens": 1024, "extra_body": {"enable_thinking": False}},
    )
    session = agent.create_session()
    bucket = session.state.setdefault(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
    messages = bucket.setdefault("messages", [])
    messages.extend(Message(item["role"], [item["text"]]) for item in history)
    stream = agent.run(user_text, session=session, stream=True)
    yielded = False
    spoken = ""
    buffered: list[str] = []
    try:
        async for update in stream:
            piece = assistant_piece(update)
            if not piece:
                continue
            if tool_list:
                from favlo.talk import spoken_text

                buffered.append(piece)
                visible = spoken_text("".join(buffered))
                extra = visible[len(spoken) :] if visible.startswith(spoken) else visible
                spoken = visible
                if not extra:
                    continue
                yielded = True
                yield extra
                continue
            yielded = True
            yield piece
        if not yielded:
            final = await stream.get_final_response()
            text = getattr(final, "text", None)
            if isinstance(text, str) and text.strip():
                if tool_list:
                    from favlo.talk import spoken_text

                    text = spoken_text(text)
                if text:
                    yield text
    finally:
        closer = getattr(client, "close", None)
        if closer is not None:
            closed = closer()
            if hasattr(closed, "__await__"):
                await closed
        await http.close()
