"""One Agent Framework turn against the local listener."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from favlo.chat_client import history_message, open_chat_client, reasoning_text


@dataclass(frozen=True)
class ReplyPiece:
    """One streamed piece. ``kind`` is ``reasoning`` or ``content``."""

    kind: str
    text: str


def assistant_piece(update: Any) -> str:
    role = getattr(update, "role", None)
    if role not in (None, "assistant"):
        return ""
    text = getattr(update, "text", "")
    return text if isinstance(text, str) else ""


_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def spoken_text(text: str) -> str:
    """Reply text with tool-call markup removed. An unfinished call stays hidden."""
    kept: list[str] = []
    rest = text
    while rest:
        start = rest.find(_TOOL_OPEN)
        if start < 0:
            keep = _marker_prefix(rest, _TOOL_OPEN)
            kept.append(rest[:-keep] if keep else rest)
            break
        kept.append(rest[:start])
        end = rest.find(_TOOL_CLOSE, start)
        if end < 0:
            break
        rest = rest[end + len(_TOOL_CLOSE) :]
    return "".join(kept).strip()


def _marker_prefix(text: str, marker: str) -> int:
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


def reasoning_piece(update: Any) -> str:
    """Reasoning text on one stream update."""
    contents = getattr(update, "contents", None) or []
    parts = [
        item.text
        for item in contents
        if getattr(item, "type", None) == "text_reasoning"
        and isinstance(getattr(item, "text", None), str)
        and item.text
    ]
    if parts:
        return "".join(parts)
    raw = getattr(update, "raw_representation", None)
    choices = getattr(raw, "choices", None) or []
    found: list[str] = []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        message = getattr(choice, "message", None)
        text = reasoning_text(delta if delta is not None else message)
        if text:
            found.append(text)
    return "".join(found)


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
    from web_api.session_tools import GROUP_ID as SESSION_ID
    from web_api.session_tools import INSTRUCTIONS as SESSION_INSTRUCTIONS
    from web_api.session_tools import tools as session_tools

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
    if SESSION_ID in tool_groups and store is not None and session_id and bus is not None:
        if SESSION_INSTRUCTIONS not in instructions:
            instructions = f"{instructions}\n\n{SESSION_INSTRUCTIONS}"
        tool_list.extend(session_tools(store, session_id, bus))
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
) -> AsyncIterator[ReplyPiece]:
    from agent_framework import Agent, InMemoryHistoryProvider
    from openai import AsyncOpenAI

    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=3600.0)
    client = open_chat_client(model, base_url, async_client=http)
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
    messages.extend(
        history_message(item["role"], item["text"], item.get("reasoning") or "")
        for item in history
    )
    stream = agent.run(user_text, session=session, stream=True)
    content_yielded = False
    saw_reasoning = False
    spoken = ""
    buffered: list[str] = []
    try:
        async for update in stream:
            reason = reasoning_piece(update)
            if reason:
                saw_reasoning = True
                yield ReplyPiece("reasoning", reason)
            piece = assistant_piece(update)
            if not piece:
                continue
            if tool_list:
                buffered.append(piece)
                visible = spoken_text("".join(buffered))
                extra = visible[len(spoken) :] if visible.startswith(spoken) else visible
                spoken = visible
                if not extra:
                    continue
                content_yielded = True
                yield ReplyPiece("content", extra)
                continue
            content_yielded = True
            yield ReplyPiece("content", piece)
        if not content_yielded:
            final = await stream.get_final_response()
            if not saw_reasoning:
                for message in getattr(final, "messages", None) or []:
                    reason = reasoning_piece(message)
                    if reason:
                        yield ReplyPiece("reasoning", reason)
            text = getattr(final, "text", None)
            if isinstance(text, str) and text.strip():
                if tool_list:
                    text = spoken_text(text)
                if text:
                    yield ReplyPiece("content", text)
    finally:
        closer = getattr(client, "close", None)
        if closer is not None:
            closed = closer()
            if hasattr(closed, "__await__"):
                await closed
        await http.close()
