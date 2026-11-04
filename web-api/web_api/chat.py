"""One Agent Framework turn against the local listener."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from favlo.chat_client import close_chat_client, open_chat_client, prompt_window, reasoning_text


@dataclass(frozen=True)
class ReplyPiece:
    """One streamed piece.

    ``kind`` is ``reasoning``, ``content``, ``tool``, or ``tool_result``.
    A tool piece carries the call id, the function name, and one argument delta.
    """

    kind: str
    text: str = ""
    call_id: str = ""
    name: str = ""
    index: int | None = None


def sampling_options(
    enable_thinking: bool | None,
    preserve_thinking: bool | None,
    reasoning_effort: str | None,
) -> dict[str, Any] | None:
    """Responses options the agent set. An unset field stays off the request.

    Agent config stores ``reasoning_effort``. That value becomes
    ``reasoning.effort``. A legacy ``enable_thinking`` value maps only when
    effort is unset (``none`` or ``medium``). ``preserve_thinking`` stays a
    body field.
    """
    options: dict[str, Any] = {}
    effort: str | None = None
    if reasoning_effort:
        effort = reasoning_effort
    elif enable_thinking is not None:
        effort = "medium" if enable_thinking else "none"
    if effort is not None:
        options["reasoning"] = {"effort": effort}
    if preserve_thinking is not None:
        options["extra_body"] = {"preserve_thinking": preserve_thinking}
    if not options:
        return None
    return options


def assistant_piece(update: Any) -> str:
    role = getattr(update, "role", None)
    if role not in (None, "assistant"):
        return ""
    text = getattr(update, "text", "")
    return text if isinstance(text, str) else ""


def _grow(current: str, incoming: str) -> str:
    if not incoming or incoming == current:
        return current
    if not current or incoming.startswith(current):
        return incoming
    if current.endswith(incoming):
        return current
    return current + incoming


def tool_pieces(update: Any) -> list[ReplyPiece]:
    """Function calls and results on one stream update. Markup is left unread."""
    pieces: list[ReplyPiece] = []
    for item in getattr(update, "contents", None) or []:
        kind = getattr(item, "type", None)
        if kind not in ("function_call", "function_result"):
            continue
        props = getattr(item, "additional_properties", None) or {}
        raw_index = props.get("tool_call_index") if isinstance(props, dict) else None
        index = raw_index if isinstance(raw_index, int) else None
        call_id = getattr(item, "call_id", None) or ""
        if not isinstance(call_id, str):
            call_id = str(call_id)
        name = getattr(item, "name", None) or ""
        if not isinstance(name, str):
            name = ""
        if kind == "function_result":
            result = getattr(item, "result", None)
            text = result if isinstance(result, str) else ""
            pieces.append(ReplyPiece("tool_result", text, call_id, name, index))
            continue
        arguments = getattr(item, "arguments", None) or ""
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = str(arguments)
        pieces.append(ReplyPiece("tool", arguments, call_id, name, index))
    return pieces


def record_tool(calls: list[dict[str, Any]], piece: ReplyPiece) -> dict[str, Any]:
    """Merge one streamed function call or result into ``calls``."""
    found: dict[str, Any] | None = None
    if piece.call_id:
        found = next((row for row in calls if row["id"] == piece.call_id), None)
    if found is None and piece.index is not None:
        found = next((row for row in calls if row.get("index") == piece.index), None)
    if found is None:
        key = piece.call_id or (
            f"index:{piece.index}" if piece.index is not None else f"call-{len(calls)}"
        )
        found = {"id": key, "name": "", "arguments": "", "result": "", "index": piece.index}
        calls.append(found)
    if piece.call_id:
        found["id"] = piece.call_id
    if piece.index is not None:
        found["index"] = piece.index
    if piece.kind == "tool_result":
        if piece.text:
            found["result"] += piece.text
        if piece.name:
            found["name"] = _grow(str(found["name"]), piece.name)
        return found
    if piece.name:
        found["name"] = _grow(str(found["name"]), piece.name)
    if piece.text:
        found["arguments"] = str(found["arguments"]) + piece.text
    return found


def public_call(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row["id"]),
        "name": str(row["name"]),
        "arguments": str(row["arguments"]),
        "result": str(row["result"]),
    }


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
    mcp_text: str = "",
    mcp_tools: list[object] | None = None,
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
    if mcp_tools:
        from web_api.mcp_tools import INSTRUCTIONS as MCP_INSTRUCTIONS

        if MCP_INSTRUCTIONS not in instructions:
            instructions = f"{instructions}\n\n{mcp_text}"
        tool_list.extend(mcp_tools)
    if store is not None and session_id:
        from web_api.gallery_tools import INSTRUCTIONS as GALLERY_INSTRUCTIONS
        from web_api.gallery_tools import tools as gallery_tools
        from web_api.workspace import skills_text
        from web_api.workspace import text_for as workspace_text

        workspace = store.workspace_for_session(session_id)
        if workspace is not None:
            extra = workspace_text(workspace["type"])
            if extra and extra not in instructions:
                instructions = f"{instructions}\n\n{extra}"
        skill_rows: list[dict[str, Any]] = []
        if agent_id:
            skill_rows.extend(store.list_agent_skills(agent_id))
        if workspace is not None:
            skill_rows.extend(store.list_skills(workspace["id"]))
        skill_block = skills_text(skill_rows)
        if skill_block and skill_block not in instructions:
            instructions = f"{instructions}\n\n{skill_block}"
        workspace_id = store.gallery_for_session(session_id)
        if workspace_id:
            if GALLERY_INSTRUCTIONS not in instructions:
                instructions = f"{instructions}\n\n{GALLERY_INSTRUCTIONS}"
            tool_list.extend(gallery_tools(store, workspace_id, bus))
            from web_api.workspace_tools import tools as workspace_place_tools

            tool_list.extend(workspace_place_tools(store, workspace_id, guide_id="gallery"))
    return instructions, tool_list


def unsent_images(images: list[tuple[bytes, str]] | None, session: Any) -> list[tuple[bytes, str]]:
    """Drop images whose bytes are already on an earlier turn of this session."""
    if not images:
        return []
    seen = _attached_hashes(session)
    fresh: list[tuple[bytes, str]] = []
    for data, media_type in images:
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        fresh.append((data, media_type))
    return fresh


def _attached_hashes(session: Any) -> set[str]:
    found: set[str] = set()
    for message in _session_messages(session):
        for content in _message_contents(message):
            payload = _image_bytes(content)
            if payload is not None:
                found.add(hashlib.sha256(payload).hexdigest())
    return found


def _session_messages(session: Any) -> list[Any]:
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return []
    memory = state.get("in_memory")
    if isinstance(memory, dict) and isinstance(memory.get("messages"), list):
        return list(memory["messages"])
    messages = state.get("messages")
    return list(messages) if isinstance(messages, list) else []


def _message_contents(message: Any) -> list[Any]:
    if isinstance(message, dict):
        contents = message.get("contents")
    else:
        contents = getattr(message, "contents", None)
    return list(contents) if isinstance(contents, list) else []


def _image_bytes(content: Any) -> bytes | None:
    if isinstance(content, dict):
        media = content.get("media_type")
        uri = content.get("uri")
    else:
        media = getattr(content, "media_type", None)
        uri = getattr(content, "uri", None)
    if not isinstance(uri, str) or not uri.startswith("data:image/") or "," not in uri:
        return None
    known = media in {"image/png", "image/jpeg"}
    prefixed = uri.startswith(("data:image/png;", "data:image/jpeg;"))
    if not known and not prefixed:
        return None
    try:
        return base64.b64decode(uri.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def user_turn(text: str, images: list[tuple[bytes, str]] | None = None) -> Any:
    """The new user message. Named image bytes stay on this message when present."""
    if not images:
        return text
    from agent_framework import Content, Message

    parts: list[Any] = [text]
    for data, media_type in images:
        parts.append(Content.from_data(data=data, media_type=media_type))
    return Message("user", parts)


def _open_session(agent: Any, store: Any | None, session_id: str | None) -> Any:
    """Load the stored Agent Framework session, or open one for this chat."""
    from agent_framework import AgentSession

    if store is not None and session_id:
        raw = store.maf_session(session_id)
        if isinstance(raw, dict):
            try:
                return AgentSession.from_dict(raw)
            except Exception:
                return agent.create_session(session_id=session_id)
        return agent.create_session(session_id=session_id)
    return agent.create_session()


def _mcp_for_definition(
    definition: dict[str, Any], opened: list[object]
) -> list[object]:
    """Keep MCP tools whose server id was frozen with the chat."""
    rows = definition.get("mcp") or []
    if not isinstance(rows, list) or not rows:
        return []
    allowed = {
        str(row.get("server_id") or "")
        for row in rows
        if isinstance(row, dict) and row.get("server_id")
    }
    if not allowed:
        return []
    kept: list[object] = []
    for tool in opened:
        turn = getattr(tool, "turn", None)
        server_id = getattr(turn, "server_id", None) if turn is not None else None
        if server_id and str(server_id) in allowed:
            kept.append(tool)
    return kept


async def stream_reply(
    *,
    base_url: str,
    model: str,
    persona: str,
    agent_name: str,
    user_text: str,
    tool_groups: list[str] | None = None,
    store: Any | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    bus: Any | None = None,
    pending: set[Any] | None = None,
    cancel: asyncio.Event | None = None,
    user_images: list[tuple[bytes, str]] | None = None,
    enable_thinking: bool | None = None,
    preserve_thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> AsyncIterator[ReplyPiece]:
    from agent_framework import Agent, InMemoryHistoryProvider

    mcp_text = ""
    mcp_tools: list[object] = []
    if store is not None and agent_id and session_id and bus is not None and pending is not None:
        from web_api.mcp_tools import bundle

        mcp_text, mcp_tools = await bundle(store, agent_id, session_id, bus, pending)
    frozen = store.definition(session_id) if store is not None and session_id else None
    if frozen is None:
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
            mcp_text=mcp_text,
            mcp_tools=mcp_tools,
        )
        options = sampling_options(enable_thinking, preserve_thinking, reasoning_effort) or {}
        options = {**options, "store": True}
        if store is not None and session_id and agent_id:
            from web_api.definition import build as build_definition
            from web_api.definition import hash_of

            agent_row = {
                "id": agent_id,
                "name": agent_name,
                "persona": persona,
                "tool_groups": tool_groups or [],
                "enable_thinking": enable_thinking,
                "preserve_thinking": preserve_thinking,
                "reasoning_effort": reasoning_effort,
                "default_model_id": None,
            }
            payload = build_definition(
                store=store,
                agent=agent_row,
                session_id=session_id,
                base_url=base_url,
                model=model,
                mcp_text=mcp_text,
                mcp_tools=mcp_tools,
                bus=bus,
                pending=pending,
            )
            payload["instructions"] = instructions
            payload["options"] = options
            payload["hash"] = hash_of(payload)
            store.freeze_definition(session_id, payload)
            frozen = payload
        client = open_chat_client(model, base_url)
        if tool_list:
            client.function_invocation_configuration["max_iterations"] = int(
                (frozen or {}).get("max_iterations") or 6
            )
        providers: list[Any] = [
            InMemoryHistoryProvider(skip_excluded=True, load_messages=False)
        ]
        if store is not None and session_id:
            from web_api.compaction import percent_window

            providers.append(percent_window(store.compaction_percent(session_id)))
        else:
            providers.append(prompt_window())
        agent = Agent(
            client=client,
            name=agent_name,
            instructions=instructions,
            tools=tool_list,
            default_options=options,
            require_per_service_call_history_persistence=True,
            context_providers=providers,
        )
    else:
        from web_api.agent_factory import from_definition

        if store is None or not session_id:
            raise RuntimeError("A frozen definition needs a store session.")
        mcp_tools = _mcp_for_definition(frozen, mcp_tools)
        agent = from_definition(
            definition=frozen,
            store=store,
            session_id=session_id,
            bus=bus,
            mcp_tools=mcp_tools,
        )
        client = agent.client
    session = _open_session(agent, store, session_id)
    prompt = user_turn(user_text, unsent_images(user_images, session))
    clients = contextlib.AsyncExitStack()
    await clients.__aenter__()
    try:
        from web_api.mcp_tools import open_clients, restore_loaded

        await open_clients(mcp_tools, clients)
        if store is not None and session_id:
            await restore_loaded(mcp_tools, store, session_id)
        stream = agent.run(prompt, session=session, stream=True)
    except Exception:
        await clients.aclose()
        raise
    content_yielded = False
    saw_tool = False
    abort: asyncio.Task[None] | None = None
    if cancel is not None:

        async def _abort() -> None:
            await cancel.wait()
            closer = getattr(stream, "close", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()
            with contextlib.suppress(Exception):
                await close_chat_client(client)

        abort = asyncio.create_task(_abort())
    stopped = False
    keep = False
    try:
        try:
            async for update in stream:
                if cancel is not None and cancel.is_set():
                    stopped = True
                    break
                reason = reasoning_piece(update)
                if reason:
                    yield ReplyPiece("reasoning", reason)
                for call in tool_pieces(update):
                    saw_tool = True
                    yield call
                piece = assistant_piece(update)
                if not piece:
                    continue
                content_yielded = True
                yield ReplyPiece("content", piece)
        except Exception:
            if cancel is not None and cancel.is_set():
                stopped = True
            else:
                raise
        if cancel is not None and cancel.is_set():
            stopped = True
        if stopped:
            return
        if not content_yielded:
            final = await stream.get_final_response()
            messages = getattr(final, "messages", None) or []
            if not saw_tool:
                for message in messages:
                    for call in tool_pieces(message):
                        yield call
            text = getattr(final, "text", None)
            if isinstance(text, str) and text.strip():
                yield ReplyPiece("content", text)
        keep = True
    finally:
        if abort is not None:
            abort.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await abort
        if keep and store is not None and session_id and hasattr(session, "to_dict"):
            store.save_maf_session(session_id, session.to_dict())
        if keep and store is not None and session_id and mcp_tools:
            with contextlib.suppress(Exception):
                from web_api.mcp_tools import persist_loaded

                persist_loaded(mcp_tools, store, session_id)
        with contextlib.suppress(Exception):
            closer = getattr(agent, "close", None)
            if closer is not None:
                closed = closer()
                if hasattr(closed, "__await__"):
                    await closed
        with contextlib.suppress(Exception):
            await clients.aclose()
        with contextlib.suppress(Exception):
            await close_chat_client(client)
