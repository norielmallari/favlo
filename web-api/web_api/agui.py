"""Favlo chat on the Agent Framework AG-UI endpoint.

The HTTP route is ``add_agent_framework_fastapi_endpoint``. This module resolves
the per-thread agent and then calls ``run_agent_stream``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ag_ui.core import EventType, RunErrorEvent, RunFinishedEvent
from agent_framework_ag_ui import AgentFrameworkAgent, add_agent_framework_fastapi_endpoint
from agent_framework_ag_ui._agent_run import run_agent_stream
from fastapi import FastAPI

from web_api.chat import stream_reply
from web_api.ids import guid
from web_api.reply_error import spoken_error
from web_api.session_commands import ARCHIVE_CUE
from web_api.snapshots import FavloSnapshotStore
from web_api.store import Conflict, NotFound, Rejected

_SNAPSHOT_SCOPE = "favlo"


class _IdleAgent:
    """Stand-in for hydrate requests. A real turn replaces it before the model runs."""

    name = "Favlo"
    description = "Favlo"
    default_options: dict[str, Any] = {"store": True}

    def run(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        async def _empty() -> AsyncIterator[Any]:
            if False:
                yield None

        return _empty()


class _ScriptedAgent:
    """Test double. Production turns build a real agent."""

    def __init__(self, reply: Any, cancel: asyncio.Event, tool_groups: list[str]) -> None:
        self._reply = reply
        self._cancel = cancel
        self._tool_groups = tool_groups
        self.name = "Favlo"
        self.default_options = {"store": True}

    def run(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        reply = self._reply
        cancel = self._cancel
        tool_groups = self._tool_groups

        async def _updates() -> AsyncIterator[Any]:
            from agent_framework import AgentResponseUpdate, Content

            from web_api.chat import record_tool

            calls: list[dict[str, Any]] = []
            async for piece in reply(cancel=cancel, tool_groups=tool_groups):
                if isinstance(piece, str):
                    if piece:
                        yield AgentResponseUpdate(
                            contents=[Content.from_text(text=piece)],
                            role="assistant",
                        )
                    continue
                kind = getattr(piece, "kind", None)
                text = getattr(piece, "text", None)
                if kind in {"tool", "tool_result"}:
                    row = record_tool(calls, piece)
                    if kind == "tool_result":
                        yield AgentResponseUpdate(
                            contents=[
                                Content.from_function_call(
                                    call_id=str(row["id"]),
                                    name=str(row["name"]),
                                    arguments=str(row["arguments"]),
                                )
                            ],
                            role="assistant",
                        )
                        yield AgentResponseUpdate(
                            contents=[
                                Content.from_function_result(
                                    call_id=str(row["id"]),
                                    result=row["result"],
                                )
                            ],
                            role="assistant",
                        )
                    continue
                if not text:
                    continue
                if kind == "reasoning":
                    yield AgentResponseUpdate(
                        contents=[Content.from_text_reasoning(text=text)],
                        role="assistant",
                    )
                elif kind == "content":
                    yield AgentResponseUpdate(
                        contents=[Content.from_text(text=text)],
                        role="assistant",
                    )

        return _updates()


class FavloAguiHost(AgentFrameworkAgent):
    """One host for every chat. ``run`` picks the agent for ``threadId``."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(
            _IdleAgent(),
            name="Favlo",
            description="Favlo chat",
            require_confirmation=False,
            snapshot_store=FavloSnapshotStore(app),
            emit_messages_snapshot=True,
            use_service_session=True,
        )
        self._app = app

    async def run(
        self,
        input_data: dict[str, Any],
        *,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        store = self._app.state.store
        bus = self._app.state.bus
        thread_id = str(input_data.get("thread_id") or "").strip()
        run_id = str(input_data.get("run_id") or "").strip() or guid()
        if not thread_id:
            yield RunErrorEvent(message="threadId is required.", code="rejected")
            return
        props = input_data.get("forwarded_props") or {}
        if not isinstance(props, dict):
            props = {}
        archive = bool(props.get("favloArchive") or props.get("favlo_archive"))
        adopt = bool(props.get("favloAdopt"))
        silent = bool(props.get("favloSilent"))
        messages = input_data.get("messages") or []
        if not isinstance(messages, list):
            yield RunErrorEvent(message="messages must be a list.", code="rejected")
            return
        prepared = dict(input_data)
        prepared["run_id"] = run_id
        prepared.setdefault("__ag_ui_snapshot_scope", _SNAPSHOT_SCOPE)
        text = "" if archive else _user_text(messages)
        if archive:
            prepared["messages"] = [{"id": guid(), "role": "user", "content": ARCHIVE_CUE}]
            text = ARCHIVE_CUE
        elif not text:
            async for event in run_agent_stream(
                prepared,
                self.agent,
                self.config,
                approval_state_store=self._approval_state_store,
                function_invocation_kwargs=function_invocation_kwargs,
            ):
                yield event
            return

        try:
            snapshot = await asyncio.to_thread(store.session_snapshot, thread_id)
            session = snapshot["session"]
            agent_id = session.get("participant_id")
            if not agent_id:
                raise Rejected("That chat has no agent.")
            agent_row = next(
                (row for row in store.agents() if row["id"] == agent_id),
                None,
            )
            if agent_row is None:
                raise NotFound("That agent does not exist.")
            if not adopt:
                if archive:
                    running = next(
                        (
                            row
                            for row in snapshot["runs"]
                            if row["id"] == run_id and row["status"] == "running"
                        ),
                        None,
                    )
                    if running is None:
                        started = store.begin_agui_run(thread_id, run_id, "", archive=True)
                        for event in started.get("events") or []:
                            await bus.publish(event)
                else:
                    started = store.begin_agui_run(thread_id, run_id, text, archive=False)
                    for event in started.get("events") or []:
                        await bus.publish(event)
            _prefer_stored_state(store, thread_id, prepared)
            if not archive:
                if silent:
                    # Server cue (greeting, follow-up): fenced user host cue + empty pad.
                    from web_api.host_cues import host_cue_message

                    store.append_thread_snapshot_message(
                        _SNAPSHOT_SCOPE,
                        thread_id,
                        host_cue_message(text, message_id=guid()),
                    )
                    prepared["messages"] = [{"id": guid(), "role": "user", "content": ""}]
                else:
                    prior_messages = list(snapshot.get("messages") or [])
                    prepared["messages"] = _with_selection_cue(
                        store,
                        thread_id,
                        list(prepared.get("messages") or []),
                    )
                    names = text
                    if adopt:
                        names = _adopted_named_text(store, run_id, text)
                    prepared["messages"] = _with_images(
                        list(prepared.get("messages") or []),
                        _images(store, thread_id, names, prior_messages),
                    )
            before_ids = _assistant_ids(snapshot["messages"])
        except (Conflict, NotFound, Rejected) as exc:
            if adopt:
                failed = await asyncio.to_thread(
                    store.settle_agui_run,
                    run_id,
                    "failed",
                    spoken_error(str(exc)),
                )
                for event in failed:
                    await bus.publish(event)
            yield RunErrorEvent(message=spoken_error(str(exc)), code=type(exc).__name__.lower())
            return

        cancel = self._app.state.cancels.get(run_id)
        if not isinstance(cancel, asyncio.Event):
            cancel = asyncio.Event()
            self._app.state.cancels[run_id] = cancel
        closer: contextlib.AbstractAsyncContextManager[Any] | None = None
        agent: Any = self.agent
        settled = False
        try:
            agent, closer = await _open_agent(
                app=self._app,
                store=store,
                thread_id=thread_id,
                agent_row=agent_row,
                archive=archive,
                cancel=cancel,
            )
            async for event in run_agent_stream(
                prepared,
                agent,
                self.config,
                approval_state_store=self._approval_state_store,
                function_invocation_kwargs=function_invocation_kwargs,
            ):
                if cancel.is_set():
                    events = await asyncio.to_thread(store.settle_agui_run, run_id, "cancelled", None)
                    for item in events:
                        await bus.publish(item)
                    settled = True
                    yield RunFinishedEvent(
                        thread_id=thread_id,
                        run_id=run_id,
                        result={"status": "cancelled"},
                    )
                    return
                kind = getattr(event, "type", None)
                if kind == EventType.RUN_FINISHED or kind == "RUN_FINISHED":
                    if archive:
                        events = await asyncio.to_thread(store.finish_archive, run_id) or []
                        for item in events:
                            await bus.publish(item)
                        new_id = _cleared_session(events) or thread_id
                        settled = True
                        yield RunFinishedEvent(
                            thread_id=thread_id,
                            run_id=run_id,
                            result={"status": "archived", "session_id": new_id},
                        )
                        continue
                    view = await asyncio.to_thread(store.session_snapshot, thread_id)
                    if _assistant_arrived(before_ids, view["messages"]):
                        events = await asyncio.to_thread(
                            store.settle_agui_run, run_id, "completed", None
                        )
                        _persist_mcp(agent, store, thread_id)
                    else:
                        events = await asyncio.to_thread(
                            store.settle_agui_run,
                            run_id,
                            "failed",
                            "The model returned an empty reply.",
                        )
                    for item in events:
                        await bus.publish(item)
                    settled = True
                yield event
        except Exception as exc:
            if not settled:
                events = await asyncio.to_thread(
                    store.settle_agui_run,
                    run_id,
                    "failed",
                    spoken_error(str(exc)),
                )
                for item in events:
                    await bus.publish(item)
                settled = True
            yield RunErrorEvent(message=spoken_error(str(exc)), code="failed")
        finally:
            self._app.state.cancels.pop(run_id, None)
            await _close_mcp(agent)
            if closer is not None:
                await closer.aclose()
            await _close_client(agent)
            if not settled and not archive:
                events = await asyncio.to_thread(
                    store.settle_agui_run,
                    run_id,
                    "failed",
                    "The reply stopped.",
                )
                for item in events:
                    await bus.publish(item)


def install(app: FastAPI) -> None:
    """Register the library route at ``/ag-ui``."""
    host = FavloAguiHost(app)
    app.state.agui_host = host
    add_agent_framework_fastapi_endpoint(
        app,
        host,
        "/ag-ui",
        snapshot_scope_resolver=lambda _request: "favlo",
        keepalive_seconds=15,
        detached_runs=False,
    )


def _user_text(messages: list[Any]) -> str:
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {None, "text", "input_text"}
            ]
            text = "".join(parts).strip()
            if text:
                return text
    return ""


def _prefer_stored_state(store: Any, thread_id: str, prepared: dict[str, Any]) -> None:
    stored = store.agui_state(thread_id) or {}
    incoming = prepared.get("state")
    if not isinstance(incoming, dict):
        incoming = {}
    if isinstance(stored, dict) and isinstance(stored.get("favlo_workspace"), dict):
        if not isinstance(incoming.get("favlo_workspace"), dict):
            prepared["state"] = {**incoming, **stored}


def _images(
    store: Any,
    thread_id: str,
    text: str,
    prior_messages: list[Any],
) -> list[tuple[bytes, str]]:
    """Named png/jpeg bytes, skipping paths already named on the previous user turn."""
    from web_api.folder_port import folder_for
    from web_api.gallery_tools import named_images

    prior = _previous_user_text(prior_messages)
    already = {name.lower() for name in named_images(prior)}
    fresh_names = [name for name in named_images(text) if name.lower() not in already]
    if not fresh_names:
        return []
    workspace_id = store.gallery_for_session(thread_id)
    if not workspace_id:
        return []
    try:
        folder = folder_for(store, workspace_id)
        relative_paths = folder.find_images(fresh_names)
    except Rejected:
        return []
    out: list[tuple[bytes, str]] = []
    for relative in relative_paths:
        try:
            payload = folder.read(relative)
            media = str(folder.stat(relative)["media_type"])
        except Rejected:
            continue
        if media not in {"image/png", "image/jpeg"}:
            continue
        out.append((payload, media))
    return out


def _with_images(
    messages: list[Any],
    images: list[tuple[bytes, str]],
) -> list[Any]:
    if not images:
        return messages
    copied = [dict(item) if isinstance(item, dict) else item for item in messages]
    for index in range(len(copied) - 1, -1, -1):
        item = copied[index]
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        text = content if isinstance(content, str) else _user_text([item])
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for data, media_type in images:
            encoded = base64.b64encode(data).decode("ascii")
            parts.append(
                {
                    "type": "image",
                    "mimeType": media_type,
                    "source": {"type": "base64", "value": encoded, "mimeType": media_type},
                }
            )
        item["content"] = parts
        break
    return copied


def _with_selection_cue(
    store: Any,
    thread_id: str,
    messages: list[Any],
) -> list[Any]:
    """Persist a selection-change cue as a fenced, UI-hidden user message.

    The typed user message stays clean. The cue is a durable host row in the
    thread snapshot (``<system_reminder>`` fence + AG-UI
    ``metadata.favlo.ui=hidden``). No-op when selection is unchanged.
    """
    from web_api.gallery_tools import (
        announced_selection_from_history,
        selection_announcement,
    )
    from web_api.gallery_metadata import read_sidecar
    from web_api.folder_port import folder_for
    from web_api.host_cues import host_cue_message

    workspace_id = store.gallery_for_session(thread_id)
    if not workspace_id:
        return messages
    placement = store.agui_state(thread_id) or {}
    workspace = placement.get("favlo_workspace") if isinstance(placement, dict) else None
    selected = None
    if isinstance(workspace, dict):
        selection = workspace.get("selection")
        if isinstance(selection, dict):
            selected = str(selection.get("path") or "").strip() or None
    document = store.thread_snapshot(_SNAPSHOT_SCOPE, thread_id) or {}
    history = list(document.get("messages") or []) if isinstance(document, dict) else []
    previous = announced_selection_from_history(history)
    description = ""
    if selected:
        try:
            folder = folder_for(store, workspace_id)
            side = read_sidecar(folder.root, selected) or {}
            description = str(side.get("description") or "").strip()
        except Rejected:
            description = ""
    cue = selection_announcement(selected, previous, description)
    if not cue:
        return messages
    store.append_thread_snapshot_message(
        _SNAPSHOT_SCOPE,
        thread_id,
        host_cue_message(cue, message_id=guid()),
    )
    return messages


def _previous_user_text(messages: list[Any]) -> str:
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {None, "text", "input_text"}
            ]
            return "".join(parts)
        body = item.get("body")
        if isinstance(body, str):
            return body
    return ""


def _assistant_ids(messages: list[Any]) -> set[str]:
    found: set[str] = set()
    for item in messages:
        if isinstance(item, dict) and item.get("role") == "assistant" and item.get("id"):
            found.add(str(item["id"]))
    return found


def _assistant_arrived(before: set[str], messages: list[Any]) -> bool:
    for item in messages:
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        if str(item.get("id") or "") in before:
            continue
        if str(item.get("body") or "").strip() or str(item.get("reasoning") or "").strip():
            return True
    return False


def _adopted_named_text(store: Any, run_id: str, text: str) -> str:
    """Append an add-to-chat attachment path only. Selection uses the text cue."""
    try:
        context = store.run_context(run_id)
    except (NotFound, Rejected):
        return text
    attached = context.get("gallery_attachment")
    if not attached:
        return text
    return f"{text}\n{attached}"


def _persist_mcp(agent: Any, store: Any, thread_id: str) -> None:
    """Keep progressively loaded MCP remotes for the next turn's tool schema."""
    tools = getattr(agent, "_favlo_mcp", None)
    if not tools:
        return
    with contextlib.suppress(Exception):
        from web_api.mcp_tools import persist_loaded

        persist_loaded(list(tools), store, thread_id)


async def _open_agent(
    *,
    app: Any,
    store: Any,
    thread_id: str,
    agent_row: dict[str, Any],
    archive: bool,
    extra_note: str = "",
    cancel: asyncio.Event | None = None,
) -> tuple[Any, contextlib.AbstractAsyncContextManager[Any] | None]:
    reply = app.state.reply
    if reply is not stream_reply:
        return (
            _ScriptedAgent(
                reply,
                cancel or asyncio.Event(),
                list(agent_row.get("tool_groups") or []),
            ),
            None,
        )
    agent = await _build_agent(app, store, thread_id, agent_row, archive, extra_note)
    stack = contextlib.AsyncExitStack()
    await stack.__aenter__()
    try:
        from web_api.mcp_tools import open_clients, restore_loaded

        await open_clients(getattr(agent, "_favlo_mcp", []), stack)
        await restore_loaded(getattr(agent, "_favlo_mcp", []), store, thread_id)
    except Exception:
        await stack.aclose()
        raise
    return agent, stack


async def _build_agent(
    app: Any,
    store: Any,
    thread_id: str,
    agent_row: dict[str, Any],
    archive: bool,
    extra_note: str = "",
) -> Any:
    from agent_framework import Agent, InMemoryHistoryProvider

    from web_api.agent_factory import from_definition
    from web_api.chat import _mcp_for_definition, sampling_options, turn_setup
    from web_api.compaction import percent_window
    from web_api.mcp_tools import bundle

    agent_id = str(agent_row["id"])
    model_id, base_url = store.chat_model(agent_id)
    bus = app.state.bus
    pending = app.state.tasks
    mcp_text, mcp_tools = await bundle(store, agent_id, thread_id, bus, pending)
    frozen = store.definition(thread_id)
    note = "" if archive else extra_note
    if frozen is None:
        instructions, tool_list = turn_setup(
            str(agent_row.get("persona") or ""),
            str(agent_row.get("name") or "Agent"),
            list(agent_row.get("tool_groups") or []),
            base_url,
            store,
            agent_id,
            thread_id,
            bus,
            pending,
            mcp_text=mcp_text,
            mcp_tools=mcp_tools,
        )
        if note and note not in instructions:
            instructions = f"{instructions}\n\n{note}"
        options = sampling_options(
            agent_row.get("enable_thinking"),
            agent_row.get("preserve_thinking"),
            store.reasoning_effort(agent_id),
        ) or {}
        options = {**options, "store": True}
        from web_api.definition import build as build_definition
        from web_api.definition import hash_of

        payload = build_definition(
            store=store,
            agent={
                "id": agent_id,
                "name": agent_row.get("name"),
                "persona": agent_row.get("persona"),
                "tool_groups": list(agent_row.get("tool_groups") or []),
                "enable_thinking": agent_row.get("enable_thinking"),
                "preserve_thinking": agent_row.get("preserve_thinking"),
                "reasoning_effort": agent_row.get("reasoning_effort"),
                "default_model_id": None,
            },
            session_id=thread_id,
            base_url=base_url,
            model=str(model_id),
            mcp_text=mcp_text,
            mcp_tools=mcp_tools,
            bus=bus,
            pending=pending,
        )
        payload["instructions"] = instructions
        payload["options"] = options
        payload["hash"] = hash_of(payload)
        store.freeze_definition(thread_id, payload)
        from favlo.chat_client import open_chat_client

        client = open_chat_client(str(model_id), base_url)
        if tool_list:
            client.function_invocation_configuration["max_iterations"] = int(
                payload.get("max_iterations") or 6
            )
        agent = Agent(
            client=client,
            name=str(agent_row.get("name") or "Agent"),
            instructions=instructions,
            tools=tool_list,
            default_options=options,
            require_per_service_call_history_persistence=True,
            context_providers=[
                InMemoryHistoryProvider(skip_excluded=True, load_messages=False),
                percent_window(store.compaction_percent(thread_id)),
            ],
        )
        agent._favlo_mcp = mcp_tools  # type: ignore[attr-defined]
        return agent
    kept = _mcp_for_definition(frozen, mcp_tools)
    if note:
        frozen = dict(frozen)
        instructions = str(frozen.get("instructions") or "")
        if note not in instructions:
            frozen["instructions"] = f"{instructions}\n\n{note}"
    agent = from_definition(
        definition=frozen,
        store=store,
        session_id=thread_id,
        bus=bus,
        mcp_tools=kept,
    )
    agent._favlo_mcp = kept  # type: ignore[attr-defined]
    return agent


def _cleared_session(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") != "session.cleared":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("session_id"), str):
            return payload["session_id"]
    return None


async def _close_mcp(agent: Any) -> None:
    """Finish each MCP lifecycle task before the agent is dropped.

    A pending lifecycle task is destroyed on the event loop and that cleanup
    can stop the API from answering.
    """
    for tool in getattr(agent, "_favlo_mcp", []) or []:
        close = getattr(tool, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            continue


async def _close_client(agent: Any) -> None:
    client = getattr(agent, "client", None)
    if client is None:
        return
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        return
