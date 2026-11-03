"""AG-UI HTTP endpoint for Favlo chat turns."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Request
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from web_api.chat import ReplyPiece, stream_reply
from web_api.ids import guid
from web_api.reply_error import spoken_error
from web_api.session_commands import ARCHIVE_CUE
from web_api.store import Conflict, NotFound, Rejected

router = APIRouter()


def _user_text(messages: list[Any]) -> str:
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            ]
            text = "".join(parts).strip()
            if text:
                return text
    return ""


@router.post("/ag-ui")
async def ag_ui_run(request: Request) -> StreamingResponse:
    body = await request.json()
    thread_id = body.get("threadId") or body.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise Rejected("threadId is required.")
    thread_id = thread_id.strip()
    run_id = body.get("runId") or body.get("run_id") or guid()
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = guid()
    else:
        run_id = run_id.strip()
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise Rejected("messages must be a list.")
    archive = bool(body.get("favlo_archive"))
    text = "" if archive else _user_text(messages)
    if not archive and not text:
        raise Rejected("A user message is required.")
    store = request.app.state.store
    snapshot = store.session_snapshot(thread_id)
    session = snapshot["session"]
    agent_id = session.get("participant_id")
    if not agent_id:
        raise Rejected("That chat has no agent.")
    agent = next((row for row in store.agents() if row["id"] == agent_id), None)
    if agent is None:
        raise NotFound("That agent does not exist.")
    model_id, base_url = store.chat_model(agent_id)
    if not model_id or not base_url:
        raise Rejected("Choose an enabled model in configuration.")

    state = body.get("state")
    if isinstance(state, dict):
        store.save_agui_state(thread_id, state)

    if archive:
        running = next(
            (row for row in snapshot["runs"] if row["id"] == run_id and row["status"] == "running"),
            None,
        )
        if running is None:
            store.begin_agui_run(thread_id, run_id, "", archive=True)
        text = ARCHIVE_CUE
    else:
        started = store.begin_agui_run(thread_id, run_id, text, archive=False)
        for event in started.get("events") or []:
            await request.app.state.bus.publish(event)

    cancel = asyncio.Event()
    workspace_id = session.get("workspace_id")
    request.app.state.cancels[run_id] = cancel
    encoder = EventEncoder()
    reply = request.app.state.reply or stream_reply

    async def events() -> AsyncIterator[bytes]:
        message_id = guid()
        parts: list[str] = []
        try:
            yield encoder.encode(
                RunStartedEvent(type=EventType.RUN_STARTED, run_id=run_id, thread_id=thread_id)
            ).encode()
            yield encoder.encode(
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=message_id,
                    role="assistant",
                )
            ).encode()
            from web_api.gallery_tools import images_in_message, turn_note

            gallery_note = ""
            placement = store.agui_state(thread_id) or {}
            workspace = placement.get("favlo_workspace") if isinstance(placement, dict) else None
            selection = workspace.get("selection") if isinstance(workspace, dict) else None
            selected_path = None
            strip_count = None
            if isinstance(selection, dict):
                selected_path = selection.get("path")
            if isinstance(workspace, dict):
                for element in workspace.get("elements") or []:
                    if isinstance(element, dict) and element.get("element") == "filmstrip":
                        items = (element.get("props") or {}).get("items") or []
                        strip_count = len(items) if isinstance(items, list) else 0
            if selected_path or strip_count is not None:
                gallery_note = turn_note("", selected_path, strip_count=strip_count)
            user_images = images_in_message(store, thread_id, text) if not archive else []
            async for piece in reply(
                base_url=base_url,
                model=str(model_id),
                persona=str(agent.get("persona") or ""),
                agent_name=str(agent.get("name") or "Agent"),
                user_text=text,
                tool_groups=list(agent.get("tool_groups") or []),
                store=store,
                agent_id=agent_id,
                session_id=thread_id,
                bus=request.app.state.bus,
                pending=request.app.state.tasks,
                cancel=cancel,
                gallery_note=gallery_note,
                user_images=user_images,
                enable_thinking=agent.get("enable_thinking"),
                preserve_thinking=agent.get("preserve_thinking"),
                reasoning_effort=agent.get("reasoning_effort"),
            ):
                if cancel.is_set():
                    break
                incoming = piece if isinstance(piece, ReplyPiece) else ReplyPiece("content", str(piece))
                if incoming.kind == "content" and incoming.text:
                    parts.append(incoming.text)
                    yield encoder.encode(
                        TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=message_id,
                            delta=incoming.text,
                        )
                    ).encode()
                elif incoming.kind == "tool":
                    call_id = incoming.call_id or guid()
                    yield encoder.encode(
                        ToolCallStartEvent(
                            type=EventType.TOOL_CALL_START,
                            tool_call_id=call_id,
                            tool_call_name=incoming.name or "tool",
                            parent_message_id=message_id,
                        )
                    ).encode()
                    if incoming.text:
                        yield encoder.encode(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=call_id,
                                delta=incoming.text,
                            )
                        ).encode()
                elif incoming.kind == "tool_result":
                    yield encoder.encode(
                        ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=incoming.call_id or "",
                        )
                    ).encode()
                    if incoming.text:
                        yield encoder.encode(
                            ToolCallResultEvent(
                                type=EventType.TOOL_CALL_RESULT,
                                message_id=message_id,
                                tool_call_id=incoming.call_id or "",
                                content=incoming.text,
                                role="tool",
                            )
                        ).encode()
                    # workspace_place (and similar) update agui_state before the turn ends.
                    placed = store.agui_state(thread_id)
                    if isinstance(placed, dict) and placed:
                        yield encoder.encode(
                            StateSnapshotEvent(
                                type=EventType.STATE_SNAPSHOT,
                                snapshot=placed,
                            )
                        ).encode()
            yield encoder.encode(
                TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
            ).encode()
            result: dict[str, Any] = {"status": "completed"}
            if cancel.is_set():
                store.finish_run(run_id, "cancelled", None)
                result = {"status": "cancelled"}
            elif archive:
                cleared, clear_events = store.clear_chat(thread_id)
                for event in clear_events:
                    await request.app.state.bus.publish(event)
                result = {"status": "archived", "session_id": cleared["session_id"]}
            else:
                text_out = "".join(parts).strip()
                for event in store.complete_run(run_id, text_out):
                    await request.app.state.bus.publish(event)
            placed = store.agui_state(thread_id)
            if isinstance(placed, dict) and placed:
                yield encoder.encode(
                    StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT,
                        snapshot=placed,
                    )
                ).encode()
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    run_id=run_id,
                    thread_id=thread_id,
                    result=result,
                )
            ).encode()
        except (Conflict, NotFound, Rejected) as exc:
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=str(exc) or "The run failed.",
                    code=type(exc).__name__,
                )
            ).encode()
        except Exception as exc:
            message = spoken_error(str(exc))
            try:
                for event in store.finish_run(run_id, "failed", message):
                    await request.app.state.bus.publish(event)
            except Exception:
                pass
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=message,
                    code=type(exc).__name__,
                )
            ).encode()
        finally:
            # Disconnect or Stop must wake the MAF abort (stream.close + chat client)
            # before this entry is dropped, or a long tool turn keeps running.
            pending = request.app.state.cancels.pop(run_id, None)
            if pending is not None:
                pending.set()
            try:
                for event in store.finish_run(run_id, "cancelled", None):
                    await request.app.state.bus.publish(event)
            except Exception:
                pass
            # Path-only index after the turn; vision enrichment stays on the idle watcher
            # so a chat image attach still owns the first vision embed.
            if workspace_id:
                from web_api.gallery_embed import request_wake

                request_wake(request.app, workspace_id, vision=False)

    # Same transport keepalive as agent-framework-ag-ui (keepalive_seconds=15).
    return EventSourceResponse(
        events(),
        ping=15,
        ping_message_factory=lambda: ServerSentEvent(comment="keepalive"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
