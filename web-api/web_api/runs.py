"""Run a stored user message and publish the reply."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from web_api.bus import Bus
from web_api.chat import ReplyPiece, public_call, record_tool, stream_reply
from web_api.events import envelope
from web_api.ids import guid
from web_api.reply_error import spoken_error
from web_api.store import Store

Reply = Callable[..., AsyncIterator[str | ReplyPiece]]


def _delta(context: dict[str, Any], run_id: str, agui: dict[str, Any]) -> dict[str, Any]:
    session_id = context["run"]["session_id"]
    return envelope(
        event_id=guid(),
        stream_id=f"session:{session_id}",
        event_type="run.delta",
        payload={"agui": agui},
        channel_id=context["session"]["channel_id"],
        session_id=session_id,
        run_id=run_id,
        actor={"kind": "agent", "id": context["agent"]["id"]},
    )


async def perform(
    store: Store,
    bus: Bus,
    cancel: asyncio.Event,
    run_id: str,
    reply: Reply = stream_reply,
    pending: set[asyncio.Task[Any]] | None = None,
) -> None:
    context = store.run_context(run_id)
    if not context["model"] or not context["base_url"]:
        await _publish(
            bus,
            store.finish_run(run_id, "failed", "Choose an enabled model in configuration."),
        )
        return
    message_id = guid()
    parts: list[str] = []
    reasoning: list[str] = []
    calls: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    try:
        await bus.publish(
            _delta(
                context,
                run_id,
                {"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"},
            )
        )
        gallery_note = ""
        agent = context["agent"] or {}
        folder = context.get("gallery_folder")
        image = context.get("gallery_image")
        attached = context.get("gallery_attachment")
        from web_api.gallery_tools import images_in_message, turn_note

        if folder is not None or image or attached:
            gallery_note = turn_note(folder or "", image)
        user_images = images_in_message(
            store,
            context["run"]["session_id"],
            context["user_text"],
        )
        async for piece in reply(
            base_url=context["base_url"],
            model=context["model"],
            persona=context["agent"]["persona"],
            agent_name=context["agent"]["name"],
            user_text=context["user_text"],
            tool_groups=context["agent"]["tool_groups"],
            store=store,
            agent_id=context["agent"]["id"] if context["agent"] else None,
            session_id=context["run"]["session_id"],
            bus=bus,
            pending=pending,
            cancel=cancel,
            gallery_note=gallery_note,
            user_images=user_images,
            enable_thinking=agent.get("enable_thinking"),
            preserve_thinking=agent.get("preserve_thinking"),
            reasoning_effort=agent.get("reasoning_effort"),
        ):
            if cancel.is_set():
                await _publish(bus, store.finish_run(run_id, "cancelled", None))
                return
            incoming_piece = _as_piece(piece)
            if incoming_piece.kind == "reasoning":
                if incoming_piece.text:
                    reasoning.append(incoming_piece.text)
                    _note_text(trace, "thinking", incoming_piece.text)
                    await bus.publish(
                        _delta(
                            context,
                            run_id,
                            {
                                "type": "REASONING_MESSAGE_CONTENT",
                                "messageId": message_id,
                                "delta": incoming_piece.text,
                            },
                        )
                    )
                continue
            if incoming_piece.kind in ("tool", "tool_result"):
                updated = record_tool(calls, incoming_piece)
                _note_tool(trace, updated)
                if updated["name"] or updated["arguments"] or updated["result"]:
                    shown = public_call(updated)
                    await bus.publish(
                        _delta(
                            context,
                            run_id,
                            {
                                "type": "TOOL_CALL",
                                "messageId": message_id,
                                "id": shown["id"],
                                "name": shown["name"],
                                "arguments": shown["arguments"],
                                "result": shown["result"],
                            },
                        )
                    )
                continue
            incoming = incoming_piece.text
            joined = "".join(parts)
            extra = incoming[len(joined) :] if incoming.startswith(joined) else incoming
            if not extra:
                continue
            parts.append(extra)
            _note_text(trace, "reply", extra)
            await bus.publish(
                _delta(
                    context,
                    run_id,
                    {"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id, "delta": extra},
                )
            )
        if cancel.is_set():
            await _publish(bus, store.finish_run(run_id, "cancelled", None))
            return
        text = "".join(parts).strip()
        thought = "".join(reasoning).strip()
        stored_calls = [public_call(row) for row in calls if row["name"] or row["arguments"]]
        stored_trace = public_trace(trace)
        if not text:
            archived = store.finish_archive(run_id)
            if archived is not None:
                await _publish(bus, archived)
                return
            if thought or stored_calls:
                await _publish(
                    bus,
                    store.complete_run(run_id, "", thought, stored_calls, stored_trace),
                )
                return
            await _publish(
                bus,
                store.finish_run(run_id, "failed", "The model returned an empty reply."),
            )
            return
        await bus.publish(
            _delta(context, run_id, {"type": "TEXT_MESSAGE_END", "messageId": message_id})
        )
        await _publish(
            bus,
            store.complete_run(run_id, text, thought, stored_calls, stored_trace),
        )
    except Exception as exc:
        await _publish(bus, store.finish_run(run_id, "failed", spoken_error(str(exc))))


def _note_text(trace: list[dict[str, Any]], kind: str, text: str) -> None:
    """Append one thinking or reply delta. A blank delta does not open a new part."""
    if not text:
        return
    if trace and trace[-1]["kind"] == kind:
        trace[-1]["text"] += text
        return
    if not text.strip():
        return
    trace.append({"kind": kind, "text": text})


def _note_tool(trace: list[dict[str, Any]], call: dict[str, Any]) -> None:
    """Keep one tool part for the call record, in the place it started."""
    if any(part.get("kind") == "tool" and part.get("call") is call for part in trace):
        return
    trace.append({"kind": "tool", "call": call})


def public_trace(trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The parts a stored turn shows, in the order they arrived."""
    shown: list[dict[str, str]] = []
    for part in trace:
        kind = part["kind"]
        if kind == "tool":
            call = public_call(part["call"])
            if call["name"] or call["arguments"]:
                shown.append({"kind": "tool", **call})
            continue
        text = str(part.get("text") or "")
        if kind == "thinking":
            text = text.strip()
        if text.strip():
            shown.append({"kind": kind, "text": text})
    return shown


def _as_piece(piece: str | ReplyPiece) -> ReplyPiece:
    if isinstance(piece, ReplyPiece):
        return piece
    return ReplyPiece("content", piece)


async def _publish(bus: Bus, events: list[dict[str, Any]]) -> None:
    for event in events:
        await bus.publish(event)
