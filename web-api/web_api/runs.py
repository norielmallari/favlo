"""Drive an already-open run through the AG-UI host."""

from __future__ import annotations

import asyncio
from typing import Any

from web_api.bus import Bus
from web_api.chat import ReplyPiece, public_call
from web_api.session_commands import ARCHIVE_CUE
from web_api.store import NotFound, Rejected


async def perform(app: Any, cancel: asyncio.Event, run_id: str) -> None:
    """One heartbeat, greeting, follow-up, or REST message, on the library runner."""
    store = app.state.store
    bus = app.state.bus
    try:
        context = await asyncio.to_thread(store.run_context, run_id)
    except (Rejected, NotFound) as exc:
        await _publish(bus, store.finish_run(run_id, "failed", str(exc)))
        return
    session_id = str(context["run"]["session_id"])
    text = str(context["user_text"] or "")
    cue = str(context["run"].get("cue") or "")
    props: dict[str, Any] = {"favloAdopt": True}
    if cue == ARCHIVE_CUE:
        props["favloArchive"] = True
    elif cue:
        props["favloSilent"] = True
    app.state.cancels[run_id] = cancel
    host = app.state.agui_host
    async for _event in host.run(
        {
            "thread_id": session_id,
            "run_id": run_id,
            "messages": [] if cue == ARCHIVE_CUE else [{"role": "user", "content": text}],
            "forwarded_props": props,
        }
    ):
        if cancel.is_set():
            return


def _note_reasoning(trace: list[dict[str, Any]], text: str) -> None:
    """Record a thinking delta. Text recovered after the reply starts stays above that reply."""
    if not text:
        return
    if any(part.get("kind") == "thinking" for part in trace):
        _note_text(trace, "thinking", text)
        return
    if not text.strip():
        return
    if trace:
        trace.insert(0, {"kind": "thinking", "text": text})
        return
    trace.append({"kind": "thinking", "text": text})


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
