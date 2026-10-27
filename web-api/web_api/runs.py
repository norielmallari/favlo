"""Run a stored user message and publish the reply."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from web_api.bus import Bus
from web_api.chat import stream_reply
from web_api.events import envelope
from web_api.ids import mint
from web_api.store import Store

Reply = Callable[..., AsyncIterator[str]]


def _delta(context: dict[str, Any], run_id: str, agui: dict[str, Any]) -> dict[str, Any]:
    session_id = context["run"]["session_id"]
    return envelope(
        event_id=mint("evt"),
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
    message_id = mint("message")
    parts: list[str] = []
    try:
        await bus.publish(
            _delta(
                context,
                run_id,
                {"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"},
            )
        )
        async for piece in reply(
            base_url=context["base_url"],
            model=context["model"],
            persona=context["agent"]["persona"],
            agent_name=context["agent"]["name"],
            history=context["prior"],
            user_text=context["user_text"],
            tool_groups=context["agent"]["tool_groups"],
            store=store,
            agent_id=context["agent"]["id"] if context["agent"] else None,
            session_id=context["run"]["session_id"],
            bus=bus,
            pending=pending,
        ):
            if cancel.is_set():
                await _publish(bus, store.finish_run(run_id, "cancelled", None))
                return
            extra = piece[len("".join(parts)) :] if piece.startswith("".join(parts)) else piece
            if not extra:
                continue
            parts.append(extra)
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
        if not text:
            await _publish(
                bus,
                store.finish_run(run_id, "failed", "The model returned an empty reply."),
            )
            return
        await bus.publish(
            _delta(context, run_id, {"type": "TEXT_MESSAGE_END", "messageId": message_id})
        )
        await _publish(bus, store.complete_run(run_id, text))
    except Exception as exc:
        message = str(exc).strip() or "The listener did not complete the reply."
        await _publish(bus, store.finish_run(run_id, "failed", message[:400]))


async def _publish(bus: Bus, events: list[dict[str, Any]]) -> None:
    for event in events:
        await bus.publish(event)
