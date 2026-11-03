"""Open one chat turn from a frozen definition or a first-turn freeze."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from web_api.chat import ReplyPiece, stream_reply


async def run_stream(
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
    cancel: Any | None = None,
    gallery_note: str = "",
    user_images: list[tuple[bytes, str]] | None = None,
    enable_thinking: bool | None = None,
    preserve_thinking: bool | None = None,
    reasoning_effort: str | None = None,
) -> AsyncIterator[ReplyPiece]:
    """Stream one reply. Delegates to ``stream_reply`` until that path is retired."""
    async for piece in stream_reply(
        base_url=base_url,
        model=model,
        persona=persona,
        agent_name=agent_name,
        user_text=user_text,
        tool_groups=tool_groups,
        store=store,
        agent_id=agent_id,
        session_id=session_id,
        bus=bus,
        pending=pending,
        cancel=cancel,
        gallery_note=gallery_note,
        user_images=user_images,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        reasoning_effort=reasoning_effort,
    ):
        yield piece


async def run_server_turn(
    store: Any,
    bus: Any,
    cancel: Any,
    run_id: str,
    pending: set[Any] | None = None,
) -> None:
    """Drive a heartbeat or MCP follow-up through the same reply path."""
    from web_api.runs import perform

    await perform(store, bus, cancel, run_id, stream_reply, pending)
