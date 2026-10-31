"""Session tools. One switch lets an agent clear the conversation it is in."""

from __future__ import annotations

from typing import Any

from web_api.tool_names import qualified

GROUP_ID = "session"
GROUP_NAME = "Session tools"
SOURCE = "session"

INSTRUCTIONS = (
    "When someone asks to clear, reset, or forget this conversation, call "
    f"{qualified(SOURCE, 'clear_session')}. "
    "That drops the stored turns and clears the in-memory session. "
    "Do not repeat anything from before that call."
)


def tools(store: Any, session_id: str, bus: Any) -> list[object]:
    return [_clear_tool(store, session_id, bus)]


def clear_maf_session(session: Any) -> None:
    """Drop the messages on this Agent Framework session."""
    from agent_framework import InMemoryHistoryProvider

    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return
    bucket = state.setdefault(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
    if not isinstance(bucket, dict):
        state[InMemoryHistoryProvider.DEFAULT_SOURCE_ID] = {"messages": []}
        return
    messages = bucket.get("messages")
    if isinstance(messages, list):
        messages.clear()
    else:
        bucket["messages"] = []


async def clear_current_session(store: Any, session_id: str, bus: Any, session: Any) -> str:
    clear_maf_session(session)
    from web_api.store import NotFound

    try:
        events = store.clear_session_messages(session_id)
    except NotFound as exc:
        return str(exc)
    if bus is not None:
        for event in events:
            await bus.publish(event)
    return "This session is clear."


def _clear_tool(store: Any, session_id: str, bus: Any) -> object:
    from agent_framework import FunctionInvocationContext, tool

    @tool(approval_mode="never_require", name=qualified(SOURCE, "clear_session"))
    async def clear_session_tool(ctx: FunctionInvocationContext) -> str:
        """Clear this conversation.

        Call this when someone asks to clear, reset, or forget the current session.
        """
        return await clear_current_session(store, session_id, bus, ctx.session)

    return clear_session_tool
