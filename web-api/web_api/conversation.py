"""Conversation tools. One switch reads every stored transcript."""

from __future__ import annotations

from typing import Any

from web_api.tool_names import qualified

GROUP_ID = "conversation"
GROUP_NAME = "Conversation tools"
SOURCE = "conversation"
LIST_MAX = 20
SEARCH_MAX = 20
READ_MAX = 50
_SNIPPET = 240
_BODY = 2000

INSTRUCTIONS = (
    "When a question depends on an earlier message in any agent session, channel, "
    "or workspace chat, use the conversation tools before you answer. "
    f"Call {qualified(SOURCE, 'list_conversations')} to see which conversations exist. "
    f"Call {qualified(SOURCE, 'search_messages')} with the words you are looking for. "
    f"Call {qualified(SOURCE, 'read_recent_messages')} with the conversation id and how many of the latest "
    "messages you need. Ask for only as many as the question needs. "
    "A read returns those messages from oldest to newest. "
    "These tools cover every stored agent session, channel, and workspace chat."
)


def tools(store: Any) -> list[object]:
    return [_list_tool(store), _search_tool(store), _read_tool(store)]


def list_conversations(store: Any, kind: str, limit: int) -> str:
    chosen = _kind(kind)
    if chosen is None:
        return "Kind is agent, channel, workspace_chat, or all."
    rows = store.list_conversations(chosen, _cap(limit, 10, LIST_MAX))
    if not rows:
        return "No conversations are stored."
    lines = [_heading(row) for row in rows]
    return "\n".join(lines)


def search_messages(store: Any, query: str, kind: str, limit: int) -> str:
    text = query.strip()
    if not text:
        return "A search needs text."
    chosen = _kind(kind)
    if chosen is None:
        return "Kind is agent, channel, workspace_chat, or all."
    rows = store.search_messages(text, chosen, _cap(limit, 10, SEARCH_MAX))
    if not rows:
        return "No messages matched."
    lines = [
        f"{_heading(row)} {row['created_at']} {row['role']}: {_snippet(row['body'], text)}"
        for row in rows
    ]
    return "\n".join(lines)


def read_recent_messages(store: Any, conversation: str, limit: int) -> str:
    needle = conversation.strip()
    if not needle:
        return "Name a conversation."
    matches = store.conversations_matching(needle)
    chosen = _pick(matches, needle)
    if isinstance(chosen, str):
        return chosen
    count = _cap(limit, 20, READ_MAX)
    messages = store.recent_messages(chosen["id"], count)
    if not messages:
        return f"{_heading(chosen)} has no messages."
    lines = [f"Last {len(messages)} messages in {_heading(chosen)}"]
    for row in messages:
        lines.append(f"{row['created_at']} {row['role']}: {_short(row['body'])}")
    return "\n".join(lines)


def _list_tool(store: Any) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    @tool(approval_mode="never_require", name=qualified(SOURCE, "list_conversations"))
    def list_conversations_tool(
        kind: Annotated[
            str,
            Field(description="agent, channel, workspace_chat, or all."),
        ] = "all",
        limit: Annotated[
            int,
            Field(description="How many conversations to return. At most 20."),
        ] = 10,
    ) -> str:
        """List agent sessions, channels, and workspace chats.

        Use this to find a conversation id before reading its messages.
        """
        return list_conversations(store, kind, limit)

    return list_conversations_tool


def _search_tool(store: Any) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    @tool(approval_mode="never_require", name=qualified(SOURCE, "search_messages"))
    def search_messages_tool(
        query: Annotated[str, Field(description="Text to find in a message.")],
        kind: Annotated[
            str,
            Field(description="agent, channel, workspace_chat, or all."),
        ] = "all",
        limit: Annotated[int, Field(description="How many matches to return. At most 20.")] = 10,
    ) -> str:
        """Search message text in every agent session, channel, and workspace chat.

        Each match is a short snippet with its conversation id.
        """
        return search_messages(store, query, kind, limit)

    return search_messages_tool


def _read_tool(store: Any) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    @tool(approval_mode="never_require", name=qualified(SOURCE, "read_recent_messages"))
    def read_recent_messages_tool(
        conversation: Annotated[str, Field(description="A conversation id or its exact name.")],
        limit: Annotated[
            int,
            Field(description="How many of the latest messages to read. At most 50."),
        ] = 20,
    ) -> str:
        """Read the latest messages in one agent session, channel, or workspace chat.

        The messages come back from oldest to newest. Pass only as many as you need.
        """
        return read_recent_messages(store, conversation, limit)

    return read_recent_messages_tool


def _kind(value: str) -> str | None:
    chosen = value.strip().lower()
    if chosen in {"all", "agent", "channel", "workspace_chat"}:
        return chosen
    return None


def _cap(value: int, default: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number < 1:
        return 1
    return min(number, maximum)


def _heading(row: dict[str, Any]) -> str:
    label = {"agent": "agent session", "channel": "channel", "workspace_chat": "workspace chat"}
    return f"{row['name']} ({label.get(row['kind'], row['kind'])} {row['id']})"


def _pick(matches: list[dict[str, Any]], needle: str) -> dict[str, Any] | str:
    exact = [row for row in matches if row["id"] == needle]
    if len(exact) == 1:
        return exact[0]
    named = [row for row in matches if row["name"].casefold() == needle.casefold()]
    if len(named) == 1:
        return named[0]
    if len(matches) > 1:
        lines = ["More than one conversation matches.", *(_heading(row) for row in matches)]
        return "\n".join(lines)
    return f"No conversation is named {needle}."


def _snippet(body: str, query: str) -> str:
    folded = body.casefold()
    at = folded.find(query.strip().casefold())
    if at < 0:
        return _short(body)
    start = max(0, at - 80)
    end = min(len(body), at + len(query.strip()) + 160)
    piece = body[start:end].replace("\n", " ")
    if start > 0:
        piece = f"…{piece}"
    if end < len(body):
        piece = f"{piece}…"
    return piece[:_SNIPPET]


def _short(body: str) -> str:
    text = body.replace("\n", " ")
    if len(text) <= _BODY:
        return text
    return f"{text[:_BODY]}…"
