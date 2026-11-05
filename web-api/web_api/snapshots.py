"""Durable AG-UI thread snapshots and the REST view of those messages."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_framework_ag_ui import AGUIThreadSnapshot

from web_api.ids import guid


def project_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Map a library snapshot onto the session rows the transcript already reads.

    Reasoning stays on the following assistant row. A blank assistant row
    (whitespace only, no reply) does not become its own turn, so thinking
    before a tool stays with the reply. Tool rows stay out of that view.
    Fenced host cues (``metadata.favlo.ui=hidden``) stay out of the You bubble.
    """
    from web_api.host_cues import is_host_cue

    out: list[dict[str, Any]] = []
    pending: list[str] = []
    pending_calls: list[dict[str, str]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        if is_host_cue(raw):
            continue
        role = str(raw.get("role") or "")
        if role == "reasoning":
            text = _text(raw.get("content"))
            if text:
                pending.append(text)
            continue
        if role == "tool":
            _attach_result(pending_calls, raw)
            continue
        if role not in {"user", "assistant"}:
            continue
        body = _text(raw.get("content"))
        if role == "assistant" and not body.strip():
            # The model often writes a whitespace assistant row between
            # reasoning and a tool call. Keep that reasoning for the reply.
            pending_calls.extend(_tool_calls(raw))
            continue
        if role == "user":
            body = _typed_text(body)
            if not body.strip():
                # Silent server turns keep an empty user row for the protocol only.
                continue
            pending = []
            pending_calls = []
            calls = []
            reasoning = None
            trace = None
        else:
            reasoning = "\n".join(pending).strip() or None
            pending = []
            calls = pending_calls or _tool_calls(raw)
            pending_calls = []
            trace = _trace(reasoning, calls, body)
        out.append(
            {
                "id": str(raw.get("id") or guid()),
                "role": role,
                "body": body,
                "reasoning": reasoning,
                "tool_calls": calls or None,
                "trace": trace or None,
            }
        )
    if pending or pending_calls:
        reasoning = "\n".join(pending).strip() or None
        trace = _trace(reasoning, pending_calls, "")
        out.append(
            {
                "id": guid(),
                "role": "assistant",
                "body": "",
                "reasoning": reasoning,
                "tool_calls": pending_calls or None,
                "trace": trace or None,
            }
        )
    return out


def _attach_result(calls: list[dict[str, str]], message: dict[str, Any]) -> None:
    call_id = str(message.get("tool_call_id") or message.get("toolCallId") or "")
    result = _text(message.get("content"))
    for call in calls:
        if call["id"] == call_id:
            call["result"] = result
            return
    if calls and not call_id:
        calls[-1]["result"] = result


def _tool_calls(message: dict[str, Any]) -> list[dict[str, str]]:
    raw = message.get("toolCalls") or message.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    calls: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("id") or item.get("toolCallId") or "")
        name = str(item.get("function", {}).get("name") if isinstance(item.get("function"), dict) else item.get("name") or "")
        arguments = item.get("function", {}).get("arguments") if isinstance(item.get("function"), dict) else item.get("arguments")
        if not isinstance(arguments, str):
            arguments = "" if arguments is None else str(arguments)
        result = item.get("result")
        if not isinstance(result, str):
            result = "" if result is None else str(result)
        if call_id or name or arguments:
            calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                }
            )
    return calls


def _trace(
    reasoning: str | None,
    calls: list[dict[str, str]],
    body: str,
) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    if reasoning:
        parts.append({"kind": "thinking", "text": reasoning})
    for call in calls:
        if call["name"] or call["arguments"]:
            parts.append({"kind": "tool", **call})
    if body.strip():
        parts.append({"kind": "reply", "text": body})
    return parts


def _typed_text(body: str) -> str:
    """Drop a selection fence riding on the user message. The model still receives it."""
    from web_api.host_cues import FENCE_OPEN

    start = body.find(FENCE_OPEN)
    if start == -1:
        return body
    return body[:start].strip()


def message_text(content: Any) -> str:
    """Visible text of one snapshot message."""
    return _text(content)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") in {None, "text", "input_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


class FavloSnapshotStore:
    """``AGUIThreadSnapshotStore`` backed by the web database."""

    def __init__(self, app: Any) -> None:
        self._app = app

    def _store(self) -> Any:
        store = getattr(self._app.state, "store", None)
        if store is None:
            raise RuntimeError("The web store is not open.")
        return store

    async def save(self, *, scope: str, thread_id: str, snapshot: AGUIThreadSnapshot) -> None:
        document = {
            "messages": list(snapshot.messages or []),
            "state": snapshot.state,
            "interrupt": snapshot.interrupt,
            "session_state": snapshot.session_state,
        }
        store = self._store()
        await asyncio.to_thread(store.save_thread_snapshot, scope, thread_id, document)
        if isinstance(snapshot.state, dict) and snapshot.state:
            await asyncio.to_thread(store.save_agui_state, thread_id, snapshot.state)

    async def get(self, *, scope: str, thread_id: str) -> AGUIThreadSnapshot | None:
        document = await asyncio.to_thread(self._store().thread_snapshot, scope, thread_id)
        if not isinstance(document, dict):
            return None
        messages = document.get("messages")
        return AGUIThreadSnapshot(
            messages=list(messages) if isinstance(messages, list) else [],
            state=document.get("state") if isinstance(document.get("state"), dict) else None,
            interrupt=document.get("interrupt") if isinstance(document.get("interrupt"), list) else None,
            session_state=document.get("session_state")
            if isinstance(document.get("session_state"), dict)
            else None,
        )

    async def delete(self, *, scope: str, thread_id: str) -> bool:
        return bool(await asyncio.to_thread(self._store().delete_thread_snapshot, scope, thread_id))

    async def clear(self, *, scope: str | None = None) -> None:
        await asyncio.to_thread(self._store().clear_thread_snapshots, scope)
