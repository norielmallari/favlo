"""Process-local Responses chain for one listener.

``store`` defaults to on. ``store`` false leaves the body unchanged and drops
the response. A later call names the kept response with ``previous_response_id``.
This module rebuilds one transcript, the same input a ``store`` false client
would replay, and that transcript is what ``handle_responses`` encodes. The
pipeline does not read ``store`` or ``previous_response_id``.

A Microsoft Agent Framework client sends the previous id and also resends user
and assistant messages. Those copies are dropped. A new user message and a
``function_call_output`` stay. An image that is already in the chain stays once.

The chain lives in the listener process. A missing id starts a new chain
instead of failing the turn. A restart of ``favlo serve`` drops the chain.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


class ChainError(Exception):
    """A conversation id was named and this process does not have it."""


def remember(
    state: dict[str, Any] | None,
    body: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """Keep a response when store is on, and append it to a conversation."""
    if not isinstance(state, dict):
        return
    if body.get("store", True) is not False:
        store = state.setdefault("responses", {})
        store[response["id"]] = response
        inputs = state.setdefault("response_inputs", {})
        inputs[response["id"]] = _input_list(body.get("input"))
    conversation = body.get("conversation")
    if isinstance(conversation, str) and conversation:
        bucket = state.setdefault("conversations", {})
        container = bucket.setdefault(
            conversation,
            {"id": conversation, "object": "conversation", "items": []},
        )
        container["items"].extend(_input_list(body.get("input")))
        container["items"].extend(response.get("output") or [])


class Turn:
    """One ``store`` true call after the transcript has been rebuilt."""

    def __init__(self, body: dict[str, Any], previous: str | None) -> None:
        self.body = body
        self.previous = previous


def present(
    body: dict[str, Any], state: dict[str, Any] | None
) -> tuple[dict[str, Any], Turn | None]:
    """The body ``handle_responses`` encodes, and the turn to keep afterwards.

    ``store`` false returns ``body`` and no turn. Otherwise the stored chain
    becomes ``input``, repeated client messages are dropped, and ``store`` and
    ``previous_response_id`` are left off the body the pipeline reads.
    """
    if body.get("store", True) is False:
        return body, None
    if body.get("conversation") and body.get("previous_response_id"):
        raise ChainError("conversation and previous_response_id are not set together")
    previous = body.get("previous_response_id")
    named = previous if isinstance(previous, str) and previous else None
    presented = dict(body)
    presented["input"] = joined_input(body, state)
    presented.pop("store", None)
    presented.pop("previous_response_id", None)
    return presented, Turn(presented, named)


def finish_turn(turn: Turn | None, reply: Any, state: dict[str, Any] | None) -> None:
    """Keep the response the pipeline wrote, when this turn stores."""
    if turn is None:
        return
    payload = getattr(reply, "payload", None)
    if not isinstance(payload, dict) or payload.get("object") != "response":
        return
    if not payload.get("id"):
        return
    remember(state, {**turn.body, "store": True}, payload)


def recording_reply(reply: Any, *, previous: str | None) -> Any:
    """Forward the reply and keep the Response the pipeline writes."""
    return _RecordingReply(reply, previous)


def joined_input(body: dict[str, Any], state: dict[str, Any] | None) -> Any:
    """Input with a stored chain prepended.

    An unknown ``previous_response_id`` is ignored so the turn starts a new
    chain. An unknown ``conversation`` still fails.
    """
    fresh = body.get("input")
    previous = body.get("previous_response_id")
    conversation = body.get("conversation")
    stored: list[Any] = []
    if isinstance(previous, str) and previous and isinstance(state, dict):
        found = stored_response(state, previous)
        if found is not None:
            stored.extend(_stored_input(state, previous))
            stored.extend(found.get("output") or [])
    if isinstance(conversation, str) and conversation:
        if not isinstance(state, dict):
            raise ChainError("conversation is not stored")
        bucket = state.get("conversations")
        container = bucket.get(conversation) if isinstance(bucket, dict) else None
        if not isinstance(container, dict):
            raise ChainError("conversation is not stored")
        stored = list(container.get("items") or []) + stored
    if not stored:
        return fresh
    return [*stored, *_unsent(stored, fresh)]


def stored_response(state: dict[str, Any], identifier: str) -> dict[str, Any] | None:
    store = state.get("responses")
    if not isinstance(store, dict):
        return None
    found = store.get(identifier)
    return found if isinstance(found, dict) else None


def drop_response(state: dict[str, Any], identifier: str) -> bool:
    store = state.get("responses")
    if not isinstance(store, dict) or identifier not in store:
        return False
    del store[identifier]
    inputs = state.get("response_inputs")
    if isinstance(inputs, dict):
        inputs.pop(identifier, None)
    return True


def _stored_input(state: dict[str, Any], identifier: str) -> list[Any]:
    inputs = state.get("response_inputs")
    if not isinstance(inputs, dict):
        return []
    found = inputs.get(identifier)
    return list(found) if isinstance(found, list) else []


def _input_list(raw: Any) -> list[Any]:
    if isinstance(raw, str):
        return [{"type": "message", "role": "user", "content": raw}]
    if isinstance(raw, list):
        return list(raw)
    return []


def _unsent(stored: list[Any], fresh: Any) -> list[Any]:
    """Client items that are not already in the stored transcript.

    The Agent Framework resends user and assistant messages along with
    ``previous_response_id``. A second copy would embed an image again.
    """
    counts: Counter[tuple[Any, ...]] = Counter(_item_key(item) for item in stored)
    kept: list[Any] = []
    for item in _input_list(fresh):
        # A tool result is new even when an earlier turn used the same call id
        # and the same text. Dropping it leaves two calls in one assistant turn.
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            kept.append(item)
            continue
        key = _item_key(item)
        if counts[key] > 0:
            counts[key] -= 1
            continue
        kept.append(item)
    return kept


def _item_key(item: Any) -> tuple[Any, ...]:
    if isinstance(item, str):
        return ("message", "user", _content_key(item))
    if not isinstance(item, dict):
        return ("raw", repr(item))
    kind = item.get("type")
    if kind in ("message", None) and item.get("role"):
        return ("message", str(item.get("role") or "user"), _content_key(item.get("content")))
    if kind == "reasoning":
        return ("reasoning", _content_key(item.get("content")))
    if kind == "function_call":
        return (
            "function_call",
            str(item.get("call_id") or item.get("id") or ""),
            str(item.get("name") or ""),
            str(item.get("arguments") or ""),
        )
    if kind == "function_call_output":
        return (
            "function_call_output",
            str(item.get("call_id") or ""),
            str(item.get("output") or ""),
        )
    if kind == "input_image":
        return ("image", _image_url(item), _detail(item))
    return ("item", str(kind or ""), _content_key(item.get("content")))


def _content_key(content: Any) -> tuple[Any, ...]:
    if isinstance(content, str):
        return (("text", content),)
    if not isinstance(content, list):
        return (("raw", repr(content)),)
    parts: list[tuple[Any, ...]] = []
    for part in content:
        if isinstance(part, str):
            parts.append(("text", part))
            continue
        if not isinstance(part, dict):
            parts.append(("raw", repr(part)))
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text", "reasoning_text") or (
            kind is None and "text" in part
        ):
            parts.append(("text", str(part.get("text") or "")))
            continue
        if kind in ("input_image", "image_url") or "image_url" in part or "image" in part:
            parts.append(("image", _image_url(part), _detail(part)))
            continue
        parts.append(("part", str(kind or ""), str(part.get("text") or "")))
    return tuple(parts)


def _image_url(item: dict[str, Any]) -> str:
    raw = item.get("image_url", item.get("image", item.get("url")))
    if isinstance(raw, dict):
        raw = raw.get("url")
    if not isinstance(raw, str):
        return ""
    return "".join(raw.split())


def _detail(item: dict[str, Any]) -> str:
    nested = item.get("image_url")
    if isinstance(nested, dict) and nested.get("detail"):
        return str(nested["detail"])
    detail = item.get("detail")
    return str(detail) if detail else "auto"


class _RecordingReply:
    """Copies the Response onto this object and echoes ``previous_response_id``."""

    def __init__(self, inner: Any, previous: str | None) -> None:
        self._inner = inner
        self._previous = previous
        self.payload: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_inner", "_previous", "payload"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._inner, name, value)

    def json(self, status: int, payload: dict[str, Any]) -> None:
        self._keep(payload)
        self._inner.json(status, payload)

    def event(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "response.completed":
            response = payload.get("response")
            if isinstance(response, dict):
                self._keep(response)
        self._inner.event(payload)

    def _keep(self, payload: dict[str, Any]) -> None:
        if payload.get("object") != "response":
            return
        if self._previous and "previous_response_id" not in payload:
            payload["previous_response_id"] = self._previous
        self.payload = payload
