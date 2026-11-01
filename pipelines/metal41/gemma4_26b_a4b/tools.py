"""Tool-call text from this checkpoint's chat template."""

from __future__ import annotations

import json
from typing import Any

_OPEN = "<|tool_call>call:"
_CLOSE = "<tool_call|>"
_QUOTE = '<|"|>'


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Text outside complete ``<|tool_call>call:`` blocks, then those calls.

    Text before the first call, between calls, and after the last call stays
    in the reply. An unclosed call stays in the reply too.
    """
    calls: list[dict[str, Any]] = []
    pieces: list[str] = []
    cursor = 0
    while True:
        start = text.find(_OPEN, cursor)
        if start < 0:
            tail = text[cursor:].strip()
            if tail:
                pieces.append(tail)
            break
        gap = text[cursor:start].strip()
        if gap:
            pieces.append(gap)
        name_at = start + len(_OPEN)
        brace = text.find("{", name_at)
        close = text.find(_CLOSE, name_at)
        if brace < 0 or close < 0 or brace > close:
            rest = text[start:].strip()
            if rest:
                pieces.append(rest)
            break
        name = text[name_at:brace].strip()
        body_end = close - 1 if text[close - 1] == "}" else close
        arguments = _arguments(text[brace + 1 : body_end])
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
        cursor = close + len(_CLOSE)
    if not calls:
        return text, []
    body = "\n".join(pieces).strip()
    return (body or None), calls


def _arguments(body: str) -> dict[str, Any]:
    stripped = body.strip()
    if not stripped:
        return {}
    if _QUOTE not in stripped:
        try:
            parsed = json.loads("{" + stripped + "}")
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    values, _end = _object("{" + body + "}", 0)
    return values


def _object(text: str, index: int) -> tuple[dict[str, Any], int]:
    values: dict[str, Any] = {}
    index += 1
    length = len(text)
    while index < length:
        index = _skip(text, index)
        if index < length and text[index] == "}":
            return values, index + 1
        key, index = _key(text, index)
        index = _skip(text, index)
        if index >= length or text[index] != ":":
            break
        value, index = _value(text, index + 1)
        if key:
            values[key] = value
        index = _skip(text, index)
        if index < length and text[index] == ",":
            index += 1
    return values, index


def _key(text: str, index: int) -> tuple[str, int]:
    index = _skip(text, index)
    if text.startswith(_QUOTE, index):
        value, index = _string(text, index)
        return value, index
    start = index
    while index < len(text) and text[index] not in ":},":
        index += 1
    return text[start:index].strip().strip('"'), index


def _value(text: str, index: int) -> tuple[Any, int]:
    index = _skip(text, index)
    if index >= len(text):
        return "", index
    if text.startswith(_QUOTE, index):
        return _string(text, index)
    if text[index] == "{":
        return _object(text, index)
    if text[index] == "[":
        return _array(text, index)
    if text[index] == '"':
        return _json_string(text, index)
    start = index
    while index < len(text) and text[index] not in ",}]":
        index += 1
    token = text[start:index].strip()
    if token == "true":
        return True, index
    if token == "false":
        return False, index
    if token in {"null", "none"}:
        return None, index
    try:
        if "." in token:
            return float(token), index
        return int(token), index
    except ValueError:
        return token, index


def _array(text: str, index: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    index += 1
    length = len(text)
    while index < length:
        index = _skip(text, index)
        if index < length and text[index] == "]":
            return items, index + 1
        value, index = _value(text, index)
        items.append(value)
        index = _skip(text, index)
        if index < length and text[index] == ",":
            index += 1
    return items, index


def _string(text: str, index: int) -> tuple[str, int]:
    index += len(_QUOTE)
    end = text.find(_QUOTE, index)
    if end < 0:
        return text[index:], len(text)
    return text[index:end], end + len(_QUOTE)


def _json_string(text: str, index: int) -> tuple[str, int]:
    end = index + 1
    while end < len(text):
        if text[end] == "\\" and end + 1 < len(text):
            end += 2
            continue
        if text[end] == '"':
            try:
                return json.loads(text[index : end + 1]), end + 1
            except json.JSONDecodeError:
                return text[index + 1 : end], end + 1
        end += 1
    return text[index:], len(text)


_THOUGHT = "<|channel>thought"
_THOUGHT_CLOSE = "<channel|>"


def _held_suffix(text: str, marker: str) -> int:
    """How many trailing characters of ``text`` are a proper prefix of ``marker``."""
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


class StreamText:
    """Reasoning and reply as tokens arrive.

    An open thought, including one the prompt already started, is reasoning
    until ``<channel|>`` or ``<|tool_call>``. A finished call is held out of
    the reply so the caller can send ``tool_calls``. Text after that call is
    split again. An unclosed call is the reply.
    """

    def __init__(self, *, channel_open: bool) -> None:
        self._reasoning = channel_open
        self._held = ""
        self._call = ""
        self._in_call = False

    def push(self, piece: str) -> list[tuple[str, str]]:
        if piece == "":
            return []
        if self._in_call:
            self._call += piece
            return self._close_call()
        self._held += piece
        if self._reasoning:
            return self._push_reasoning()
        return self._push_reply()

    def finish(self) -> list[tuple[str, str]]:
        if self._in_call:
            text = self._call
            self._call = ""
            self._in_call = False
            return [("content", text)] if text else []
        if not self._held:
            return []
        channel = "reasoning" if self._reasoning else "content"
        text = self._held
        self._held = ""
        return [(channel, text)] if text else []

    def _close_call(self) -> list[tuple[str, str]]:
        at = self._call.find(_CLOSE)
        if at < 0:
            return []
        rest = self._call[at + len(_CLOSE) :]
        self._call = ""
        self._in_call = False
        self._reasoning = False
        if rest.startswith("\n"):
            rest = rest[1:]
        return self.push(rest)

    def _push_reasoning(self) -> list[tuple[str, str]]:
        close_at = self._held.find(_THOUGHT_CLOSE)
        tool_at = self._held.find(_OPEN)
        if tool_at >= 0 and (close_at < 0 or tool_at < close_at):
            reasoning = self._held[:tool_at]
            rest = self._held[tool_at:]
            self._held = ""
            self._reasoning = False
            self._call = rest
            self._in_call = True
            out = [("reasoning", reasoning)] if reasoning else []
            out.extend(self._close_call())
            return out
        if close_at < 0:
            return self._emit_held("reasoning", _THOUGHT_CLOSE, _OPEN)
        reasoning = self._held[:close_at]
        rest = self._held[close_at + len(_THOUGHT_CLOSE) :]
        self._held = ""
        self._reasoning = False
        if rest.startswith("\n"):
            rest = rest[1:]
        out = [("reasoning", reasoning)] if reasoning else []
        out.extend(self.push(rest))
        return out

    def _push_reply(self) -> list[tuple[str, str]]:
        tool_at = self._held.find(_OPEN)
        thought_at = self._held.find(_THOUGHT)
        if tool_at >= 0 and (thought_at < 0 or tool_at < thought_at):
            emit = self._held[:tool_at]
            self._call = self._held[tool_at:]
            self._held = ""
            self._in_call = True
            out = [("content", emit)] if emit else []
            out.extend(self._close_call())
            return out
        if thought_at >= 0:
            emit = self._held[:thought_at]
            rest = self._held[thought_at + len(_THOUGHT) :]
            self._held = ""
            self._reasoning = True
            if rest.startswith("\n"):
                rest = rest[1:]
            out = [("content", emit)] if emit else []
            out.extend(self.push(rest))
            return out
        return self._emit_held("content", _OPEN, _THOUGHT)

    def _emit_held(self, channel: str, *markers: str) -> list[tuple[str, str]]:
        keep = max(_held_suffix(self._held, marker) for marker in markers)
        if keep == 0:
            emit = self._held
            self._held = ""
        else:
            emit = self._held[:-keep]
            self._held = self._held[-keep:]
        return [(channel, emit)] if emit else []


def _skip(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index
