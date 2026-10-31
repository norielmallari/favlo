"""Tool-call text from this checkpoint's chat template."""

from __future__ import annotations

import json
from typing import Any

_OPEN = "<|tool_call>call:"
_CLOSE = "<tool_call|>"
_QUOTE = '<|"|>'


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Leading text, then each complete ``<|tool_call>call:`` block."""
    calls: list[dict[str, Any]] = []
    cursor = 0
    prefix_end: int | None = None
    while True:
        start = text.find(_OPEN, cursor)
        if start < 0:
            break
        name_at = start + len(_OPEN)
        brace = text.find("{", name_at)
        close = text.find(_CLOSE, name_at)
        if brace < 0 or close < 0 or brace > close:
            break
        if prefix_end is None:
            prefix_end = start
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
    if not calls or prefix_end is None:
        return text, []
    prefix = text[:prefix_end].strip()
    return (prefix or None), calls


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


def _skip(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index
