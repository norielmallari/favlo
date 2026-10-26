"""Tool-call text from this checkpoint's chat template."""

from __future__ import annotations

import json
import re
from typing import Any

_CALL = re.compile(
    r"<tool_call>\s*<function=([^\n>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM = re.compile(r"<parameter=([^\n>]+)>\n?(.*?)</parameter>", re.DOTALL)
_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")
_UNICODE_TAIL = re.compile(r"(?:\\u[0-9a-fA-F]{0,3}|\\)$")
_HIGH_SURROGATE = re.compile(r"\\u[Dd][89ABab][0-9A-Fa-f]{2}$")


def decode_unicode_escapes(text: str) -> str:
    """Replace each ``\\uXXXX`` escape with its character."""
    if "\\u" not in text:
        return text
    replaced = _UNICODE.sub(lambda match: chr(int(match.group(1), 16)), text)
    if any(0xD800 <= ord(char) <= 0xDFFF for char in replaced):
        return replaced.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
    return replaced


def decode_shown_text(text: str) -> str:
    """Decode ``\\uXXXX`` in reply text. A JSON parameter body stays encoded."""
    if "\\u" not in text:
        return text
    parts: list[str] = []
    cursor = 0
    for match in _PARAM.finditer(text):
        parts.append(decode_unicode_escapes(text[cursor : match.start()]))
        raw = match.group(0)
        value = match.group(2)
        if _json_container(value.strip()) is None:
            decoded = decode_unicode_escapes(value)
            rel = match.start(2) - match.start()
            raw = raw[:rel] + decoded + raw[match.end(2) - match.start() :]
        parts.append(raw)
        cursor = match.end()
    parts.append(decode_unicode_escapes(text[cursor:]))
    return "".join(parts)


def _unicode_tail(text: str) -> tuple[str, str]:
    """A trailing incomplete ``\\uXXXX``, or a high surrogate waiting for its pair."""
    match = _UNICODE_TAIL.search(text)
    if match is not None:
        return text[: match.start()], match.group(0)
    match = _HIGH_SURROGATE.search(text)
    if match is not None:
        return text[: match.start()], match.group(0)
    return text, ""


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Leading text, then each complete ``<tool_call>`` block."""
    matches = list(_CALL.finditer(text))
    if not matches:
        return decode_unicode_escapes(text), []
    calls = []
    for index, match in enumerate(matches):
        arguments = {
            name.strip(): _parameter_value(value) for name, value in _PARAM.findall(match.group(2))
        }
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": match.group(1).strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    prefix = decode_unicode_escapes(text[: matches[0].start()].strip())
    return (prefix or None), calls


def _parameter_value(value: str) -> Any:
    """The parameter body. A JSON array or object is that value. Other text stays a string."""
    text = value.strip("\n")
    parsed = _json_container(text.strip())
    if parsed is not None:
        return parsed
    return decode_unicode_escapes(text)


def _json_container(text: str) -> Any | None:
    if len(text) < 2 or text[0] not in "[{":
        return None
    if (text[0] == "[" and not text.endswith("]")) or (text[0] == "{" and not text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (list, dict)):
        return parsed
    return None


def function_has_parameters(tools: Any, name: str) -> bool:
    """True unless that function's schema lists no properties and no required names."""
    if not isinstance(tools, list):
        return True
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is None and tool.get("name") == name:
            function = tool
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return False
        properties = parameters.get("properties")
        required = parameters.get("required")
        if isinstance(required, list) and required:
            return True
        if isinstance(properties, dict) and properties:
            return True
        return False
    return True


def required_function(tool_choice: Any) -> str | None:
    """The function named by an OpenAI ``tool_choice`` object."""
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            return name
    name = tool_choice.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn Chat Completions tool calls into the mapping the template walks."""
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        item = dict(message)
        if item.get("content") is None:
            item["content"] = ""
        calls = item.get("tool_calls")
        if isinstance(calls, list):
            item["tool_calls"] = [_normalize_call(call) for call in calls]
        normalized.append(item)
    return normalized


def _normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": function.get("name") or "", "arguments": arguments}


_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"


def _held_suffix(text: str, marker: str) -> int:
    """How many trailing characters of ``text`` are a proper prefix of ``marker``."""
    limit = min(len(text), len(marker) - 1)
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


class StreamText:
    """Reasoning and reply as tokens arrive.

    Thinking stays open until ``</think>``. A reply suffix that may still grow
    into ``<tool_call>`` is held and is not sent as content. The caller parses
    the call into ``tool_calls``. ``keep_calls`` leaves that text in the reply
    when the request asked for no tool message.
    """

    def __init__(self, thinking: bool, *, keep_calls: bool = False) -> None:
        self._reasoning = thinking
        self._keep_calls = keep_calls
        self._held = ""
        self._call = ""
        self._in_call = False
        self._escape_tail = ""
        self._escape_channel: str | None = None

    def push(self, piece: str) -> list[tuple[str, str]]:
        if piece == "":
            return []
        if self._in_call:
            self._call += piece
            return []
        self._held += piece
        if self._reasoning:
            return self._push_reasoning()
        return self._push_reply()

    def finish(self) -> list[tuple[str, str]]:
        if self._in_call:
            xml = self._call
            self._call = ""
            self._in_call = False
            self._held = ""
            flushed = self._flush_escape()
            if self._keep_calls and xml:
                shown = decode_shown_text(xml)
                if shown:
                    flushed.append(("content", shown))
            return flushed
        if not self._held:
            return self._flush_escape()
        channel = "reasoning" if self._reasoning else "content"
        emit = self._held
        self._held = ""
        return self._show(channel, emit, final=True)

    def _show(self, channel: str, text: str, *, final: bool = False) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self._escape_channel not in (None, channel):
            out.extend(self._flush_escape())
        self._escape_channel = channel
        combined = self._escape_tail + text
        if final:
            self._escape_tail = ""
            self._escape_channel = None
            shown = decode_unicode_escapes(combined)
            if shown:
                out.append((channel, shown))
            return out
        body, self._escape_tail = _unicode_tail(combined)
        shown = decode_unicode_escapes(body)
        if shown:
            out.append((channel, shown))
        return out

    def _flush_escape(self) -> list[tuple[str, str]]:
        channel = self._escape_channel
        tail = self._escape_tail
        self._escape_channel = None
        self._escape_tail = ""
        if channel is None or tail == "":
            return []
        shown = decode_unicode_escapes(tail)
        return [(channel, shown)] if shown else []

    def _push_reasoning(self) -> list[tuple[str, str]]:
        at = self._held.find(_THINK_CLOSE)
        if at < 0:
            keep = _held_suffix(self._held, _THINK_CLOSE)
            emit = self._held[:-keep] if keep else self._held
            self._held = self._held[-keep:] if keep else ""
            return self._show("reasoning", emit) if emit else []
        reasoning = self._held[:at]
        rest = self._held[at + len(_THINK_CLOSE) :]
        self._held = ""
        self._reasoning = False
        out = self._show("reasoning", reasoning, final=True) if reasoning else self._flush_escape()
        out.extend(self.push(rest.lstrip("\n")))
        return out

    def _push_reply(self) -> list[tuple[str, str]]:
        at = self._held.find(_TOOL_OPEN)
        if at >= 0:
            emit = self._held[:at]
            self._call = self._held[at:]
            self._held = ""
            self._in_call = True
            if emit:
                return self._show("content", emit, final=True)
            return self._flush_escape()
        keep = _held_suffix(self._held, _TOOL_OPEN)
        if keep == 0:
            emit = self._held
            self._held = ""
            return self._show("content", emit) if emit else []
        emit = self._held[:-keep]
        self._held = self._held[-keep:]
        return self._show("content", emit) if emit else []


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """Reasoning and reply. A closed think block in the prompt leaves only the reply."""
    if not thinking:
        return "", text
    marker = "</think>"
    at = text.find(marker)
    if at < 0:
        return text, ""
    return text[:at], text[at + len(marker) :].lstrip("\n")
