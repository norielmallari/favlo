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


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Leading text, then each complete ``<tool_call>`` block."""
    matches = list(_CALL.finditer(text))
    if not matches:
        return text, []
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
    prefix = text[: matches[0].start()].strip()
    return (prefix or None), calls


def _parameter_value(value: str) -> Any:
    """The parameter body. A JSON array or object is that value. Other text stays a string."""
    text = value.strip("\n")
    parsed = _json_container(text.strip())
    if parsed is not None:
        return parsed
    return text


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
    into ``<tool_call>`` is held until the call is complete, then sent as content
    so a later request can echo that text.
    """

    def __init__(self, thinking: bool) -> None:
        self._reasoning = thinking
        self._held = ""
        self._call = ""
        self._in_call = False

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
            return [("content", xml)] if xml else []
        if not self._held:
            return []
        channel = "reasoning" if self._reasoning else "content"
        emit = self._held
        self._held = ""
        return [(channel, emit)]

    def _push_reasoning(self) -> list[tuple[str, str]]:
        at = self._held.find(_THINK_CLOSE)
        if at < 0:
            keep = _held_suffix(self._held, _THINK_CLOSE)
            emit = self._held[:-keep] if keep else self._held
            self._held = self._held[-keep:] if keep else ""
            return [("reasoning", emit)] if emit else []
        reasoning = self._held[:at]
        rest = self._held[at + len(_THINK_CLOSE) :]
        self._held = ""
        self._reasoning = False
        out: list[tuple[str, str]] = []
        if reasoning:
            out.append(("reasoning", reasoning))
        out.extend(self.push(rest.lstrip("\n")))
        return out

    def _push_reply(self) -> list[tuple[str, str]]:
        at = self._held.find(_TOOL_OPEN)
        if at >= 0:
            emit = self._held[:at]
            self._call = self._held[at:]
            self._held = ""
            self._in_call = True
            return [("content", emit)] if emit else []
        keep = _held_suffix(self._held, _TOOL_OPEN)
        if keep == 0:
            emit = self._held
            self._held = ""
            return [("content", emit)] if emit else []
        emit = self._held[:-keep]
        self._held = self._held[-keep:]
        return [("content", emit)] if emit else []


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """Reasoning and reply. A closed think block in the prompt leaves only the reply."""
    if not thinking:
        return "", text
    marker = "</think>"
    at = text.find(marker)
    if at < 0:
        return text, ""
    return text[:at], text[at + len(marker) :].lstrip("\n")
