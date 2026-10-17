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
            name.strip(): value.strip("\n") for name, value in _PARAM.findall(match.group(2))
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
    """Reasoning, reply, and tool calls as tokens arrive.

    Thinking stays open until ``</think>``. A reply suffix that may still grow
    into ``<tool_call>`` is held back. Once the call opens, the function name
    and each finished parameter are emitted as OpenAI tool-call pieces. The
    argument pieces concatenate to one JSON object.
    """

    def __init__(self, thinking: bool) -> None:
        self._reasoning = thinking
        self._held = ""
        self._in_call = False
        self._tool = ""
        self._phase = "seek_fn"
        self._index = 0
        self._json_open = False
        self._param = ""

    def push(self, piece: str) -> list[tuple[str, Any]]:
        if piece == "":
            return []
        if self._in_call:
            return self._push_tool(piece)
        self._held += piece
        if self._reasoning:
            return self._push_reasoning()
        return self._push_reply()

    def finish(self) -> list[tuple[str, Any]]:
        if self._in_call:
            return self._finish_call()
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

    def _push_reply(self) -> list[tuple[str, Any]]:
        at = self._held.find(_TOOL_OPEN)
        if at >= 0:
            emit = self._held[:at]
            rest = self._held[at + len(_TOOL_OPEN) :]
            self._held = ""
            self._in_call = True
            self._tool = ""
            self._phase = "seek_fn"
            out: list[tuple[str, Any]] = [("content", emit)] if emit else []
            out.extend(self._push_tool(rest))
            return out
        keep = _held_suffix(self._held, _TOOL_OPEN)
        if keep == 0:
            emit = self._held
            self._held = ""
            return [("content", emit)] if emit else []
        emit = self._held[:-keep]
        self._held = self._held[-keep:]
        return [("content", emit)] if emit else []

    def _push_tool(self, piece: str) -> list[tuple[str, Any]]:
        self._tool += piece
        out: list[tuple[str, Any]] = []
        while self._in_call:
            if self._phase == "seek_fn":
                match = re.search(r"<function=([^\n>]+)>", self._tool)
                if match is None:
                    break
                name = match.group(1).strip()
                self._tool = self._tool[match.end() :]
                out.append(
                    (
                        "tool_start",
                        {"index": self._index, "id": f"call_{self._index}", "name": name},
                    )
                )
                self._phase = "seek_param"
                continue
            if self._phase == "in_value":
                end = self._tool.find("</parameter>")
                if end < 0:
                    break
                value = self._tool[:end].strip("\n")
                self._tool = self._tool[end + len("</parameter>") :]
                out.append(
                    (
                        "tool_args",
                        {"index": self._index, "arguments": self._param_fragment(self._param, value)},
                    )
                )
                self._phase = "seek_param"
                continue
            param = re.search(r"<parameter=([^\n>]+)>\n?", self._tool)
            close = self._tool.find("</tool_call>")
            param_at = param.start() if param is not None else -1
            if param is not None and (close < 0 or param_at < close):
                self._param = param.group(1).strip()
                self._tool = self._tool[param.end() :]
                self._phase = "in_value"
                continue
            if close < 0:
                break
            out.extend(self._close_call())
        return out

    def _finish_call(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        if self._phase == "in_value" and self._tool:
            out.append(
                (
                    "tool_args",
                    {
                        "index": self._index,
                        "arguments": self._param_fragment(self._param, self._tool.strip("\n")),
                    },
                )
            )
            self._tool = ""
        if self._phase != "seek_fn":
            out.extend(self._close_call())
        self._in_call = False
        self._tool = ""
        return out

    def _close_call(self) -> list[tuple[str, Any]]:
        arguments = "{}" if not self._json_open else "}"
        self._json_open = False
        index = self._index
        self._index += 1
        self._phase = "seek_fn"
        end = self._tool.find("</tool_call>")
        if end >= 0:
            rest = self._tool[end + len("</tool_call>") :]
        else:
            rest = ""
        self._tool = ""
        self._in_call = False
        out: list[tuple[str, Any]] = [("tool_args", {"index": index, "arguments": arguments})]
        if rest.strip():
            out.extend(self.push(rest.lstrip("\n")))
        return out

    def _param_fragment(self, name: str, value: str) -> str:
        body = json.dumps(name) + ": " + json.dumps(value)
        if self._json_open:
            return ", " + body
        self._json_open = True
        return "{" + body


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """Reasoning and the reply after a closed think block."""
    if not thinking:
        return "", text
    marker = "</think>"
    at = text.find(marker)
    if at < 0:
        return text, ""
    return text[:at], text[at + len(marker) :].lstrip("\n")
