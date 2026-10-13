"""Tool-call text from the checkpoint chat template, as an OpenAI message."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

_CALL = re.compile(
    r"<tool_call>\s*<function=([^\n>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM = re.compile(r"<parameter=([^\n>]+)>\n?(.*?)</parameter>", re.DOTALL)
_MARKER = "<tool_call>"


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a completion into leading text and complete tool calls.

    A call matches the template form: a function name and one parameter block
    per argument. The argument object is a JSON string, which is what the
    Chat Completions message carries. Incomplete markup stays in the text.
    """
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


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make assistant tool calls renderable by the checkpoint template.

    The template walks `arguments` as a mapping. Chat Completions sends that
    object as a JSON string. Empty content becomes an empty string so the
    template can trim it.
    """
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


class ContentGate:
    """Hold back a completion suffix that may still grow into `<tool_call>`."""

    def __init__(self) -> None:
        self._held = ""
        self._in_call = False

    def push(self, piece: str) -> str:
        if self._in_call or piece == "":
            return ""
        self._held += piece
        at = self._held.find(_MARKER)
        if at >= 0:
            emit = self._held[:at]
            self._held = ""
            self._in_call = True
            return emit
        keep = _marker_prefix(self._held)
        if keep == 0:
            emit = self._held
            self._held = ""
            return emit
        emit = self._held[:-keep]
        self._held = self._held[-keep:]
        return emit

    def finish(self) -> str:
        if self._in_call:
            return ""
        emit = self._held
        self._held = ""
        return emit


def tool_names(tools: Any) -> list[str]:
    """Function names declared on a Chat Completions ``tools`` list."""
    if not isinstance(tools, list):
        return []
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if tool.get("type") == "function" and isinstance(function, dict):
            name = function.get("name")
        else:
            name = tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def required_function(tool_choice: Any) -> str | None:
    """The function name when ``tool_choice`` names one function."""
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def canonical_continuation(
    produced: Sequence[int],
    option: str,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> int | None:
    """The next id of ``encode(decode(produced) + option)`` when it continues ``produced``.

    A forced opening has to be that encoding. A later render of the same text
    is then the same ids, and the prefix cache can extend.
    """
    if not option:
        return None
    prefix = decode(list(produced)) if produced else ""
    ids = list(encode(prefix + option))
    if ids[: len(produced)] != list(produced):
        return None
    if len(ids) == len(produced):
        return None
    return int(ids[len(produced)])


def call_closes(tools: Any, name: str | None) -> bool:
    """True when the named function declares no parameters.

    The forced tokens then include ``</function>`` and ``</tool_call>``.
    """
    if not name or not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return True
        properties = parameters.get("properties")
        return not isinstance(properties, dict) or not properties
    return False


def forced_call_options(text: str, name: str | None, *, close: bool = False) -> list[str] | None:
    """The unwritten opening of a named tool call.

    ``None`` once that opening is complete, or when no function was required.
    The opening is ``<tool_call>`` plus ``<function=name>``. A function with no
    parameters also forces the closing tags.
    """
    if not name:
        return None
    target = f"<tool_call>\n<function={name}>\n"
    if close:
        target += "</function>\n</tool_call>"
    if not target.startswith(text):
        return None
    rest = target[len(text) :]
    if not rest:
        return None
    return [rest]


def function_name_options(text: str, names: Sequence[str]) -> list[str] | None:
    """What an open ``<function=`` may still spell.

    ``None`` when the completion is outside a function name. Each string is the
    unwritten suffix of ``=name>`` for one tool name from the request.
    """
    if not names:
        return None
    start = text.rfind("<tool_call>")
    if start < 0 or "</tool_call>" in text[start:]:
        return None
    mark = text.rfind("<function", start)
    if mark < 0:
        return None
    written = text[mark + len("<function") :]
    if ">" in written:
        return None
    options = []
    for name in names:
        target = f"={name}>"
        if target.startswith(written):
            rest = target[len(written) :]
            if rest:
                options.append(rest)
    if not options:
        return None
    return options


def continuing_ids(options: Sequence[str], pieces: Sequence[str]) -> list[int]:
    """Token ids whose text is a prefix of one remaining function-name suffix."""
    allowed = []
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        for option in options:
            if option.startswith(piece):
                allowed.append(index)
                break
    return allowed


def _parameter_value(value: str) -> str:
    if value.endswith("\n"):
        return value[:-1]
    return value


def _normalize_call(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    item = dict(call)
    function = item.get("function")
    if not isinstance(function, dict):
        return item
    function = dict(function)
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        if arguments == "":
            function["arguments"] = {}
        else:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                message = "tool call arguments must be a JSON object"
                raise RuntimeError(message)
            function["arguments"] = parsed
    item["function"] = function
    return item


def _marker_prefix(text: str) -> int:
    keep = 0
    for size in range(1, len(_MARKER)):
        if text.endswith(_MARKER[:size]):
            keep = size
    return keep
