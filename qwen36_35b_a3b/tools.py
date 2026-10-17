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


def _parameter_value(value: str) -> str:
    text = value.strip("\n")
    return text


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


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """Reasoning and reply. A closed think block in the prompt leaves only the reply."""
    if not thinking:
        return "", text
    marker = "</think>"
    at = text.find(marker)
    if at < 0:
        return text, ""
    return text[:at], text[at + len(marker) :].lstrip("\n")
