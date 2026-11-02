"""Responses request, Response object, and the Chat Completions translation.

Generation reads a Responses body. Chat Completions is translated into that
body and the Response is translated back. A function tool is a schema on the
request. The client runs it.
"""

from __future__ import annotations

import json
import re
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_HOSTED = frozenset(
    {
        "file_search",
        "code_interpreter",
        "computer",
        "computer_use",
        "image_generation",
        "mcp",
        "apply_patch",
        "tool_search",
        "shell",
    }
)
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_RESULT_CAP = {"low": 3, "medium": 5, "high": 8}
_LINK = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span)>', re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


class RequestError(ValueError):
    """The Responses body cannot be run."""


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def chat_to_responses(body: dict[str, Any]) -> dict[str, Any]:
    """One Chat Completions body as a Responses request."""
    count = body.get("n", 1)
    if count not in (None, 1):
        raise RequestError("n greater than 1 is rejected")
    instructions, items = _chat_items(body.get("messages"))
    translated: dict[str, Any] = {
        "model": body.get("model") or "",
        "input": items,
        "stream": bool(body.get("stream", False)),
    }
    if instructions:
        translated["instructions"] = instructions
    effort = _chat_effort(body)
    if effort is not None:
        translated["reasoning"] = {"effort": effort}
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if limit is not None:
        translated["max_output_tokens"] = int(limit)
    if body.get("temperature") is not None:
        translated["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        translated["top_p"] = body["top_p"]
    if "store" in body:
        translated["store"] = body["store"]
    tools = _chat_tools(body.get("tools"))
    search = body.get("web_search_options")
    if isinstance(search, dict):
        tools.append({"type": "web_search", **search})
    if tools:
        translated["tools"] = tools
    choice = _chat_tool_choice(body.get("tool_choice"))
    if choice is not None:
        translated["tool_choice"] = choice
    formatted = _chat_format(body.get("response_format"))
    if formatted is not None:
        translated["text"] = {"format": formatted}
    for name in ("metadata", "user", "service_tier"):
        if name in body:
            translated[name] = body[name]
    for source, target in (
        ("mtp", "favlo:mtp"),
        ("mtp_depth", "favlo:mtp_depth"),
    ):
        if source in body:
            translated[target] = body[source]
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict):
        if "mtp" in kwargs and "favlo:mtp" not in translated:
            translated["favlo:mtp"] = kwargs["mtp"]
        if "mtp_depth" in kwargs and "favlo:mtp_depth" not in translated:
            translated["favlo:mtp_depth"] = kwargs["mtp_depth"]
    return translated


def prepare_request(body: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    """Reject a body this plan does not run, and prepend a stored chain."""
    if _has_audio(body):
        raise RequestError("audio is an invalid request")
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") in _HOSTED:
                raise RequestError(f"{tool.get('type')} is not run")
    limit = body.get("max_output_tokens")
    if limit is not None and int(limit) < 16:
        raise RequestError("max_output_tokens below 16 is rejected")
    effort = _effort(body.get("reasoning"))
    if effort is not None and effort not in _EFFORTS:
        raise RequestError("reasoning.effort is not a listed value")
    if body.get("conversation") and body.get("previous_response_id"):
        raise RequestError("conversation and previous_response_id are not set together")
    prepared = dict(body)
    prepared["input"] = _joined_input(body, state)
    return prepared


def thinking_enabled(body: dict[str, Any]) -> bool | None:
    """True when effort asks for a trace. None when the body omitted effort."""
    effort = _effort(body.get("reasoning"))
    if effort is None:
        return None
    return effort != "none"


def function_tools(tools: Any) -> list[dict[str, Any]]:
    """Function tools in the shape the chat template walks."""
    if not isinstance(tools, list):
        return []
    shaped: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") not in (None, "function"):
            continue
        if tool.get("type") == "web_search":
            continue
        nested = tool.get("function") if isinstance(tool.get("function"), dict) else None
        source = nested or tool
        name = source.get("name")
        if not isinstance(name, str) or not name:
            continue
        shaped.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": source.get("description") or "",
                    "parameters": source.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return shaped


def template_messages(body: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Messages for the checkpoint template, and whether a reasoning item was present.

    A reasoning item and the assistant message or function calls that follow it
    become one assistant turn, so a later prompt that keeps that trace stays a
    prefix of the tokens already stored.
    """
    instructions = body.get("instructions")
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions.strip()})
    raw = body.get("input")
    if isinstance(raw, str):
        messages.append({"role": "user", "content": raw})
        return messages, False
    if not isinstance(raw, list):
        return messages, False
    saw_reasoning = False
    pending: list[dict[str, Any]] = []
    carried = ""

    def flush() -> None:
        nonlocal carried
        if not pending and not carried:
            return
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        if carried:
            message["reasoning_content"] = carried
            carried = ""
        if pending:
            message["tool_calls"] = pending.copy()
            pending.clear()
        messages.append(message)

    def _attach_calls(message: dict[str, Any], calls: list[dict[str, Any]]) -> None:
        if not calls:
            return
        existing = message.get("tool_calls")
        merged = list(calls)
        if isinstance(existing, list) and existing:
            merged.extend(existing)
        message["tool_calls"] = merged

    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            saw_reasoning = True
            text = _reasoning_text(item)
            if text:
                carried = text
            continue
        if kind == "function_call":
            call = _call_message(item)
            last = messages[-1] if messages else None
            if (
                isinstance(last, dict)
                and last.get("role") == "assistant"
                and not carried
                and not pending
            ):
                _attach_calls(last, [call])
            else:
                pending.append(call)
            continue
        if kind == "function_call_output":
            flush()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": _output_text(item.get("output")),
                }
            )
            continue
        if kind in ("message", None) and item.get("role"):
            message = _message_item(item)
            if message.get("role") == "assistant":
                if carried:
                    if not isinstance(message.get("reasoning_content"), str):
                        message["reasoning_content"] = carried
                    carried = ""
                if pending:
                    _attach_calls(message, pending)
                    pending.clear()
                messages.append(message)
                continue
            flush()
            messages.append(message)
            continue
        if kind == "input_file":
            flush()
            messages.append({"role": "user", "content": _file_text(item)})
    flush()
    return messages, saw_reasoning


def web_search_context(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """One DuckDuckGo search, or no search when the request has no web_search tool."""
    tool = _web_tool(body.get("tools"))
    if tool is None:
        return None, ""
    query = _latest_user_text(body.get("input"))
    location = tool.get("user_location")
    if isinstance(location, dict):
        hint = " ".join(
            str(location.get(name) or "")
            for name in ("city", "region", "country")
            if location.get(name)
        ).strip()
        if hint:
            query = f"{query} {hint}".strip()
    size = tool.get("search_context_size") or "medium"
    if size not in _RESULT_CAP:
        size = "medium"
    hits = search_duckduckgo(query, _RESULT_CAP[str(size)])
    lines = [f"{hit['title']} {hit['url']}\n{hit['snippet']}".strip() for hit in hits]
    note = "\n\n".join(lines)
    call = {
        "type": "web_search_call",
        "id": new_id("ws_"),
        "status": "completed",
        "action": {
            "type": "search",
            "query": query,
            "queries": [query] if query else [],
            "sources": [{"type": "url", "url": hit["url"]} for hit in hits if hit.get("url")],
        },
    }
    return call, note


def search_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    """Titles, URLs, and snippets from the DuckDuckGo HTML results."""
    if not query.strip() or limit < 1:
        return []
    target = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(target, headers={"User-Agent": "favlo"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            page = response.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return []
    links = _LINK.findall(page)
    snippets = [_plain(text) for text in _SNIPPET.findall(page)]
    hits: list[dict[str, str]] = []
    for index, (href, title) in enumerate(links):
        if len(hits) >= limit:
            break
        url = _result_url(href)
        if not url:
            continue
        hits.append(
            {
                "title": _plain(title) or url,
                "url": url,
                "snippet": snippets[index] if index < len(snippets) else "",
            }
        )
    return hits


def citations(hits: list[dict[str, str]], text: str) -> list[dict[str, Any]]:
    """url_citation annotations for results named in the reply."""
    notes: list[dict[str, Any]] = []
    for hit in hits:
        url = hit.get("url") or ""
        if not url or url not in text:
            continue
        start = text.index(url)
        notes.append(
            {
                "type": "url_citation",
                "url": url,
                "title": hit.get("title") or url,
                "start_index": start,
                "end_index": start + len(url),
            }
        )
    return notes


def response_object(
    *,
    model: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any],
    body: dict[str, Any],
    status: str,
    incomplete: str | None,
    response_id: str,
) -> dict[str, Any]:
    """One Response the OpenAI client accepts."""
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": time.time(),
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": body.get("tool_choice") or "auto",
        "tools": [tool for tool in body.get("tools") or [] if isinstance(tool, dict)],
        "usage": usage,
    }
    if body.get("instructions"):
        payload["instructions"] = body["instructions"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("metadata") is not None:
        payload["metadata"] = body["metadata"]
    if body.get("previous_response_id"):
        payload["previous_response_id"] = body["previous_response_id"]
    if incomplete:
        payload["incomplete_details"] = {"reason": incomplete}
    if status == "completed":
        payload["completed_at"] = time.time()
    return payload


def usage_object(prompt: int, completion: int, cached: int, reasoning_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_tokens_details": {
            "cached_tokens": cached,
            "cache_write_tokens": 0,
        },
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def remember(state: dict[str, Any] | None, body: dict[str, Any], response: dict[str, Any]) -> None:
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
        container = bucket.setdefault(conversation, {"id": conversation, "object": "conversation", "items": []})
        container["items"].extend(_input_list(body.get("input")))
        container["items"].extend(response.get("output") or [])


def open_conversation(state: dict[str, Any]) -> dict[str, Any]:
    identifier = new_id("conv_")
    container = {"id": identifier, "object": "conversation", "items": []}
    state.setdefault("conversations", {})[identifier] = container
    return container


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
    return True


def response_to_chat(payload: dict[str, Any], timings: dict[str, Any] | None = None) -> dict[str, Any]:
    """A Response as one Chat Completions object."""
    text, reasoning, calls = _read_output(payload.get("output") or [])
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    reason = _chat_finish(payload, bool(calls))
    usage = _chat_usage(payload.get("usage"), timings)
    return {
        "id": "chatcmpl-favlo",
        "object": "chat.completion",
        "created": int(payload.get("created_at") or time.time()),
        "model": payload.get("model") or "",
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": usage,
    }


def _chat_finish(payload: dict[str, Any], calls: bool) -> str:
    if calls:
        return "tool_calls"
    details = payload.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return "length"
    if payload.get("status") == "incomplete":
        return "length"
    return "stop"


def _chat_usage(usage: Any, timings: dict[str, Any] | None) -> dict[str, Any]:
    source = usage if isinstance(usage, dict) else {}
    prompt = int(source.get("input_tokens") or 0)
    completion = int(source.get("output_tokens") or 0)
    details = source.get("input_tokens_details") if isinstance(source.get("input_tokens_details"), dict) else {}
    chat: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": int(details.get("cached_tokens") or 0)},
    }
    if timings:
        chat["timings"] = timings
        if timings.get("context_length"):
            chat["context_length"] = timings["context_length"]
    return chat


def _joined_input(body: dict[str, Any], state: dict[str, Any] | None) -> Any:
    fresh = body.get("input")
    previous = body.get("previous_response_id")
    conversation = body.get("conversation")
    stored: list[Any] = []
    if isinstance(previous, str) and previous:
        if not isinstance(state, dict):
            raise RequestError("previous_response_id is not stored")
        found = stored_response(state, previous)
        if found is None:
            raise RequestError("previous_response_id is not stored")
        stored.extend(_stored_input(state, previous))
        stored.extend(found.get("output") or [])
    if isinstance(conversation, str) and conversation:
        if not isinstance(state, dict):
            raise RequestError("conversation is not stored")
        bucket = state.get("conversations")
        container = bucket.get(conversation) if isinstance(bucket, dict) else None
        if not isinstance(container, dict):
            raise RequestError("conversation is not stored")
        stored = list(container.get("items") or []) + stored
    if not stored:
        return fresh
    return [*stored, *_input_list(fresh)]


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


def _chat_items(messages: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(messages, list):
        raise RequestError("messages must be a list")
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("system", "developer"):
            text = _content_text(message.get("content"))
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": _content_text(message.get("content")),
                }
            )
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or new_id("fc_"),
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            content = message.get("content")
            if content:
                items.append({"type": "message", "role": "assistant", "content": content})
            continue
        items.append({"type": "message", "role": role or "user", "content": _content_parts(message.get("content"))})
    return "\n\n".join(instructions), items


def _content_parts(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append(part)
            continue
        if part.get("type") == "image_url" or "image_url" in part:
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
            continue
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": part.get("text") or ""})
            continue
        parts.append(part)
    return parts


def _chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    shaped: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is None:
            shaped.append(tool)
            continue
        shaped.append(
            {
                "type": "function",
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return shaped


def _chat_tool_choice(choice: Any) -> Any:
    if choice is None:
        return None
    if choice in ("none", "auto", "required"):
        return choice
    if isinstance(choice, dict):
        function = choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "name": function["name"]}
        if choice.get("name"):
            return {"type": "function", "name": choice["name"]}
    return choice


def _chat_format(fmt: Any) -> dict[str, Any] | None:
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": fmt.get("json_schema") or fmt}
    return None


def _chat_effort(body: dict[str, Any]) -> str | None:
    named = body.get("reasoning_effort")
    if isinstance(named, str) and named:
        return named
    flag = body.get("enable_thinking")
    kwargs = body.get("chat_template_kwargs")
    if flag is None and isinstance(kwargs, dict):
        flag = kwargs.get("enable_thinking")
    if flag is None:
        return None
    return "medium" if bool(flag) else "none"


def _effort(reasoning: Any) -> str | None:
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str) or not effort:
        return None
    return effort


def _has_audio(body: dict[str, Any]) -> bool:
    raw = body.get("input")
    if not isinstance(raw, list):
        return False
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("input_audio", "audio"):
            return True
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_audio", "audio"):
                return True
    return False


def _message_item(item: dict[str, Any]) -> dict[str, Any]:
    role = item.get("role") or "user"
    content = item.get("content")
    if isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("input_text", "output_text"):
                parts.append({"type": "text", "text": part.get("text") or ""})
            elif part.get("type") == "input_image":
                url = part.get("image_url") or part.get("url")
                parts.append({"type": "image_url", "image_url": {"url": url, "detail": part.get("detail") or "auto"}})
            elif part.get("type") == "text" or "image_url" in part:
                parts.append(part)
        return {"role": role, "content": parts or ""}
    return {"role": role, "content": content if isinstance(content, str) else _content_text(content)}


def _call_message(item: dict[str, Any]) -> dict[str, Any]:
    arguments = item.get("arguments") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": item.get("call_id") or item.get("id") or new_id("fc_"),
        "type": "function",
        "function": {
            "name": item.get("name") or "",
            "arguments": arguments,
        },
    }


def _reasoning_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, list):
        parts = [part.get("text") or "" for part in content if isinstance(part, dict)]
        text = "".join(parts).strip()
        if text:
            return text
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str):
        return encrypted
    return ""


def _output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(_content_text(part) for part in output)
    return _content_text(output)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        return "".join(_content_text(part) for part in content)
    return ""


def _file_text(item: dict[str, Any]) -> str:
    name = str(item.get("filename") or "")
    media = str(item.get("media_type") or item.get("mime_type") or "")
    data = item.get("file_data") or item.get("text") or ""
    if not isinstance(data, str):
        raise RequestError("input_file is an invalid request")
    textual = media.startswith("text/") or name.endswith(".txt") or name.endswith(".md") or media == ""
    if not textual:
        raise RequestError("input_file is an invalid request")
    if item.get("file_data") and not data.startswith("data:"):
        import base64

        try:
            return base64.b64decode(data).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RequestError("input_file is an invalid request") from exc
    return data


def _web_tool(tools: Any) -> dict[str, Any] | None:
    if not isinstance(tools, list):
        return None
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "web_search":
            return tool
    return None


def _latest_user_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return ""
    text = ""
    for item in raw:
        if isinstance(item, dict) and item.get("role") == "user":
            text = _content_text(item.get("content")) or text
        elif isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user":
            text = _content_text(item.get("content")) or text
    return text


def _plain(text: str) -> str:
    return _TAG.sub("", text).replace("&amp;", "&").replace("&quot;", '"').strip()


def _result_url(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    query = urllib.parse.parse_qs(parsed.query)
    target = query.get("uddg", [href])[0]
    if target.startswith("//"):
        target = "https:" + target
    if not target.startswith("http"):
        return ""
    return target


def _read_output(output: list[Any]) -> tuple[str | None, str, list[dict[str, Any]]]:
    text = ""
    reasoning = ""
    calls: list[dict[str, Any]] = []
    saw_text = False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            reasoning = _reasoning_text(item)
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text += part.get("text") or ""
                    saw_text = True
        elif item.get("type") == "function_call":
            calls.append(
                {
                    "id": item.get("call_id") or "",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
    if calls and not text:
        return None, reasoning, calls
    return (text if saw_text else None), reasoning, calls


def format_instruction(body: dict[str, Any]) -> str:
    """Extra instruction for a text.format the template has no slot for."""
    text = body.get("text")
    if not isinstance(text, dict):
        return ""
    formatted = text.get("format")
    if not isinstance(formatted, dict):
        return ""
    if formatted.get("type") == "json_object":
        return "Reply with a JSON object."
    if formatted.get("type") == "json_schema":
        return "Reply with JSON matching this schema:\n" + json.dumps(formatted.get("json_schema") or formatted)
    return ""
