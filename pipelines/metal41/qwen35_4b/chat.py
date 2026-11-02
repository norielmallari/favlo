"""Responses generation shell. Chat Completions translates into this path."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, TemplateError
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.responses import (
    RequestError,
    chat_to_responses,
    format_instruction,
    function_tools,
    new_id,
    prepare_request,
    remember,
    response_object,
    response_to_chat,
    template_messages,
    usage_object,
    web_search_context,
)
from qwen35_4b.closed import RequestClosed
from qwen35_4b.vision import prepare_prompt
from qwen35_4b.layout import EOS, IM_END
from qwen35_4b.sampling import CONTEXT_LENGTH, mode_card

STOP = {EOS, IM_END}
_THINK_END = "</think>"
_TOOL_MARK = "<tool_call>"
_TOOL = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_TOKENIZERS: dict[str, Tokenizer] = {}


def load_tokenizer(root: Path) -> Tokenizer:
    key = str(root)
    found = _TOKENIZERS.get(key)
    if found is not None:
        return found
    found = Tokenizer.from_file(str(root / "tokenizer.json"))
    _TOKENIZERS[key] = found
    return found


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    tools: Any,
    enable_thinking: bool,
) -> list[int]:
    template = (root / "chat_template.jinja").read_text(encoding="utf-8")
    env = Environment()
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    rendered = env.from_string(template).render(
        messages=_template_messages(_without_historical_reasoning(messages)),
        tools=tools or None,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        add_vision_id=False,
    )
    return list(load_tokenizer(root).encode(rendered, add_special_tokens=False).ids)


def _without_historical_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop thinking from assistant turns at or before the latest user message.

    The Qwen3.5-4B model card says a multi-turn history keeps the final output
    and leaves the thinking content out. The official template does the same:
    only an assistant turn after that user message keeps ``reasoning_content``.
    """
    merged = _merge_split_reasoning(messages)
    last_user = _last_user_index(merged)
    shaped: list[dict[str, Any]] = []
    for index, message in enumerate(merged):
        if not isinstance(message, dict) or message.get("role") != "assistant" or index > last_user:
            shaped.append(message)
            continue
        copied = dict(message)
        copied.pop("reasoning_content", None)
        content = copied.get("content")
        if isinstance(content, str) and "</think>" in content:
            copied["content"] = content.split("</think>")[-1].lstrip("\n")
        shaped.append(copied)
    return shaped


def _merge_split_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put a reasoning-only assistant turn onto the following assistant reply."""
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        nxt = messages[index + 1] if index + 1 < len(messages) else None
        if _reasoning_only(message) and isinstance(nxt, dict) and nxt.get("role") == "assistant":
            combined = dict(nxt)
            if not isinstance(combined.get("reasoning_content"), str):
                combined["reasoning_content"] = message.get("reasoning_content")
            if message.get("tool_calls") and not combined.get("tool_calls"):
                combined["tool_calls"] = message["tool_calls"]
            merged.append(combined)
            index += 2
            continue
        merged.append(message)
        index += 1
    return merged


def _reasoning_only(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    if not isinstance(message.get("reasoning_content"), str):
        return False
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    return text == ""


def _last_user_index(messages: list[dict[str, Any]]) -> int:
    """Index of the latest user message, matching the template's last query."""
    last = len(messages) - 1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if text.startswith("<tool_response>") and text.endswith("</tool_response>"):
            continue
        return index
    return last


def _attach_chat_reasoning(messages: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put each chat ``reasoning_content`` back in front of that assistant turn."""
    if not isinstance(messages, list):
        return items
    attached: list[dict[str, Any]] = []
    cursor = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("system", "developer"):
            continue
        if role != "assistant":
            if cursor < len(items):
                attached.append(items[cursor])
                cursor += 1
            continue
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            attached.append(
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": reasoning}],
                }
            )
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for _call in calls:
                if cursor < len(items):
                    attached.append(items[cursor])
                    cursor += 1
        if message.get("content"):
            if cursor < len(items):
                attached.append(items[cursor])
                cursor += 1
    attached.extend(items[cursor:])
    return attached


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qwen3.5's template walks ``arguments`` with ``|items``, so keep a mapping."""
    shaped: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        copied = dict(message)
        calls = copied.get("tool_calls")
        if isinstance(calls, list):
            copied["tool_calls"] = [_template_tool_call(call) for call in calls if isinstance(call, dict)]
        shaped.append(copied)
    return shaped


def _template_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments") if isinstance(function, dict) else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "id": call.get("id"),
        "type": "function",
        "function": {
            "name": (function.get("name") if isinstance(function, dict) else "") or "",
            "arguments": arguments,
        },
    }


def encode_prompt(root: Path, text: str, *, thinking: bool = False) -> list[int]:
    return encode_chat(root, [{"role": "user", "content": text}], None, thinking)


def decode_text(root: Path, ids: list[int]) -> str:
    if not ids:
        return ""
    return load_tokenizer(root).decode(ids, skip_special_tokens=True)


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    if not thinking:
        return "", text
    tool_at = text.find(_TOOL_MARK)
    end = text.find(_THINK_END)
    if tool_at >= 0 and (end < 0 or tool_at < end):
        reasoning = text[:tool_at]
        visible = text[tool_at:]
    elif end < 0:
        reasoning, visible = text, ""
    else:
        reasoning = text[:end]
        visible = text[end + len(_THINK_END) :]
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :]
    return reasoning.lstrip("\n"), visible.lstrip("\n")


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Qwen3.5 markup: ``<tool_call><function=name><parameter=k>v</parameter></function></tool_call>``."""
    calls: list[dict[str, Any]] = []
    for match in _TOOL.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, Any] = {}
        for key, value in _PARAM.findall(body):
            arguments[key.strip()] = _coerce(value.strip())
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":"), ensure_ascii=False),
                },
            }
        )
    visible = _TOOL.sub("", text).strip()
    return visible, calls


def _coerce(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class _Hooks(LanguagePipeline):
    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        del selection
        runtime._halted = False
        runtime._stop = reply.stopped

    def client_closed(self, exc: OSError) -> bool:
        return str(exc) == "client closed"


class _ChatCompletions(_Hooks):
    def complete(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        del args
        try:
            translated = chat_to_responses(body)
        except RequestError as exc:
            reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        if isinstance(body.get("reasoning"), dict) and "reasoning" not in translated:
            translated["reasoning"] = body["reasoning"]
        if body.get("enable_thinking") is not None and "reasoning" not in translated:
            translated["reasoning"] = {
                "effort": "medium" if body["enable_thinking"] else "none"
            }
        if "stop" in body and "stop" not in translated:
            translated["stop"] = body["stop"]
        raw_input = translated.get("input")
        if isinstance(raw_input, list):
            translated["input"] = _attach_chat_reasoning(body.get("messages"), raw_input)
        _respond(ChatWire(reply), runtime, translated, selection, state)


class _Responses(_Hooks):
    def complete(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        del args
        _respond(ResponseWire(reply), runtime, body, selection, state)


handle_chat = _ChatCompletions()
handle_responses = _Responses()


class ChatWire:
    """Writes a Response as Chat Completions chunks."""

    responses_wire = True

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model_id = getattr(inner, "model_id", "")
        self._created = int(time.time())
        self._calls = 0

    def stopped(self) -> bool:
        return bool(self._inner.stopped())

    def begin_events(self) -> None:
        self._inner.begin_events()

    def done(self) -> None:
        self._inner.done()

    def json(self, status: int, payload: dict[str, Any]) -> None:
        if payload.get("object") != "response":
            self._inner.json(status, payload)
            return
        extra = payload.pop("_chat_usage", None)
        chat = response_to_chat(payload, None)
        if isinstance(extra, dict):
            usage = chat.setdefault("usage", {})
            usage.update(extra)
        self.model_id = payload.get("model") or self.model_id
        chat["model"] = self.model_id
        self._inner.json(status, chat)

    def event(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "response.created":
            self._send([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
            return
        if kind == "response.reasoning_text.delta":
            delta = payload.get("delta") or ""
            if delta:
                self._send(
                    [
                        {
                            "index": 0,
                            "delta": {"reasoning": delta, "reasoning_content": delta},
                            "finish_reason": None,
                        }
                    ]
                )
            return
        if kind == "response.output_text.delta":
            self._send(
                [
                    {
                        "index": 0,
                        "delta": {"content": payload.get("delta") or ""},
                        "finish_reason": None,
                    }
                ]
            )
            return
        if kind == "response.output_item.done":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                self._send(
                    [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": self._calls,
                                        "id": item.get("call_id"),
                                        "type": "function",
                                        "function": {
                                            "name": item.get("name") or "",
                                            "arguments": item.get("arguments") or "{}",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                )
                self._calls += 1
            return
        if kind == "response.completed":
            response = payload.get("response")
            if not isinstance(response, dict):
                return
            extra = response.pop("_chat_usage", None)
            calls = any(
                isinstance(item, dict) and item.get("type") == "function_call"
                for item in response.get("output") or []
            )
            reason = "tool_calls" if calls else _chat_reason(response)
            self._send([{"index": 0, "delta": {}, "finish_reason": reason}])
            usage = extra if isinstance(extra, dict) else {}
            self._inner.event(_frame(self, [], self._created, usage))

    def _send(self, choices: list[dict[str, Any]]) -> None:
        self._inner.event(_frame(self, choices, self._created))


class ResponseWire:
    responses_wire = True

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model_id = getattr(inner, "model_id", "")

    def stopped(self) -> bool:
        return bool(self._inner.stopped())

    def begin_events(self) -> None:
        self._inner.begin_events()

    def done(self) -> None:
        return

    def json(self, status: int, payload: dict[str, Any]) -> None:
        if isinstance(payload, dict):
            payload.pop("_chat_usage", None)
        self._inner.json(status, payload)

    def event(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "response.completed":
            response = payload.get("response")
            if isinstance(response, dict):
                response.pop("_chat_usage", None)
        self._inner.event(payload)


def _frame(
    wire: ChatWire,
    choices: list[dict[str, Any]],
    created: int,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-favlo",
        "object": "chat.completion.chunk",
        "created": created,
        "model": wire.model_id,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _chat_reason(response: dict[str, Any]) -> str:
    details = response.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return "length"
    if response.get("status") == "incomplete":
        return "length"
    return "stop"


def _request_effort(body: dict[str, Any]) -> str | None:
    """``reasoning.effort`` after the Chat Completions translation.

    ``reasoning_effort`` and ``enable_thinking`` are already mapped onto that
    field. An omitted effort stays omitted so the selected variant can decide.
    """
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str) or not effort.strip():
        return None
    return effort.strip().lower()


def _mtp_body(prepared: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if "favlo:mtp" in prepared:
        body["mtp"] = prepared["favlo:mtp"]
    if "favlo:mtp_depth" in prepared:
        body["mtp_depth"] = prepared["favlo:mtp_depth"]
    if "mtp" in prepared:
        body["mtp"] = prepared["mtp"]
    if "mtp_depth" in prepared:
        body["mtp_depth"] = prepared["mtp_depth"]
    return body


def _mtp_flag(body: dict[str, Any], selection: Any = None) -> bool:
    """Draft verify. A request field wins. Omitted registry row is off."""
    if "mtp" in body:
        flag = body["mtp"]
        if isinstance(flag, bool):
            return flag
        message = "mtp must be true or false"
        raise RuntimeError(message)
    folder = getattr(selection, "folder", None) if selection is not None else None
    if not isinstance(folder, str) or not folder:
        return False
    from favlo.registry import model_mtp

    enabled, _depth = model_mtp(folder)
    return enabled


def _mtp_depth(body: dict[str, Any], selection: Any = None) -> int:
    """Draft length. A request field wins. Omitted registry row is three."""
    if "mtp_depth" in body:
        raw = body["mtp_depth"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            message = "mtp_depth must be a positive integer"
            raise RuntimeError(message)
        return raw
    folder = getattr(selection, "folder", None) if selection is not None else None
    if not isinstance(folder, str) or not folder:
        return 3
    from favlo.registry import model_mtp

    _enabled, depth = model_mtp(folder)
    return depth


def _output_limit(body: dict[str, Any], thinking: bool, prompt_len: int, context: int) -> int:
    """An omitted cap on a thinking turn runs to the last context position."""
    named = body.get("max_output_tokens")
    if named is not None:
        return int(named)
    if thinking:
        return max(1, context - prompt_len)
    return 2048


def _split_reply(text: str, thinking: bool) -> tuple[str, str, list[dict[str, Any]]]:
    reasoning, visible = split_thinking(text, True)
    visible, calls = parse_tool_calls(visible)
    if not thinking:
        if visible:
            reasoning = ""
        else:
            reasoning = ""
            visible = text.replace("<think>", "").replace("</think>", "").strip()
            visible, calls = parse_tool_calls(visible)
    return reasoning, visible, calls


def _respond(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    state: dict[str, Any],
) -> None:
    # prepare_request rejects hosted tools; peel web_search off first.
    incoming = dict(body)
    tools_in = incoming.get("tools")
    search_tools: list[dict[str, Any]] = []
    if isinstance(tools_in, list):
        search_tools = [
            tool
            for tool in tools_in
            if isinstance(tool, dict) and tool.get("type") == "web_search"
        ]
        incoming["tools"] = [
            tool
            for tool in tools_in
            if not (isinstance(tool, dict) and tool.get("type") == "web_search")
        ]
    try:
        prepared = prepare_request(incoming, state)
    except RequestError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    search_body = dict(prepared)
    if search_tools:
        search_body["tools"] = list(prepared.get("tools") or []) + search_tools

    thinking, card = mode_card(selection, _request_effort(prepared))
    card = dict(card)
    if prepared.get("temperature") is not None:
        card["temperature"] = float(prepared["temperature"])
    if prepared.get("top_p") is not None:
        card["top_p"] = float(prepared["top_p"])
    mtp_fields = _mtp_body(prepared)
    try:
        mtp = _mtp_flag(mtp_fields, selection)
        mtp_depth = _mtp_depth(mtp_fields, selection)
    except RuntimeError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    emit_tools = prepared.get("tool_choice") != "none"
    try:
        messages, _saw = template_messages(prepared)
        extra = format_instruction(prepared)
        if extra:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = (messages[0].get("content") or "") + "\n" + extra
            else:
                messages.insert(0, {"role": "system", "content": extra})
        search, note = web_search_context(search_body)
        if note:
            messages.append({"role": "user", "content": note})
        tools = function_tools(prepared.get("tools")) if emit_tools else []
        prompt = encode_chat(runtime.root, messages, tools or None, thinking)
        prompt = prepare_prompt(runtime, messages, prompt)
    except (RequestError, RuntimeError, TemplateError, OSError, ValueError) as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return

    limit = _output_limit(prepared, thinking, len(prompt), int(runtime.max_pos))
    stop_strings = _stop_strings(prepared)
    stream = bool(prepared.get("stream"))
    if stream:
        _stream(
            reply,
            runtime,
            prepared,
            state,
            prompt,
            limit,
            card,
            thinking,
            emit_tools,
            search,
            stop_strings,
            mtp=mtp,
            mtp_depth=mtp_depth,
        )
        return

    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        temperature=float(card["temperature"]),
        top_p=float(card["top_p"]),
        top_k=int(card["top_k"]),
        presence=float(card["presence_penalty"]),
        mtp=mtp,
        mtp_depth=mtp_depth,
    )
    text = decode_text(runtime.root, tokens)
    for stopper in stop_strings:
        at = text.find(stopper)
        if at >= 0:
            text = text[:at]
            break
    hit_limit = len(tokens) >= limit and not any(t in STOP for t in tokens[-1:])
    # More reliable: hit limit when we produced exactly limit tokens.
    hit_limit = len(tokens) >= limit
    reasoning, visible, calls = _split_reply(text, thinking)
    if not emit_tools:
        calls = []
    payload = _payload(
        reply,
        runtime,
        prepared,
        search,
        reasoning,
        visible,
        calls,
        len(prompt),
        len(tokens),
        hit_limit,
    )
    remember(state, prepared, payload)
    reply.json(200, payload)


def _stop_strings(body: dict[str, Any]) -> list[str]:
    raw = body.get("stop")
    if isinstance(raw, str) and raw:
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str) and s]
    return []


def _payload(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    search: dict[str, Any] | None,
    reasoning: str,
    visible: str,
    calls: list[dict[str, Any]],
    prompt_len: int,
    completion_len: int,
    hit_limit: bool,
    *,
    response_id: str | None = None,
    reasoning_id: str | None = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if search is not None:
        output.append(search)
    store = body.get("store", True) is not False
    include = body.get("include")
    encrypt = (not store) or (isinstance(include, list) and "reasoning.encrypted_content" in include)
    if reasoning:
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": reasoning_id or new_id("rs_"),
            "summary": [],
            "status": "completed",
            "content": [{"type": "reasoning_text", "text": reasoning}],
        }
        summary = body.get("reasoning")
        if isinstance(summary, dict) and summary.get("summary"):
            item["summary"] = [{"type": "summary_text", "text": reasoning}]
        if encrypt:
            item["encrypted_content"] = reasoning
        output.append(item)
    for call in calls:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        output.append(
            {
                "type": "function_call",
                "id": new_id("fc_"),
                "call_id": call.get("id") or new_id("call_"),
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )
    if visible or not calls:
        output.append(
            {
                "type": "message",
                "id": new_id("msg_"),
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": visible, "annotations": []}],
            }
        )
    status = "incomplete" if hit_limit and not calls else "completed"
    incomplete = "max_output_tokens" if status == "incomplete" else None
    model = getattr(reply, "model_id", "") or body.get("model") or ""
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    usage = usage_object(prompt_len, completion_len, cached, 0)
    payload = response_object(
        model=model,
        output=output,
        usage=usage,
        body=body,
        status=status,
        incomplete=incomplete,
        response_id=response_id or new_id("resp_"),
    )
    payload["_chat_usage"] = _usage(runtime, prompt_len, completion_len)
    return payload


def _context_length(runtime: Any) -> int:
    """The opening capacity, capped by the registered window."""
    live = getattr(runtime, "max_pos", None)
    if isinstance(live, int) and live > 0:
        return min(live, CONTEXT_LENGTH)
    return CONTEXT_LENGTH


def _usage(runtime: Any, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    prompt_n = max(0, prompt_tokens - cached)
    prefill_s = float(getattr(runtime, "_prefill_seconds", 0.0) or 0.0)
    decode_s = float(getattr(runtime, "_decode_seconds", 0.0) or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "context_length": _context_length(runtime),
        "prompt_tokens_details": {"cached_tokens": cached},
        "timings": {
            "prompt_per_second": 0.0 if prefill_s <= 0 else prompt_n / prefill_s,
            "predicted_per_second": 0.0 if decode_s <= 0 else completion_tokens / decode_s,
            "prompt_n": prompt_n,
            "predicted_n": completion_tokens,
        },
    }


def _stream(
    reply: Any,
    runtime: Any,
    prepared: dict[str, Any],
    state: dict[str, Any],
    prompt: list[int],
    limit: int,
    card: dict[str, float],
    thinking: bool,
    emit_tools: bool,
    search: dict[str, Any] | None,
    stop_strings: list[str],
    *,
    mtp: bool = False,
    mtp_depth: int = 3,
) -> None:
    try:
        reply.begin_events()
    except OSError as exc:
        raise RequestClosed() from exc
    sequence = 0

    def event(payload: dict[str, Any]) -> None:
        nonlocal sequence
        payload["sequence_number"] = sequence
        sequence += 1
        try:
            reply.event(payload)
        except OSError as err:
            raise RequestClosed() from err

    response_id = new_id("resp_")
    reasoning_id = new_id("rs_")
    message_id = new_id("msg_")
    shell = response_object(
        model=getattr(reply, "model_id", "") or prepared.get("model") or "",
        output=[],
        usage=usage_object(0, 0, 0, 0),
        body=prepared,
        status="in_progress",
        incomplete=None,
        response_id=response_id,
    )
    event({"type": "response.created", "response": shell})
    event({"type": "response.in_progress", "response": shell})

    pending: queue.Queue[int | None] = queue.Queue()
    failed: list[BaseException] = []
    produced: list[int] = []
    shown_reason = ""
    shown_content = ""
    opened: set[str] = set()

    def _open(channel: str) -> None:
        if channel in opened:
            return
        opened.add(channel)
        if channel == "reasoning":
            item: dict[str, Any] = {
                "type": "reasoning",
                "id": reasoning_id,
                "summary": [],
                "content": [],
                "status": "in_progress",
            }
            index = 0
        else:
            item = {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
            index = 1
        event({"type": "response.output_item.added", "output_index": index, "item": item})

    def write() -> None:
        nonlocal shown_reason, shown_content
        try:
            while True:
                token = pending.get()
                if token is None:
                    break
                produced.append(int(token))
                text = decode_text(runtime.root, produced)
                if text.endswith("\ufffd"):
                    text = text[:-1]
                for stopper in stop_strings:
                    at = text.find(stopper)
                    if at >= 0:
                        text = text[:at]
                reasoning, visible, _calls = _split_reply(text, thinking)
                if emit_tools:
                    mark = visible.find(_TOOL_MARK)
                    if mark >= 0:
                        visible = visible[:mark]
                if thinking and reasoning.startswith(shown_reason):
                    delta = reasoning[len(shown_reason) :]
                    shown_reason = reasoning
                    if delta:
                        _open("reasoning")
                        event(
                            {
                                "type": "response.reasoning_text.delta",
                                "item_id": reasoning_id,
                                "output_index": 0,
                                "delta": delta,
                            }
                        )
                if visible.startswith(shown_content):
                    delta = visible[len(shown_content) :]
                    shown_content = visible
                    if delta:
                        _open("message")
                        event(
                            {
                                "type": "response.output_text.delta",
                                "item_id": message_id,
                                "output_index": 1,
                                "delta": delta,
                            }
                        )
        except BaseException as exc:
            failed.append(exc)

    writer = threading.Thread(target=write, name="favlo-qwen35-stream", daemon=True)
    writer.start()

    def enqueue(token: int) -> None:
        if failed:
            raise RequestClosed()
        pending.put(int(token))

    try:
        try:
            runtime.generate(
                prompt,
                limit,
                STOP,
                on_token=enqueue,
                temperature=float(card["temperature"]),
                top_p=float(card["top_p"]),
                top_k=int(card["top_k"]),
                presence=float(card["presence_penalty"]),
                mtp=mtp,
                mtp_depth=mtp_depth,
            )
        finally:
            pending.put(None)
            writer.join()
    except RequestClosed:
        if failed:
            raise failed[0] from None
        raise
    if failed:
        raise failed[0]

    text = decode_text(runtime.root, produced)
    for stopper in stop_strings:
        at = text.find(stopper)
        if at >= 0:
            text = text[:at]
            break
    reasoning, visible, calls = _split_reply(text, thinking)
    if not emit_tools:
        calls = []
    hit_limit = len(produced) >= limit
    payload = _payload(
        reply,
        runtime,
        prepared,
        search,
        reasoning,
        visible,
        calls,
        len(prompt),
        len(produced),
        hit_limit,
        response_id=response_id,
        reasoning_id=reasoning_id,
    )
    remember(state, prepared, payload)
    for index, item in enumerate(payload.get("output") or []):
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "reasoning":
            event({"type": "response.output_item.done", "output_index": index, "item": item})
        elif kind == "function_call":
            arguments = item.get("arguments") or "{}"
            opened_call = dict(item)
            opened_call["arguments"] = ""
            event({"type": "response.output_item.added", "output_index": index, "item": opened_call})
            event(
                {
                    "type": "response.function_call_arguments.done",
                    "arguments": arguments,
                    "item_id": item.get("id") or "",
                    "output_index": index,
                }
            )
            event({"type": "response.output_item.done", "output_index": index, "item": item})
        elif kind == "message":
            event({"type": "response.output_item.done", "output_index": index, "item": item})
    event({"type": "response.completed", "response": payload})
    done = getattr(reply, "done", None)
    if callable(done):
        done()
