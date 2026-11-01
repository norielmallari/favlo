"""Chat template, markup, and one Responses generation.

Chat Completions is translated into a Responses body. This module reads that
body and writes a Response. The template's levels are none, low, and high.
"""

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
    citations,
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
from spark_x26_1_7b.closed import RequestClosed
from spark_x26_1_7b.layout import BOS, EOS, PAD
from spark_x26_1_7b.sampling import DEFAULT_VARIANT, VARIANTS

# Start-of-sentence opens the next turn. End-of-sentence closes this one.
# Pad is not message text. None of these ids is copied into the reply.
STOP = {BOS, EOS, PAD}
_TOOL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_THINK_END = "</think>"
_TOOL_MARK = "<tool_call>"
_LOW = frozenset({"minimal", "low"})
_HIGH = frozenset({"medium", "high", "xhigh", "max"})


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
    reasoning_effort: str | None = None,
) -> list[int]:
    template = (root / "chat_template.jinja").read_text(encoding="utf-8")
    env = Environment()
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    rendered_args: dict[str, Any] = {
        "messages": _normalize(messages),
        "tools": tools or None,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if reasoning_effort:
        rendered_args["reasoning_effort"] = reasoning_effort
    rendered = env.from_string(template).render(**rendered_args)
    encoded = load_tokenizer(root).encode(rendered, add_special_tokens=False)
    return list(encoded.ids)


def decode_text(root: Path, ids: list[int]) -> str:
    if not ids:
        return ""
    return load_tokenizer(root).decode(ids, skip_special_tokens=False)


def split_thinking(text: str, thinking: bool) -> tuple[str, str]:
    """Trace stays out of the reply. A tool call before ``</think>`` ends the trace."""
    tool_at = text.find(_TOOL_MARK)
    end = text.find(_THINK_END)
    if tool_at >= 0 and (end < 0 or tool_at < end):
        reasoning = text[:tool_at]
        visible = text[tool_at:]
    elif end >= 0:
        reasoning = text[:end]
        visible = text[end + len(_THINK_END) :]
    elif thinking:
        reasoning = text
        visible = ""
    else:
        return "", text
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :]
    return reasoning, visible


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for match in _TOOL.finditer(text):
        body = match.group(1)
        name_at = body.find("<arg_key>")
        name = body if name_at < 0 else body[:name_at]
        arguments: dict[str, Any] = {}
        for key, value in _ARG.findall(body):
            arguments[key] = _coerce(value)
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": json.dumps(arguments, separators=(",", ":"), ensure_ascii=False),
                },
            }
        )
    visible = _TOOL.sub("", text).strip()
    return visible, calls


class _Hooks(LanguagePipeline):
    """Disconnect shell shared by the Responses route and the chat adapter."""

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        del selection
        runtime._halted = False
        runtime._stop = reply.stopped

    def client_closed(self, exc: OSError) -> bool:
        return str(exc) == "client closed"


class Pipeline(_Hooks):
    """Chat Completions translated into one Responses generation."""

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
            _reject(reply, str(exc))
            return
        _respond(ChatWire(reply), runtime, translated, selection, state)


class ResponsesPipeline(_Hooks):
    """One Responses generation."""

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


handle_chat = Pipeline()
handle_responses = ResponsesPipeline()


def _reject(reply: Any, message: str) -> None:
    reply.json(400, {"error": {"message": message, "type": "invalid_request_error"}})


def _respond(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    state: dict[str, Any] | None,
) -> None:
    try:
        prepared = prepare_request(body, state)
    except RequestError as exc:
        _reject(reply, str(exc))
        return
    thinking, effort, card = _mode(selection, prepared)
    try:
        temperature, top_p, top_k, presence, frequency = _sampling(prepared, card)
    except RequestError as exc:
        _reject(reply, str(exc))
        return
    emit_tools = prepared.get("tool_choice") != "none"
    required = _required_function(prepared.get("tool_choice")) if emit_tools else None
    if required:
        thinking = False
        effort = None
    try:
        messages, _saw = template_messages(prepared)
        _reject_images(messages)
        _merge_instruction(messages, format_instruction(prepared))
        if prepared.get("tool_choice") == "required" and not required:
            _merge_instruction(messages, "Call one of the functions.")
        search, note = web_search_context(prepared)
        if note:
            messages.append({"role": "user", "content": note})
        tools = function_tools(prepared.get("tools"))
        prompt = _fit_prompt(runtime, prepared, messages, tools, thinking, effort)
        forced, stop_after = _forced_call(runtime.root, tools, required)
    except (RequestError, RuntimeError, TemplateError, OSError, ValueError) as exc:
        _reject(reply, str(exc))
        return
    cap = prepared.get("max_output_tokens")
    limit = int(cap) if cap is not None else None
    stream = bool(prepared.get("stream", False))
    if stream:
        _stream(
            reply,
            runtime,
            prepared,
            state,
            prompt,
            forced,
            stop_after,
            limit,
            temperature,
            top_p,
            top_k,
            presence,
            frequency,
            thinking,
            emit_tools,
            search,
            note,
        )
        return
    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        forced=forced,
        stop_after_forced=stop_after,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        frequency=frequency,
    )
    text = decode_text(runtime.root, tokens)
    payload = _response_payload(
        reply,
        runtime,
        prepared,
        search,
        note,
        text,
        thinking,
        emit_tools,
        len(prompt),
        len(tokens),
    )
    remember(state, prepared, payload)
    reply.json(200, payload)


def _mode(selection: Any, body: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    """Template level and the sampler card.

    ``reasoning.effort`` replaces a variant on the id. ``none`` turns thinking
    off. ``minimal`` and ``low`` use the low instruction. The other listed
    efforts use the high instruction. An omitted effort uses the id's variant,
    or ``high``.
    """
    named = _listed_effort(body.get("reasoning"))
    variant = getattr(selection, "variant", None) if selection is not None else None
    if named is not None:
        name = named
    elif isinstance(variant, str) and variant in VARIANTS:
        name = variant
    else:
        default = getattr(selection, "default_variant", None) if selection is not None else None
        name = default if isinstance(default, str) and default in VARIANTS else DEFAULT_VARIANT
    if name == variant:
        stored = getattr(selection, "sampling", None) if selection is not None else None
        card = dict(stored) if isinstance(stored, dict) and stored else dict(VARIANTS[name])
    else:
        cards = getattr(selection, "variants", None) if selection is not None else None
        source = cards if isinstance(cards, dict) else VARIANTS
        row = source.get(name) if isinstance(source, dict) else None
        card = dict(row) if isinstance(row, dict) else dict(VARIANTS[name])
    thinking = name != "none"
    return thinking, (name if thinking else None), card


def _listed_effort(reasoning: Any) -> str | None:
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str) or not effort:
        return None
    if effort == "none":
        return "none"
    if effort in _LOW:
        return "low"
    if effort in _HIGH:
        return "high"
    return None


def _sampling(
    body: dict[str, Any], card: dict[str, Any]
) -> tuple[float, float, int, float, float]:
    """Temperature and top-p come from the body. The card fills the rest."""
    temperature = _number(body, "temperature", float(card["temperature"]))
    top_p = _number(body, "top_p", float(card["top_p"]))
    top_k = int(card.get("top_k", 0))
    if top_k < 0:
        top_k = 0
    presence = float(card.get("presence_penalty", 0.0))
    frequency = float(card.get("frequency_penalty", 0.0))
    return temperature, top_p, top_k, presence, frequency


def _number(body: dict[str, Any], name: str, default: float) -> float:
    if body.get(name) is None:
        return default
    value = body.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{name} must be a number"
        raise RequestError(message)
    return float(value)


def _reject_images(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                raise RequestError("input_image is an invalid request")


def _merge_instruction(messages: list[dict[str, Any]], extra: str) -> None:
    if not extra:
        return
    first = messages[0] if messages else None
    content = first.get("content") if isinstance(first, dict) else None
    if isinstance(first, dict) and first.get("role") == "system" and isinstance(content, str):
        first["content"] = extra + "\n\n" + content
        return
    messages.insert(0, {"role": "system", "content": extra})


def _fit_prompt(
    runtime: Any,
    prepared: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: bool,
    effort: str | None,
) -> list[int]:
    prompt = encode_chat(runtime.root, messages, tools, thinking, effort)
    ceiling = getattr(runtime, "limit", None)
    if not isinstance(ceiling, int) or ceiling < 1 or len(prompt) <= ceiling:
        return prompt
    if prepared.get("truncation") != "auto":
        raise RequestError("input does not fit")
    while len(prompt) > ceiling and len(messages) > 1:
        drop = 1 if messages and messages[0].get("role") == "system" else 0
        if drop >= len(messages):
            break
        del messages[drop]
        prompt = encode_chat(runtime.root, messages, tools, thinking, effort)
    if len(prompt) > ceiling:
        raise RequestError("input does not fit")
    return prompt


def _required_function(choice: Any) -> str | None:
    if not isinstance(choice, dict) or choice.get("type") != "function":
        return None
    function = choice.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"]:
        return str(function["name"])
    if isinstance(choice.get("name"), str) and choice["name"]:
        return str(choice["name"])
    return None


def _needs_arguments(tools: list[dict[str, Any]], name: str) -> bool:
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
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


def _forced_call(
    root: Path, tools: list[dict[str, Any]], required: str | None
) -> tuple[list[int], bool]:
    """Opening of a named call, or the whole call when that function takes no arguments."""
    if not required:
        return [], False
    if _needs_arguments(tools, required):
        text = f"<tool_call>{required}"
        stop_after = False
    else:
        text = f"<tool_call>{required}</tool_call>"
        stop_after = True
    encoded = load_tokenizer(root).encode(text, add_special_tokens=False)
    return list(encoded.ids), stop_after


def _fields(text: str, thinking: bool) -> tuple[str, str, list[dict[str, Any]]]:
    marked = "<think>" in text or _THINK_END in text
    reasoning, visible = split_thinking(text, thinking or marked)
    open_at = visible.rfind(_TOOL_MARK)
    close_at = visible.rfind("</tool_call>")
    if open_at > close_at:
        visible = visible[:open_at]
    content, calls = parse_tool_calls(visible)
    return reasoning.strip(), content, calls


def _search_hits(note: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for block in note.split("\n\n"):
        if not block.strip():
            continue
        head, _, snippet = block.partition("\n")
        url = ""
        title = head
        for token in head.split(" "):
            if token.startswith("http://") or token.startswith("https://"):
                url = token
                title = head[: head.rfind(token)].strip()
        if url:
            hits.append({"title": title or url, "url": url, "snippet": snippet})
    return hits


def _response_payload(
    reply: Any,
    runtime: Any,
    prepared: dict[str, Any],
    search: dict[str, Any] | None,
    note: str,
    text: str,
    thinking: bool,
    emit_tools: bool,
    prompt_tokens: int,
    completion_tokens: int,
    response_id: str | None = None,
) -> dict[str, Any]:
    reasoning, content, calls = _fields(text, thinking)
    if not emit_tools:
        calls = []
    output: list[dict[str, Any]] = []
    if search is not None:
        output.append(search)
    store = prepared.get("store", True) is not False
    include = prepared.get("include")
    encrypt = (not store) or (
        isinstance(include, list) and "reasoning.encrypted_content" in include
    )
    if reasoning:
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": new_id("rs_"),
            "summary": [],
            "status": "completed",
            "content": [{"type": "reasoning_text", "text": reasoning}],
        }
        summary = prepared.get("reasoning")
        if isinstance(summary, dict) and summary.get("summary"):
            item["summary"] = [{"type": "summary_text", "text": reasoning}]
        if encrypt:
            item["encrypted_content"] = reasoning
        output.append(item)
    visible = content or ""
    annotations = citations(_search_hits(note), visible) if note else []
    logprobs = isinstance(include, list) and "message.output_text.logprobs" in include
    if calls:
        if visible:
            output.append(_message_item(visible, annotations, logprobs))
        for call in calls:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            output.append(
                {
                    "type": "function_call",
                    "id": new_id("fc_"),
                    "call_id": str(call.get("id") or new_id("call_")),
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "{}",
                    "status": "completed",
                }
            )
        status = "completed"
        incomplete = None
    else:
        output.append(_message_item(visible, annotations, logprobs))
        if getattr(runtime, "_hit_limit", False):
            status = "incomplete"
            incomplete = "max_output_tokens"
        else:
            status = "completed"
            incomplete = None
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    reasoning_tokens = _count_text(getattr(runtime, "root", None), reasoning)
    payload = response_object(
        model=getattr(reply, "model_id", "") or prepared.get("model") or "",
        output=output,
        usage=usage_object(prompt_tokens, completion_tokens, cached, reasoning_tokens),
        body=prepared,
        status=status,
        incomplete=incomplete,
        response_id=response_id or new_id("resp_"),
    )
    payload["_chat_usage"] = _usage(runtime, prompt_tokens, completion_tokens, reasoning_tokens)
    return payload


def _count_text(root: Any, text: str) -> int:
    if not text or not isinstance(root, Path):
        return 0
    encoded = load_tokenizer(root).encode(text, add_special_tokens=False)
    return len(encoded.ids)


def _message_item(text: str, annotations: list[dict[str, Any]], logprobs: bool) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "output_text", "text": text, "annotations": annotations}
    if logprobs:
        part["logprobs"] = []
    return {
        "type": "message",
        "id": new_id("msg_"),
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }


def _usage(
    runtime: Any, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int
) -> dict[str, Any]:
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "context_length": int(getattr(runtime, "limit", 0) or 0),
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


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
        _copy_reasoning(chat)
        if isinstance(extra, dict):
            chat["usage"] = extra
        self.model_id = payload.get("model") or self.model_id
        chat["model"] = self.model_id
        self._inner.json(status, chat)

    def event(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "response.created":
            self._send([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
            return
        if kind == "response.reasoning_text.delta":
            visible = payload.get("delta") or ""
            if str(visible).strip():
                self._send(
                    [
                        {
                            "index": 0,
                            "delta": {"reasoning": visible, "reasoning_content": visible},
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
    """Writes Response events and omits the Chat Completions done marker."""

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


def _copy_reasoning(chat: dict[str, Any]) -> None:
    choices = chat.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return
    trace = message.get("reasoning_content")
    if isinstance(trace, str) and trace.strip():
        message["reasoning"] = trace


def _chat_reason(response: dict[str, Any]) -> str:
    details = response.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return "length"
    if response.get("status") == "incomplete":
        return "length"
    return "stop"


def _stream(
    reply: Any,
    runtime: Any,
    prepared: dict[str, Any],
    state: dict[str, Any] | None,
    prompt: list[int],
    forced: list[int],
    stop_after: bool,
    limit: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    frequency: float,
    thinking: bool,
    emit_tools: bool,
    search: dict[str, Any] | None,
    note: str,
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
        except OSError as exc:
            raise RequestClosed() from exc

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
    opened: set[str] = set()
    shown_reason = ""
    shown_content = ""

    def emit_piece(channel: str, visible: str) -> None:
        if not visible:
            return
        if channel == "reasoning" and not visible.strip():
            return
        if channel not in opened:
            opened.add(channel)
            _open_item(event, channel, reasoning_id, message_id)
        _emit_response(event, channel, visible, reasoning_id, message_id)

    def publish(text: str) -> None:
        nonlocal shown_reason, shown_content
        reasoning, content, _calls = _fields(text, thinking)
        emit_piece("reasoning", _advance(shown_reason, reasoning))
        shown_reason = reasoning
        emit_piece("content", _advance(shown_content, content))
        shown_content = content

    pending: queue.Queue[int | None] = queue.Queue()
    failed: list[BaseException] = []
    text = ""

    def write() -> None:
        nonlocal text
        produced: list[int] = []
        try:
            while True:
                token = pending.get()
                if token is None:
                    break
                produced.append(int(token))
                decoded = decode_text(runtime.root, produced)
                if decoded.endswith("\ufffd"):
                    decoded = decoded[:-1]
                publish(decoded)
            decoded = decode_text(runtime.root, produced)
            if decoded.endswith("\ufffd"):
                decoded = decoded[:-1]
            publish(decoded)
            text = decoded
        except BaseException as exc:
            failed.append(exc)

    writer = threading.Thread(target=write, name="favlo-spark-stream", daemon=True)
    writer.start()

    def enqueue(token: int) -> None:
        if failed:
            raise RequestClosed()
        pending.put(int(token))

    tokens: list[int] = []
    try:
        try:
            tokens = runtime.generate(
                prompt,
                limit,
                STOP,
                forced=forced,
                stop_after_forced=stop_after,
                on_token=enqueue,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence=presence,
                frequency=frequency,
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
    payload = _response_payload(
        reply,
        runtime,
        prepared,
        search,
        note,
        text,
        thinking,
        emit_tools,
        len(prompt),
        len(tokens),
        response_id,
    )
    for index, item in enumerate(payload["output"]):
        kind = item.get("type")
        if kind == "reasoning":
            item["id"] = reasoning_id
            if "reasoning" not in opened:
                _open_item(event, "reasoning", reasoning_id, message_id)
            event({"type": "response.output_item.done", "output_index": index, "item": item})
        elif kind == "function_call":
            arguments = item.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            opened_item = dict(item)
            opened_item["arguments"] = ""
            event(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": opened_item,
                }
            )
            event(
                {
                    "type": "response.function_call_arguments.done",
                    "arguments": arguments,
                    "item_id": item.get("id") or "",
                    "output_index": index,
                }
            )
            event({"type": "response.output_item.done", "output_index": index, "item": item})
    event({"type": "response.completed", "response": payload})
    remember(state, prepared, payload)
    try:
        reply.done()
    except OSError as exc:
        raise RequestClosed() from exc


def _advance(shown: str, content: str) -> str:
    if content.startswith(shown):
        return content[len(shown) :]
    stripped = shown.rstrip()
    if stripped and content.startswith(stripped):
        return content[len(stripped) :]
    return ""


def _open_item(event: Any, channel: str, reasoning_id: str, message_id: str) -> None:
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
            "content": [
                {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
            ],
        }
        index = 1
    event({"type": "response.output_item.added", "output_index": index, "item": item})


def _emit_response(
    event: Any, channel: str, visible: str, reasoning_id: str, message_id: str
) -> None:
    if channel == "reasoning":
        event(
            {
                "type": "response.reasoning_text.delta",
                "content_index": 0,
                "delta": visible,
                "item_id": reasoning_id,
                "output_index": 0,
            }
        )
        return
    event(
        {
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": visible,
            "item_id": message_id,
            "output_index": 1,
            "logprobs": [],
        }
    )


def _frame(
    reply: Any,
    choices: list[dict[str, Any]],
    created: int,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-favlo",
        "object": "chat.completion.chunk",
        "created": created,
        "model": reply.model_id,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _normalize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        calls = item.get("tool_calls")
        if isinstance(calls, list):
            fixed = []
            for call in calls:
                copied = dict(call)
                function = dict(copied.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments) if arguments else {}
                copied["function"] = function
                fixed.append(copied)
            item["tool_calls"] = fixed
        prepared.append(item)
    return prepared


def _tojson(value: Any, separators: tuple[str, str] = (",", ":")) -> str:
    return json.dumps(value, separators=separators, ensure_ascii=False)


def _raise(message: Any) -> None:
    raise RuntimeError(str(message))


def _coerce(value: str) -> Any:
    text = value.strip()
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    if text in ("true", "false"):
        return text == "true"
    if text == "null":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return value
