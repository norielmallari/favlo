"""Chat completion for Gemma 4 26B-A4B."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import gemma4_26b_a4b.stream as stream_mod

from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.pipeline import ClientClosed as RequestClosed
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
from gemma4_26b_a4b.geometry import CONTEXT
from gemma4_26b_a4b.sampling import mode_card
from gemma4_26b_a4b.tools import (
    function_needs_arguments,
    parse_tool_calls,
    required_function,
)
from gemma4_26b_a4b.vision import collect_images

_DEFAULT_LIMIT = 4096

_TEMPLATE = Path(__file__).with_name("chat_template.jinja")
_THOUGHT = "<|channel>thought"
_THOUGHT_CLOSE = "<channel|>"
_EMPTY_THOUGHT = f"{_THOUGHT}\n{_THOUGHT_CLOSE}"
_TOOL_OPEN = "<|tool_call>"
_TOOL_RESPONSE_CLOSE = "<tool_response|>"
_MODEL_TURN = "<|turn>model\n"


_tokenizers: dict[Path, Tokenizer] = {}


def load_tokenizer(root: Path) -> Tokenizer:
    found = _tokenizers.get(root)
    if found is None:
        found = Tokenizer.from_file(str(root / "tokenizer.json"))
        _tokenizers[root] = found
    return found


def _replay_empty_tool_thoughts(text: str) -> str:
    """Put back the empty closed thought the model wrote before a tool call.

    A whitespace-only thought is omitted from the Response, so the next encode
    would drop ``<|channel>thought\\n<channel|>`` and break the prefix cache.
    Thinking off also puts that empty channel in the generation prompt before
    the call is drawn. Replay it whenever a tool call is not already preceded
    by a closed thought.
    """
    pieces: list[str] = []
    pos = 0
    while True:
        at = text.find(_TOOL_OPEN, pos)
        if at < 0:
            pieces.append(text[pos:])
            break
        pieces.append(text[pos:at])
        before = text[:at]
        if before.endswith(_THOUGHT_CLOSE):
            pieces.append(_TOOL_OPEN)
        elif before.endswith(_MODEL_TURN) or before.endswith(_TOOL_RESPONSE_CLOSE):
            pieces.append(_EMPTY_THOUGHT)
            pieces.append(_TOOL_OPEN)
        else:
            pieces.append(_TOOL_OPEN)
        pos = at + len(_TOOL_OPEN)
    return "".join(pieces)


def render_chat(
    messages: list[dict[str, Any]],
    thinking: bool,
    tools: Any = None,
    *,
    preserve_thinking: bool = False,
) -> str:
    """The canonical Gemma 4 template. Tool arguments are objects before render."""
    environment = Environment(autoescape=False, keep_trailing_newline=True)
    template = environment.from_string(_TEMPLATE.read_text(encoding="utf-8"))
    rendered = template.render(
        messages=_messages(messages),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=thinking,
        preserve_thinking=preserve_thinking,
        bos_token="<bos>",
        eos_token="<eos>",
    )
    return _replay_empty_tool_thoughts(rendered)


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    thinking: bool,
    tools: Any = None,
    *,
    preserve_thinking: bool = False,
) -> list[int]:
    """Render the canonical template and encode it, specials included."""
    text = render_chat(messages, thinking, tools, preserve_thinking=preserve_thinking)
    return list(load_tokenizer(root).encode(text).ids)


def decode_text(root: Path, tokens: list[int]) -> str:
    if not tokens:
        return ""
    return load_tokenizer(root).decode(tokens, skip_special_tokens=False)


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        calls = item.get("tool_calls")
        if isinstance(calls, list) and calls:
            item["tool_calls"] = [_tool_call(call) for call in calls]
        else:
            item.pop("reasoning", None)
            item.pop("reasoning_content", None)
        prepared.append(item)
    return prepared


def _tool_call(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    function = call.get("function")
    if not isinstance(function, dict):
        return call
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            message = "tool call arguments must be a JSON object"
            raise ValueError(message) from exc
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        message = "tool call arguments must be a JSON object"
        raise ValueError(message)
    return {**call, "function": {**function, "arguments": arguments}}


def reply_text(text: str, *, channel_open: bool = False) -> tuple[str, str]:
    """Reasoning and the reply span.

    A closed ``<|channel>thought`` is the trace. The reply is the text outside
    that channel, which is what the carrot check reads. A ``<|tool_call>``
    before ``<channel|>`` ends the trace, and the call is the reply span. An
    open thought stays the trace. ``channel_open`` is set when the prompt
    already ended on ``<|channel>thought``, which is the turn after a tool
    result.
    """
    reasoning: list[str] = []
    answer: list[str] = []
    rest = text
    opened = channel_open
    while rest:
        if opened:
            nested = _nested_open(rest)
            if nested is not None:
                rest = nested
                continue
            tool_at = rest.find(_TOOL_OPEN)
            close_at = rest.find(_THOUGHT_CLOSE)
            cut = tool_at if close_at < 0 else close_at if tool_at < 0 else min(tool_at, close_at)
            if cut < 0:
                reasoning.append(rest)
                break
            reasoning.append(rest[:cut])
            if tool_at == cut:
                answer.append(rest[tool_at:])
                break
            rest = rest[close_at + len(_THOUGHT_CLOSE) :]
            if rest.startswith("\n"):
                rest = rest[1:]
            opened = False
            continue
        open_at = rest.find(_THOUGHT)
        if open_at < 0:
            answer.append(rest)
            break
        tool_at = rest.find(_TOOL_OPEN)
        if 0 <= tool_at < open_at:
            answer.append(rest)
            break
        answer.append(rest[:open_at])
        rest = rest[open_at + len(_THOUGHT) :]
        if rest.startswith("\n"):
            rest = rest[1:]
        opened = True
    return "".join(reasoning).strip("\n"), "".join(answer)


def completion_fields(
    text: str,
    *,
    channel_open: bool = False,
    emit_tools: bool = True,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Reasoning, reply text, and tool calls. Thought markup stays out of the reply."""
    reasoning, answer = reply_text(text, channel_open=channel_open)
    if not emit_tools or not answer:
        return reasoning, answer or None, []
    content, calls = parse_tool_calls(answer)
    if not calls:
        return reasoning, answer, []
    if content:
        extra, rest = _after_call(content)
        if extra:
            reasoning = f"{reasoning}\n{extra}" if reasoning else extra
        content = rest or None
    return reasoning, content, calls


def _nested_open(text: str) -> str | None:
    """Text after a repeated ``<|channel>thought`` inside an open channel.

    The prompt already opened the channel. A second open is the model starting
    the trace again. The fragment before that marker is not the trace.
    """
    tool_at = text.find(_TOOL_OPEN)
    close_at = text.find(_THOUGHT_CLOSE)
    open_at = text.find(_THOUGHT)
    if open_at < 0:
        return None
    if close_at >= 0 and close_at < open_at:
        return None
    if tool_at >= 0 and tool_at < open_at:
        return None
    rest = text[open_at + len(_THOUGHT) :]
    if rest.startswith("\n"):
        rest = rest[1:]
    if rest == text:
        return None
    return rest


def _after_call(text: str) -> tuple[str, str]:
    """A thought or reply that followed a tool call.

    A call that ends an open thought leaves ``<channel|>`` in that tail.
    """
    rest = text.lstrip("\n")
    if rest.startswith(_THOUGHT_CLOSE):
        rest = rest[len(_THOUGHT_CLOSE) :]
        if rest.startswith("\n"):
            rest = rest[1:]
    return reply_text(rest)


class _Hooks(LanguagePipeline):
    """Disconnect shell shared by the Responses route and the chat adapter."""

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        del reply, selection
        runtime._stop = False


class Gemma4Chat(_Hooks):
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
        _carry_tool_thoughts(body.get("messages"), translated)
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


handle_chat = Gemma4Chat()
handle_responses = ResponsesPipeline()


def _reject(reply: Any, message: str) -> None:
    reply.json(400, {"error": {"message": message, "type": "invalid_request_error"}})


def _request_effort(body: dict[str, Any]) -> str | None:
    """``reasoning.effort`` after the Chat Completions translation.

    ``reasoning_effort`` and ``enable_thinking`` are already mapped onto that
    field. An omitted effort stays omitted so the registry row can decide.
    """
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str) or not effort.strip():
        return None
    return effort.strip().lower()


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
    thinking, card = mode_card(selection, _request_effort(prepared))
    temperature = _field(prepared, "temperature", card["temperature"])
    top_p = _field(prepared, "top_p", card["top_p"])
    top_k = int(card["top_k"])
    presence = float(card["presence_penalty"])
    try:
        messages = template_messages(prepared)[0]
        _merge_instruction(messages, format_instruction(prepared))
        search, note = web_search_context(prepared)
        if note:
            messages.append({"role": "user", "content": note})
        tools = function_tools(prepared.get("tools"))
        rendered, prompt = _fit_prompt(runtime, prepared, messages, tools, thinking)
    except (RequestError, ValueError) as exc:
        _reject(reply, str(exc))
        return
    pictures = collect_images(messages)
    channel_open = rendered.endswith("<|channel>thought\n")
    emit_tools = prepared.get("tool_choice") != "none"
    required = required_function(prepared.get("tool_choice")) if emit_tools else None
    lead = _call_lead(runtime, tools, required)
    if lead:
        prompt = [*prompt, *lead[1]]
    cap = prepared.get("max_output_tokens")
    limit = int(cap) if cap is not None else _DEFAULT_LIMIT
    stream = bool(prepared.get("stream", False))
    if stream:
        _stream(
            reply,
            runtime,
            prepared,
            state,
            prompt,
            pictures,
            limit,
            temperature,
            top_p,
            top_k,
            presence,
            channel_open,
            emit_tools,
            lead[0] if lead else "",
            search,
            note,
        )
        return
    tokens, _prefill_line, _decode_line = runtime.generate(
        prompt,
        limit,
        temperature,
        top_p,
        top_k,
        presence,
        images=pictures,
    )
    text = (lead[0] if lead else "") + decode_text(runtime.root, tokens)
    payload = _response_payload(
        reply,
        runtime,
        prepared,
        search,
        note,
        text,
        channel_open,
        emit_tools,
        len(prompt),
        len(tokens),
    )
    remember(state, prepared, payload)
    reply.json(200, payload)


def _field(body: dict[str, Any], name: str, fallback: Any) -> float:
    if body.get(name) is None:
        return float(fallback)
    return float(body[name])


def _merge_instruction(messages: list[dict[str, Any]], extra: str) -> None:
    if not extra:
        return
    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        messages[0]["content"] = extra + "\n\n" + messages[0]["content"]
        return
    messages.insert(0, {"role": "system", "content": extra})


def _carry_tool_thoughts(messages: Any, translated: dict[str, Any]) -> None:
    """Put a tool-call thought back in front of that call.

    The shared translator drops ``reasoning_content``. The template keeps the
    thought while the call is still after the latest user message.
    """
    items = translated.get("input")
    if not isinstance(messages, list) or not isinstance(items, list):
        return
    cursor = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("system", "developer"):
            continue
        if role == "tool":
            cursor = _skip_kind(items, cursor, "function_call_output")
            if cursor is None:
                return
            continue
        if role != "assistant":
            cursor = _skip_kind(items, cursor, "message")
            if cursor is None:
                return
            continue
        calls = message.get("tool_calls")
        trace = _thought_text(message)
        if isinstance(calls, list) and calls and trace:
            items.insert(cursor, _reasoning_item(trace))
            cursor += 1
        if isinstance(calls, list):
            for _call in calls:
                cursor = _skip_kind(items, cursor, "function_call")
                if cursor is None:
                    return
        if message.get("content"):
            cursor = _skip_kind(items, cursor, "message")
            if cursor is None:
                return


def _thought_text(message: dict[str, Any]) -> str:
    for name in ("reasoning_content", "reasoning"):
        text = message.get(name)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _reasoning_item(text: str) -> dict[str, Any]:
    return {"type": "reasoning", "content": [{"type": "reasoning_text", "text": text}]}


def _skip_kind(items: list[Any], cursor: int, kind: str) -> int | None:
    if cursor < len(items) and isinstance(items[cursor], dict) and items[cursor].get("type") == kind:
        return cursor + 1
    return None


def _fit_prompt(
    runtime: Any,
    prepared: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: bool,
) -> tuple[str, list[int]]:
    rendered, prompt = _encode(runtime.root, messages, thinking, tools)
    if len(prompt) <= CONTEXT:
        return rendered, prompt
    if prepared.get("truncation") != "auto":
        raise RequestError("input does not fit")
    while len(prompt) > CONTEXT and len(messages) > 1:
        drop = 1 if messages and messages[0].get("role") == "system" else 0
        if drop >= len(messages):
            break
        del messages[drop]
        rendered, prompt = _encode(runtime.root, messages, thinking, tools)
    if len(prompt) > CONTEXT:
        raise RequestError("input does not fit")
    return rendered, prompt


def _encode(
    root: Path,
    messages: list[dict[str, Any]],
    thinking: bool,
    tools: list[dict[str, Any]],
) -> tuple[str, list[int]]:
    rendered = render_chat(messages, thinking, tools or None)
    return rendered, list(load_tokenizer(root).encode(rendered).ids)


def _call_lead(
    runtime: Any, tools: list[dict[str, Any]], required: str | None
) -> tuple[str, list[int]] | None:
    """The named call already in the prompt, so generation writes its arguments."""
    if not required:
        return None
    text = f"<|tool_call>call:{required}{{"
    if not function_needs_arguments(tools, required):
        text += "}<tool_call|>"
    ids = list(load_tokenizer(runtime.root).encode(text, add_special_tokens=False).ids)
    return text, ids


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
    channel_open: bool,
    emit_tools: bool,
    prompt_tokens: int,
    completion_tokens: int,
    response_id: str | None = None,
) -> dict[str, Any]:
    reasoning, content, calls = completion_fields(
        text, channel_open=channel_open, emit_tools=True
    )
    if not emit_tools:
        calls = []
    trace = reasoning.strip()
    output: list[dict[str, Any]] = []
    if search is not None:
        output.append(search)
    store = prepared.get("store", True) is not False
    include = prepared.get("include")
    encrypt = (not store) or (
        isinstance(include, list) and "reasoning.encrypted_content" in include
    )
    if trace:
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": new_id("rs_"),
            "summary": [],
            "status": "completed",
            "content": [{"type": "reasoning_text", "text": trace}],
        }
        summary = prepared.get("reasoning")
        if isinstance(summary, dict) and summary.get("summary"):
            item["summary"] = [{"type": "summary_text", "text": trace}]
        if encrypt:
            item["encrypted_content"] = trace
        output.append(item)
    visible = content or ""
    annotations = citations(_search_hits(note), visible) if note else []
    if calls:
        if visible:
            output.append(_message_item(visible, annotations))
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
        output.append(_message_item(visible, annotations))
        if getattr(runtime, "_hit_limit", False):
            status = "incomplete"
            incomplete = "max_output_tokens"
        else:
            status = "completed"
            incomplete = None
    counted = int(getattr(runtime, "prompt_tokens", 0) or 0) or prompt_tokens
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    payload = response_object(
        model=getattr(reply, "model_id", "") or prepared.get("model") or "",
        output=output,
        usage=usage_object(counted, completion_tokens, cached, 0),
        body=prepared,
        status=status,
        incomplete=incomplete,
        response_id=response_id or new_id("resp_"),
    )
    payload["_chat_usage"] = _usage(counted, completion_tokens)
    return payload


def _message_item(text: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "message",
        "id": new_id("msg_"),
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": annotations}],
    }


class ChatWire:
    """Writes a Response as Chat Completions."""

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
            self._send(
                [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            )
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
    if not isinstance(choices, list) or not choices:
        return
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
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
    pictures: list[bytes],
    limit: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    channel_open: bool,
    emit_tools: bool,
    lead: str,
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
    handle = stream_mod.launch(runtime.root, lead, channel_open)
    failed: list[BaseException] = []
    opened: set[str] = set()

    def emit_piece(channel: str, visible: str) -> None:
        if not visible:
            return
        if channel == "reasoning" and not visible.strip() and channel not in opened:
            return
        if channel not in opened:
            opened.add(channel)
            _open_item(event, channel, reasoning_id, message_id)
        _emit_response(event, channel, visible, reasoning_id, message_id)

    def drain() -> None:
        error: BaseException | None = None
        while True:
            try:
                item = handle.take()
            except queue.Empty:
                if handle.running():
                    continue
                error = error or RuntimeError("gemma stream process exited")
                break
            if item is None:
                break
            if error is not None or item[0] != "delta":
                if item[0] == "error" and error is None:
                    error = RuntimeError(str(item[1]))
                continue
            try:
                emit_piece(str(item[1]), str(item[2]))
            except BaseException as exc:
                error = exc
                runtime._stop = True
        if error is not None:
            failed.append(error)

    drainer = threading.Thread(target=drain, name="favlo-gemma-emit", daemon=True)
    drainer.start()

    def enqueue(token: int) -> None:
        handle.put(token)

    try:
        tokens, _prefill_line, _decode_line = runtime.generate(
            prompt,
            limit,
            temperature,
            top_p,
            top_k,
            presence,
            on_token=enqueue,
            images=pictures,
        )
    finally:
        handle.finish()
        drainer.join()
        handle.close()
    if failed:
        raise failed[0]
    text = lead + decode_text(runtime.root, tokens)
    payload = _response_payload(
        reply,
        runtime,
        prepared,
        search,
        note,
        text,
        channel_open,
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
            event({"type": "response.output_item.added", "output_index": index, "item": opened_item})
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
            "content": [{"type": "output_text", "text": "", "annotations": [], "logprobs": []}],
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


def openai_message(
    reasoning: str, content: str | None, calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """One assistant message. An empty thought is omitted. A missing reply is null."""
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    trace = reasoning.strip()
    if trace:
        message["reasoning"] = trace
        message["reasoning_content"] = trace
    if calls:
        message["tool_calls"] = calls
    return message


def _emit(
    event: Any,
    reply: Any,
    channel: str,
    visible: str,
    created: int,
    sent: dict[str, str] | None = None,
) -> None:
    delta = _delta(channel, visible)
    if delta is None:
        return
    if sent is not None:
        field = "reasoning" if channel == "reasoning" else "content"
        sent[field] = sent.get(field, "") + visible
    event(
        reply,
        _frame(
            reply,
            [{"index": 0, "delta": delta, "finish_reason": None}],
            created,
        ),
    )


def _emit_unsent(
    event: Any,
    reply: Any,
    created: int,
    sent: dict[str, str],
    reasoning: str,
    content: str | None,
) -> None:
    """Send a trace or reply the token stream did not already deliver."""
    if reasoning.strip() and not sent.get("reasoning", "").strip():
        _emit(event, reply, "reasoning", reasoning.strip(), created, sent)
    if content and content.strip() and not sent.get("content", "").strip():
        _emit(event, reply, "content", content, created, sent)


def _delta(channel: str, visible: str) -> dict[str, str] | None:
    """OpenAI delta for one piece. A whitespace-only thought is not a trace."""
    if channel == "reasoning":
        if not visible.strip():
            return None
        return {"reasoning": visible, "reasoning_content": visible}
    if not visible:
        return None
    return {"content": visible}


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
        "model": getattr(reply, "model_id", "") or "",
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def cut_stop(text: str, stops: list[str]) -> str:
    """Text before the earliest request stop. The stop itself stays out."""
    cut = len(text)
    for stop in stops:
        if not stop:
            continue
        at = text.find(stop)
        if at >= 0:
            cut = min(cut, at)
    return text[:cut]


def _stop_strings(body: dict[str, Any]) -> list[str]:
    raw = body.get("stop")
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item]
    return []


def _finish_reason(runtime: Any, calls: list[dict[str, Any]]) -> str:
    if calls:
        return "tool_calls"
    if getattr(runtime, "_hit_limit", False):
        return "length"
    return "stop"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
