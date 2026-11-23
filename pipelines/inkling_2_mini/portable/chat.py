"""Responses shell for Inkling 2 Mini.

The checkpoint Jinja template renders the effort line and the tool declaration.
Generation stops on ``<|content_model_end_sampling|>``. Thinking is
``<|content_thinking|>`` and a tool call is ``<|content_invoke_tool_json|>``.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.responses import (
    RequestError,
    format_instruction,
    function_tools,
    new_id,
    prepare_request,
    response_object,
    template_messages,
    usage_object,
    web_search_context,
)
from inkling_2_mini.portable import markup
from inkling_2_mini.portable.closed import RequestClosed
from inkling_2_mini.portable.sampling import (
    DEFAULT_THINKING,
    DEFAULT_VARIANT,
    VARIANTS,
    resolve_effort,
    template_effort,
)

PIPELINE = "inkling_2_mini"
EOS = 200006
STOP = {EOS}
_LOAD_LOCK = threading.Lock()
_TOKENIZERS: dict[str, Tokenizer] = {}
_TEMPLATES: dict[str, Any] = {}


def _raise(message: str) -> None:
    raise RuntimeError(message)


def _tojson(value: Any, sort_keys: bool = False, separators: tuple[str, str] = (",", ":")) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=separators, ensure_ascii=False)


def load_tokenizer(root: Path) -> Tokenizer:
    """The checkpoint tokenizer, loaded once for this process."""
    key = str(root)
    with _LOAD_LOCK:
        found = _TOKENIZERS.get(key)
        if found is None:
            found = Tokenizer.from_file(str(root / "tokenizer.json"))
            _TOKENIZERS[key] = found
        return found


def _chat_template(root: Path) -> Any:
    """The checkpoint Jinja template, compiled once for this process."""
    key = str(root)
    with _LOAD_LOCK:
        found = _TEMPLATES.get(key)
        if found is not None:
            return found
        text = (root / "chat_template.jinja").read_text()
        text = text.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
        env = Environment()
        env.globals["raise_exception"] = _raise
        env.filters["tojson"] = _tojson
        found = env.from_string(text)
        _TEMPLATES[key] = found
        return found


def _objects(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The template rejects a tool-call argument that is still a JSON string."""
    copied: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        calls = item.get("tool_calls")
        if isinstance(calls, list):
            shaped = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                clone = dict(call)
                function = clone.get("function")
                if isinstance(function, dict):
                    function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function["arguments"] = json.loads(arguments) if arguments else {}
                    clone["function"] = function
                shaped.append(clone)
            item["tool_calls"] = shaped
        copied.append(item)
    return copied


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    thinking: bool,
    *,
    effort: str | float | int | None = None,
    opening: str | None = None,
) -> list[int]:
    """Render the checkpoint template, append the opening, and encode it.

    The template has no generation prompt. ``opening`` is the markup the
    prompt ends with. When it is omitted, a request with no tools takes the
    opening of its card. ``reasoning_effort`` is the template's number, in
    ``[0.0, 0.99]``.
    """
    chosen = effort if effort is not None else ("high" if thinking else "none")
    text = _chat_template(root).render(
        messages=_objects(messages),
        tools=_declared_tools(tools),
        reasoning_effort=template_effort(chosen),
    )
    if opening is None:
        opening = forced_text(thinking, None, plain=not tools)
    return list(load_tokenizer(root).encode(text + opening).ids)


_SCHEMA_KEYS = ("type", "description", "enum", "properties", "required", "items")


def _slim_schema(value: Any) -> Any:
    """The declaration the checkpoint template was written for.

    Extra JSON-schema fields such as ``title`` and ``additionalProperties``
    make a required call come out as prose.
    """
    if isinstance(value, list):
        return [_slim_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    kept: dict[str, Any] = {}
    for key in _SCHEMA_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key in ("properties", "items"):
            kept[key] = _slim_schema(item)
        else:
            kept[key] = item
    return kept


def _declared_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    declared: list[dict[str, Any]] = []
    for tool in tools:
        copied = dict(tool)
        function = copied.get("function")
        if isinstance(function, dict):
            function = dict(function)
            function["parameters"] = _slim_schema(function.get("parameters") or {})
            copied["function"] = function
        elif "parameters" in copied:
            copied["parameters"] = _slim_schema(copied.get("parameters") or {})
        declared.append(copied)
    return declared


def encode_text(root: Path, text: str) -> list[int]:
    return list(load_tokenizer(root).encode(text).ids)


def decode_text(root: Path, ids: list[int]) -> str:
    if not ids:
        return ""
    return load_tokenizer(root).decode(ids, skip_special_tokens=False)


def _effort(body: dict[str, Any], selection: Any) -> str | float:
    raw = _requested_effort(body)
    if isinstance(raw, str) and raw.strip():
        mapped = resolve_effort(raw)
        if mapped:
            return mapped
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return template_effort(raw)
    if selection is not None:
        stored = getattr(selection, "reasoning_effort", None)
        if isinstance(stored, str) and stored:
            mapped = resolve_effort(stored)
            if mapped:
                return mapped
    return "high"


def _requested_effort(body: dict[str, Any]) -> str | float | int | None:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        effort = reasoning.get("effort")
        if isinstance(effort, str | int | float) and not isinstance(effort, bool):
            return effort
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("reasoning_effort") is not None:
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, str | int | float) and not isinstance(effort, bool):
            return effort
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False:
        return "none"
    return None


def _card(selection: Any) -> dict[str, Any]:
    sampling = getattr(selection, "sampling", None) if selection is not None else None
    if isinstance(sampling, dict):
        return sampling
    return dict(VARIANTS[DEFAULT_VARIANT])


def _number(body: dict[str, Any], card: dict[str, Any], key: str, default: float) -> float:
    if body.get(key) is not None:
        return float(body[key])
    if card.get(key) is not None:
        return float(card[key])
    return default


def _required_name(body: dict[str, Any]) -> str | None:
    """The function a required tool choice names, or None.

    The Responses body uses ``{"type": "function", "name": ...}``. A chat body
    may still carry ``required_function_name`` or a nested ``function`` object.
    """
    choice = body.get("tool_choice")
    if not isinstance(choice, dict):
        return None
    if choice.get("mode") != "required" and choice.get("type") != "function":
        return None
    name = choice.get("name")
    if isinstance(name, str) and name:
        return name
    named = choice.get("required_function_name")
    if isinstance(named, str) and named:
        return named
    function = choice.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str) and function["name"]:
        return function["name"]
    return None


def parse_completion(text: str) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Split a decoded completion into reasoning, visible text, and tool calls."""
    reasoning, visible, found = markup.parse_completion(text)
    calls = [
        {
            "id": new_id("call_"),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, separators=(",", ":"), sort_keys=True),
            },
        }
        for name, args in found
    ]
    return reasoning, visible, calls


def _forbid_tools(root: Path, emit: bool) -> set[int] | None:
    """Token ids a ``tool_choice`` of ``none`` may not sample."""
    if emit:
        return None
    return set(load_tokenizer(root).encode(markup.TOOL).ids)


def forced_text(thinking: bool, required: str | None, plain: bool) -> str:
    """Markup the prompt ends with, so the model opens the right channel.

    A required tool opens that call. Thinking off with no callable tool opens
    the reply. Otherwise the model writes its own message marker.
    """
    if required:
        name = json.dumps(required, ensure_ascii=False)
        return f'{markup.HEADER}{required}{markup.TOOL}{{"name":{name},"args":'
    if not thinking and plain:
        return markup.HEADER + markup.TEXT
    return ""


def _message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": new_id("msg_"),
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _payload(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    reasoning: str,
    content: str | None,
    calls: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    hit_limit: bool,
    response_id: str,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if search is not None:
        output.append(search)
    if reasoning:
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": new_id("rs_"),
            "summary": [],
            "status": "completed",
            "content": [{"type": "reasoning_text", "text": reasoning}],
            "encrypted_content": reasoning,
        }
        summary = body.get("reasoning")
        if isinstance(summary, dict) and summary.get("summary"):
            item["summary"] = [{"type": "summary_text", "text": reasoning}]
        output.append(item)
    if calls:
        if content:
            output.append(_message_item(content))
        for call in calls:
            function = call["function"]
            output.append(
                {
                    "type": "function_call",
                    "id": new_id("fc_"),
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": function["arguments"],
                    "status": "completed",
                }
            )
        status = "completed"
        incomplete = None
    else:
        output.append(_message_item(content or ""))
        status = "incomplete" if hit_limit else "completed"
        incomplete = "max_output_tokens" if hit_limit else None
    payload = response_object(
        model=getattr(reply, "model_id", "") or body.get("model") or "",
        output=output,
        usage=usage_object(
            prompt_tokens, completion_tokens, int(getattr(runtime, "_cached_tokens", 0) or 0), 0
        ),
        body=body,
        status=status,
        incomplete=incomplete,
        response_id=response_id,
    )
    payload["_chat_usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "context_length": int(getattr(runtime, "context", 0) or 0),
    }
    return payload


def _device_packages() -> frozenset[str]:
    """Device directories shipped beside this entry. The host picks one."""
    root = Path(__file__).resolve().parent.parent
    return frozenset(
        name for name in ("metal41", "cuda") if (root / name / "__init__.py").is_file()
    )


class Pipeline(LanguagePipeline):
    """One Responses completion. The runtime owns the token loop."""

    variants = VARIANTS
    default_variant = DEFAULT_VARIANT
    default_thinking = DEFAULT_THINKING
    default_reasoning_effort = "high"

    def has_vision(self) -> bool:
        return True

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        super().on_open(runtime, reply, selection)

    def on_finish(self, runtime: Any, selection: Any, *, complete: bool) -> None:
        super().on_finish(runtime, selection, complete=complete)

    def client_closed(self, exc: OSError) -> bool:
        return str(exc) == "client closed"

    def handle_responses(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        del args
        _respond(reply, runtime, body, selection, state)

    def main(self, argv: Any) -> int:
        from inkling_2_mini.metal41.commands import main

        return main(argv)

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.host import select_device
        from favlo.pipelines import bind_open_runtime

        chosen = select_device(_device_packages())
        if chosen != "metal41":
            message = "this runtime cannot run here"
            raise RuntimeError(message)
        from inkling_2_mini.metal41.runtime import open_device

        return bind_open_runtime(open_device, path, **kwargs)


def _respond(
    reply: Any, runtime: Any, body: dict[str, Any], selection: Any, state: dict[str, Any] | None
) -> None:
    del state
    try:
        prepared = prepare_request(body, None)
    except RequestError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    effort = _effort(prepared, selection)
    thinking = template_effort(effort) != 0.0
    card = _card(selection)
    temperature = _number(prepared, card, "temperature", 1.0)
    top_p = _number(prepared, card, "top_p", 1.0)
    top_k = int(_number(prepared, card, "top_k", 0))
    presence = _number(prepared, card, "presence_penalty", 0.0)
    messages, _saw = template_messages(prepared)
    extra = format_instruction(prepared)
    if extra:
        messages = [{"role": "system", "content": extra}, *messages]
    search, note = web_search_context(prepared)
    if note:
        messages.append({"role": "user", "content": note})
    tools = function_tools(prepared.get("tools"))
    required = _required_name(prepared)
    if required:
        thinking = False
        effort = "none"
    emit_tools = prepared.get("tool_choice") != "none" and prepared.get("tool_choice") != {
        "mode": "none"
    }
    opening = forced_text(thinking, required, plain=not tools or not emit_tools)
    runtime._vision_plan = None
    try:
        prompt = encode_chat(
            runtime.root, messages, tools, thinking, effort=effort, opening=opening
        )
        from inkling_2_mini.metal41.vision import expand_prompt

        prompt, plan = expand_prompt(messages, prompt)
        runtime._vision_plan = plan or None
    except (RuntimeError, json.JSONDecodeError, ValueError) as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    room = int(getattr(runtime, "context", len(prompt) + 1)) - len(prompt)
    if room < 1:
        reply.json(
            400, {"error": {"message": "the prompt does not fit", "type": "invalid_request_error"}}
        )
        return
    from favlo.logs import output_token_limit

    requested = prepared.get("max_output_tokens")
    if requested is None:
        requested = output_token_limit(prepared)
    limit = room if requested is None else min(int(requested), room)
    _prefix_run(runtime, prepared)
    shape = _Shape(thinking, emit_tools, opening, search)
    sampler = (temperature, top_p, top_k, presence)
    if bool(prepared.get("stream", False)):
        _stream(reply, runtime, prompt, limit, sampler, prepared, shape)
        return
    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        fill=bool(prepared.get("favlo:fill_decode")),
        forbid=_forbid_tools(runtime.root, shape.emit_tools),
    )
    text = opening + decode_text(runtime.root, tokens)
    payload = _finished(reply, runtime, prepared, shape, text, len(prompt), len(tokens))
    reply.json(200, payload)


def _prefix_run(runtime: Any, prepared: dict[str, Any]) -> None:
    """Point this request's prefix store at its scratch partition and budget.

    Absent fields send the next request back to the home partition.
    """
    budget = prepared.get("favlo:prefix_budget")
    scratch = prepared.get("favlo:prefix_scratch")
    runtime._prefix_budget = (
        budget if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0 else None
    )
    runtime._prefix_scratch = scratch if isinstance(scratch, str) and scratch else None


class _Shape:
    """What one request shows: reasoning, calls, the forced opening, and search."""

    def __init__(
        self, thinking: bool, emit_tools: bool, opening: str, search: dict[str, Any] | None
    ) -> None:
        self.thinking = thinking
        self.emit_tools = emit_tools
        self.opening = opening
        self.search = search


def _finished(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    shape: _Shape,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    response_id: str | None = None,
) -> dict[str, Any]:
    reasoning, content, calls = parse_completion(text)
    if not shape.emit_tools:
        calls = []
    return _payload(
        reply,
        runtime,
        body,
        reasoning if shape.thinking else "",
        content,
        calls,
        prompt_tokens,
        completion_tokens,
        bool(getattr(runtime, "_hit_limit", False)),
        response_id or new_id("resp_"),
        shape.search,
    )


class _Decoder:
    """Token ids to text in order, holding back a split UTF-8 character."""

    def __init__(self, root: Path) -> None:
        self.tokenizer = load_tokenizer(root)
        self.pending: list[int] = []

    def take(self, token: int | None) -> str:
        if token is not None:
            self.pending.append(int(token))
        if not self.pending:
            return ""
        text = self.tokenizer.decode(self.pending, skip_special_tokens=False)
        if token is not None and text.endswith("\ufffd") and len(self.pending) < 8:
            return ""
        self.pending = []
        return text


def _stream(
    reply: Any,
    runtime: Any,
    prompt: list[int],
    limit: int,
    sampler: tuple[float, float, int, float],
    body: dict[str, Any],
    shape: _Shape,
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
    shell = response_object(
        model=getattr(reply, "model_id", "") or body.get("model") or "",
        output=[],
        usage=usage_object(0, 0, 0, 0),
        body=body,
        status="in_progress",
        incomplete=None,
        response_id=response_id,
    )
    event({"type": "response.created", "response": shell})
    event({"type": "response.in_progress", "response": shell})
    order: list[str] = []
    ids: dict[str, str] = {}
    if shape.search is not None:
        order.append("search")
        event({"type": "response.output_item.added", "output_index": 0, "item": shape.search})
        event({"type": "response.output_item.done", "output_index": 0, "item": shape.search})

    def open_item(key: str, item: dict[str, Any]) -> int:
        order.append(key)
        ids[key] = item["id"]
        index = len(order) - 1
        event({"type": "response.output_item.added", "output_index": index, "item": item})
        return index

    def emit(channel: str, value: Any) -> None:
        if channel == "reasoning":
            if not shape.thinking or not value:
                return
            if "reasoning" not in ids:
                open_item(
                    "reasoning",
                    {
                        "type": "reasoning",
                        "id": new_id("rs_"),
                        "summary": [],
                        "content": [],
                        "status": "in_progress",
                    },
                )
            event(
                {
                    "type": "response.reasoning_text.delta",
                    "content_index": 0,
                    "delta": value,
                    "item_id": ids["reasoning"],
                    "output_index": order.index("reasoning"),
                }
            )
        elif channel == "content":
            if not value:
                return
            if "message" not in ids:
                open_item(
                    "message",
                    {
                        "type": "message",
                        "id": new_id("msg_"),
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [{"type": "output_text", "text": "", "annotations": []}],
                    },
                )
            event(
                {
                    "type": "response.output_text.delta",
                    "content_index": 0,
                    "delta": value,
                    "item_id": ids["message"],
                    "output_index": order.index("message"),
                    "logprobs": [],
                }
            )
        elif shape.emit_tools and channel == "tool_start":
            key = f"call{value['index']}"
            call_id = new_id("call_")
            open_item(
                key,
                {
                    "type": "function_call",
                    "id": "fc_" + call_id.removeprefix("call_"),
                    "call_id": call_id,
                    "name": value["name"],
                    "arguments": "",
                    "status": "in_progress",
                },
            )
            ids[key + "_call"] = call_id
        elif shape.emit_tools and channel == "tool_args":
            key = f"call{value['index']}"
            if key in ids and value["arguments"]:
                event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "delta": value["arguments"],
                        "item_id": ids[key],
                        "output_index": order.index(key),
                        "call_id": ids[key + "_call"],
                    }
                )

    pending: queue.SimpleQueue[int | None] = queue.SimpleQueue()
    failed: list[BaseException] = []
    pieces: list[str] = [shape.opening]

    def write() -> None:
        decoder = _Decoder(runtime.root)
        gate = markup.StreamGate()
        try:
            for channel, value in gate.push(shape.opening):
                emit(channel, value)
            while True:
                token = pending.get()
                piece = decoder.take(token)
                if piece:
                    pieces.append(piece)
                    for channel, value in gate.push(piece):
                        emit(channel, value)
                if token is None:
                    for channel, value in gate.finish():
                        emit(channel, value)
                    return
        except BaseException as exc:
            failed.append(exc)

    writer = threading.Thread(target=write, name="inkling-stream", daemon=True)
    writer.start()

    def enqueue(token: int) -> None:
        if failed:
            raise RequestClosed()
        pending.put(token)

    temperature, top_p, top_k, presence = sampler
    try:
        tokens = runtime.generate(
            prompt,
            limit,
            STOP,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            fill=bool(body.get("favlo:fill_decode")),
            on_token=enqueue,
            forbid=_forbid_tools(runtime.root, shape.emit_tools),
        )
    finally:
        pending.put(None)
        writer.join()
    if failed:
        raise failed[0]
    payload = _finished(
        reply, runtime, body, shape, "".join(pieces), len(prompt), len(tokens), response_id
    )
    calls = 0
    for index, item in enumerate(payload["output"]):
        kind = item.get("type")
        if kind == "function_call":
            key = f"call{calls}"
            calls += 1
            if key in ids:
                item["id"] = ids[key]
                item["call_id"] = ids[key + "_call"]
        elif kind == "reasoning":
            key = "reasoning"
        elif kind == "message":
            key = "message"
        else:
            continue
        if key in ids:
            item["id"] = ids[key]
        else:
            order.append(key)
            added = dict(item)
            added["status"] = "in_progress"
            if kind == "function_call":
                added["arguments"] = ""
            event({"type": "response.output_item.added", "output_index": index, "item": added})
            if kind == "message":
                event(
                    {
                        "type": "response.output_text.delta",
                        "content_index": 0,
                        "delta": item["content"][0]["text"],
                        "item_id": item["id"],
                        "output_index": index,
                        "logprobs": [],
                    }
                )
        if kind == "function_call":
            event(
                {
                    "type": "response.function_call_arguments.done",
                    "arguments": item.get("arguments") or "{}",
                    "item_id": item["id"],
                    "output_index": index,
                }
            )
        event({"type": "response.output_item.done", "output_index": index, "item": item})
    event({"type": "response.completed", "response": payload})
    try:
        reply.done()
    except OSError as exc:
        raise RequestClosed() from exc


pipeline = Pipeline()
handle_responses = pipeline
