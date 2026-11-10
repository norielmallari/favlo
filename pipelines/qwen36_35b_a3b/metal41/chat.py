"""Chat template and one completion on the text forward."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from qwen36_35b_a3b.metal41.cache import DEFAULT_CONTEXT_LENGTH
from favlo.language_pipeline import LanguagePipeline
from qwen36_35b_a3b.metal41.forward import RequestClosed
from qwen36_35b_a3b.metal41.sampling import DEFAULT_VARIANT, VARIANTS, mode_card
from qwen36_35b_a3b.metal41.tools import (
    StreamText,
    completion_fields,
    decode_shown_text,
    decode_unicode_escapes,
    function_has_parameters,
    normalize_messages,
    parse_tool_calls,
    required_function,
)
from qwen36_35b_a3b.metal41.vision import prepare_prompt
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

STOP = {248046, 248044}


def _sampling(
    body: dict[str, Any], thinking: bool, variant: dict[str, Any] | None
) -> tuple[float, float, int]:
    del thinking
    card = variant if variant is not None else VARIANTS[DEFAULT_VARIANT]
    defaults = (
        float(card["temperature"]),
        float(card["top_p"]),
        int(card["top_k"]),
    )
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    return (
        defaults[0] if temperature is None else float(temperature),
        defaults[1] if top_p is None else float(top_p),
        defaults[2] if top_k is None else int(top_k),
    )


def _presence(body: dict[str, Any], thinking: bool, variant: dict[str, Any] | None) -> float:
    penalty = body.get("presence_penalty")
    if penalty is not None:
        return float(penalty)
    del thinking
    card = variant if variant is not None else VARIANTS[DEFAULT_VARIANT]
    if "presence_penalty" in card:
        return float(card["presence_penalty"])
    return float(VARIANTS[DEFAULT_VARIANT]["presence_penalty"])


def _frequency(body: dict[str, Any]) -> float:
    penalty = body.get("frequency_penalty")
    if penalty is None:
        return 0.0
    return float(penalty)


def requested_limit(body: dict[str, Any]) -> int | None:
    """Request or settings cap. None when neither set one.

    ``max_output_tokens`` is the Responses field. Chat Completions names are
    read after it, then ``settings.json``.
    """
    cap = body.get("max_output_tokens")
    if cap is not None:
        return int(cap)
    from favlo.logs import output_token_limit

    return output_token_limit(body)


def thinking_token_budget(thinking: bool) -> int | None:
    """Settings cap while thinking. None does not close the think block on a count.

    Instruct passes 0 so the repeat guard stays off. The template already closed it.
    """
    if not thinking:
        return 0
    from favlo.logs import configured_thinking_budget

    return configured_thinking_budget()


def _mtp_flag(body: dict[str, Any], selection: Any = None) -> bool:
    """Draft verify. A request field wins. Otherwise the registry row is used.

    A row that omits ``mtp`` is off. Sampling settings do not change the flag.
    """
    if "mtp" in body:
        flag = body["mtp"]
        if isinstance(flag, bool):
            return flag
        message = "mtp must be true or false"
        raise RuntimeError(message)
    folder = getattr(selection, "folder", None)
    if not isinstance(folder, str) or not folder:
        return False
    from favlo.registry import model_mtp

    enabled, _depth = model_mtp(folder)
    return enabled


def _mtp_depth(body: dict[str, Any], selection: Any = None) -> int:
    """Draft length. A request field wins. Otherwise the registry row is used.

    A row that omits ``mtp_depth`` is three. Sampling settings do not change it.
    """
    if "mtp_depth" in body:
        raw = body["mtp_depth"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            message = "mtp_depth must be a positive integer"
            raise RuntimeError(message)
        return raw
    folder = getattr(selection, "folder", None)
    if not isinstance(folder, str) or not folder:
        return 3
    from favlo.registry import model_mtp

    _enabled, depth = model_mtp(folder)
    return depth


def completion_limit(max_position: int, prompt_tokens: int, requested: int | None) -> int:
    """Tokens still available in the checkpoint context, cut by a request cap."""
    room = max(0, int(max_position) - int(prompt_tokens))
    if requested is None:
        return room
    return max(0, min(int(requested), room))


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def _preserve_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    message = "preserve_thinking must be true or false"
    raise RuntimeError(message)


def _chat_preserve(body: dict[str, Any]) -> bool | None:
    """``chat_template_kwargs.preserve_thinking``, or None when that key is absent."""
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict) or "preserve_thinking" not in kwargs:
        return None
    return _preserve_flag(kwargs.get("preserve_thinking"))


def preserve_thinking(body: dict[str, Any], settings: dict[str, Any] | None = None) -> bool:
    """Whether historical reasoning stays in the prompt.

    A top-level ``preserve_thinking`` wins over ``chat_template_kwargs``.
    Either wins over ``settings.json``. Omitted is on, so a later prompt
    extends the stored prefix. False keeps a think block only on assistant
    turns after the latest user message.
    """
    requested = (
        _preserve_flag(body.get("preserve_thinking"))
        if "preserve_thinking" in body
        else None
    )
    if requested is None:
        requested = _chat_preserve(body)
    if requested is not None:
        return requested
    if settings is None:
        from favlo.logs import favlo_home, load_settings

        settings = load_settings(favlo_home())
    value = settings.get("preserve_thinking")
    if value is None:
        return True
    return bool(value)


def _bridge_omitted_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put an empty assistant turn where the previous prompt already closed thinking.

    Used when historical thinking is kept. That empty turn renders the closed
    think, so a follow-up that omitted it stays a prefix of the stored tokens.
    """
    bridged: list[dict[str, Any]] = []
    previous: str | None = None
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if role == "user" and previous in {"user", "tool"}:
            bridged.append({"role": "assistant", "content": ""})
        bridged.append(message)
        previous = role if isinstance(role, str) else None
    return bridged


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    tools: Any,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
    *,
    preserve_thinking: bool = False,
) -> list[int]:
    """Render the checkpoint template and encode it.

    A passed reasoning effort is ignored. This checkpoint opens or closes
    thinking only through ``enable_thinking``. ``preserve_thinking`` is the
    template flag: false keeps reasoning on assistant turns after the latest
    user message, and true keeps it on every assistant turn.
    """
    del reasoning_effort
    prepared = normalize_messages(messages)
    if preserve_thinking:
        prepared = _bridge_omitted_assistant(prepared)
    template = (root / "chat_template.jinja").read_text()
    env = Environment()
    env.globals["raise_exception"] = _raise
    rendered_args: dict[str, Any] = {
        "messages": prepared,
        "tools": tools or None,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        "add_vision_id": False,
        "preserve_thinking": preserve_thinking,
    }
    rendered = env.from_string(template).render(**rendered_args)
    return list(load_tokenizer(root).encode(rendered).ids)


def encode_text(root: Path, text: str) -> list[int]:
    return list(load_tokenizer(root).encode(text).ids)


def think_open_id(root: Path) -> int:
    """Token id of ``<think>``. Instruct forbids drawing it again."""
    opened = encode_text(root, "<think>")
    if len(opened) != 1:
        raise RuntimeError("the think open is not one token")
    return opened[0]


def instruct_forbid(root: Path) -> set[int]:
    """Think markers. Instruct already closed the block, so neither is drawn."""
    _close, end = think_close_ids(root)
    return {think_open_id(root), end}


def tool_call_forbid(root: Path) -> set[int]:
    """Tool markup. ``tool_choice`` ``none`` keeps those tokens out of the reply."""
    opened = encode_text(root, "<tool_call>")
    closed = encode_text(root, "</tool_call>")
    if len(opened) != 1 or len(closed) != 1:
        raise RuntimeError("the tool call markers are not one token each")
    return {opened[0], closed[0]}


def think_close_ids(root: Path) -> tuple[list[int], int]:
    """Token ids of the template close ``\\n</think>\\n\\n``, and the ``</think>`` id."""
    close = encode_text(root, "\n</think>\n\n")
    end = encode_text(root, "</think>")
    if len(end) != 1 or end[0] not in close:
        raise RuntimeError("the think close is not one token")
    return close, end[0]


def _forced_call(root: Path, tools: Any, required: str | None) -> tuple[list[int], bool]:
    """Opening of a named call, or the whole call when that function takes no arguments."""
    if not required:
        return [], False
    if function_has_parameters(tools, required):
        return encode_text(root, f"<tool_call>\n<function={required}>\n"), False
    text = f"<tool_call>\n<function={required}>\n</function>\n</tool_call>"
    return encode_text(root, text), True


def decode_text(root: Path, ids: list[int]) -> str:
    return load_tokenizer(root).decode(ids, skip_special_tokens=False)


def _raise(message: str) -> None:
    raise ValueError(message)


class Pipeline(LanguagePipeline):
    """One Responses generation for this package."""

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        from qwen36_35b_a3b.metal41.firing import open_request

        super().on_open(runtime, reply, selection)
        runtime._halted = False
        open_request(runtime)

    def client_closed(self, exc: OSError) -> bool:
        return str(exc) == "client closed"

    def on_finish(self, runtime: Any, selection: Any, *, complete: bool) -> None:
        from qwen36_35b_a3b.metal41.firing import finish_request

        from qwen36_35b_a3b.metal41.vision import release_vision

        release_vision(runtime)
        runtime._vision_at = None
        runtime._rope_positions = None
        runtime._rope_delta = 0
        super().on_finish(runtime, selection, complete=complete)
        finish_request(runtime, selection, complete=complete)

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


handle_responses = Pipeline()


def _request_effort(body: dict[str, Any]) -> str | None:
    """``reasoning.effort`` on the Responses body.

    An omitted effort stays omitted so the registry row can decide.
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
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    required = required_function(prepared.get("tool_choice"))
    thinking, card = mode_card(selection, _request_effort(prepared))
    # A named call is the whole completion. The closed think is the template's
    # empty reasoning turn, encoded with the prompt so the newline stays one token.
    # A :thinking id would otherwise keep the block open around that call.
    if required:
        thinking = False
    temperature, top_p, top_k = _sampling(prepared, thinking, card)
    presence = _presence(prepared, thinking, card)
    frequency = _frequency(prepared)
    messages, _ = template_messages(prepared)
    try:
        keep_thinking = preserve_thinking(prepared)
    except RuntimeError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    extra = format_instruction(prepared)
    if extra:
        messages = [{"role": "system", "content": extra}, *messages]
    search, note = web_search_context(prepared)
    if note:
        messages.append({"role": "user", "content": note})
    tools = function_tools(prepared.get("tools"))
    try:
        prompt = encode_chat(
            runtime.root,
            messages,
            tools,
            thinking,
            preserve_thinking=keep_thinking,
        )
        prompt = _fit_prompt(runtime, prepared, messages, tools, thinking, keep_thinking, prompt)
        prompt = prepare_prompt(runtime, messages, prompt)
    except RequestError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    forced, stop_after_forced = _forced_call(runtime.root, tools, required)
    close, end = think_close_ids(runtime.root) if thinking else ([], None)
    forbid = None if thinking else instruct_forbid(runtime.root)
    emit_tools = prepared.get("tool_choice") != "none"
    if not emit_tools:
        blocked = tool_call_forbid(runtime.root)
        forbid = blocked if forbid is None else forbid | blocked
    stream = bool(prepared.get("stream", False))
    mtp_body = _mtp_body(prepared)
    mtp = _mtp_flag(mtp_body, selection)
    mtp_depth = _mtp_depth(mtp_body, selection)
    # Grow happens inside generate. A cap taken from the previous cache is 0
    # once the prompt is longer than that cache, and decode never samples.
    requested = requested_limit(prepared)
    limit = runtime._context_limit if requested is None else requested
    if stream:
        _stream(
            reply,
            runtime,
            prompt,
            forced,
            stop_after_forced,
            limit,
            temperature,
            top_p,
            top_k,
            presence,
            thinking,
            close,
            end,
            emit_tools,
            forbid,
            frequency=frequency,
            mtp=mtp,
            mtp_depth=mtp_depth,
            prepared=prepared,
            state=state,
            search=search,
        )
        return
    tokens = runtime.generate(
        prompt,
        limit,
        temperature,
        top_p,
        top_k,
        presence,
        STOP,
        forced=forced,
        stop_after_forced=stop_after_forced,
        think_close=close,
        think_end=end,
        thinking_budget=thinking_token_budget(thinking),
        forbid=forbid,
        frequency=frequency,
        mtp=mtp,
        mtp_depth=mtp_depth,
    )
    text = decode_text(runtime.root, tokens)
    reasoning, visible = completion_fields(text, thinking)
    content, calls = parse_tool_calls(visible)
    if not emit_tools:
        # Keep the drawn text, including a mistaken tool block, so the next
        # prompt that replays this turn still matches the stored tokens.
        content = decode_shown_text(visible if thinking else text)
        calls = []
    elif not thinking:
        content = _shown_reply(text, content, calls)
    if reasoning:
        reasoning = decode_unicode_escapes(reasoning)
    produced = getattr(runtime, "_output_ids", None)
    completion_tokens = len(produced) if isinstance(produced, list) else len(tokens)
    payload = _response_payload(
        reply,
        runtime,
        prepared,
        search,
        reasoning if thinking else "",
        content,
        calls,
        len(prompt),
        completion_tokens,
        bool(getattr(runtime, "_hit_limit", False)),
    )
    reply.json(200, payload)


def _shown_reply(text: str, content: str | None, calls: list[dict[str, Any]]) -> str | None:
    """Reply text with a tool call removed. Text after the call stays."""
    if not calls:
        return decode_shown_text(text)
    if content:
        return decode_shown_text(content)
    end = text.rfind("</tool_call>")
    trailing = text[end + len("</tool_call>") :].strip() if end >= 0 else ""
    return decode_shown_text(trailing) if trailing else None


def _stream_delta(decoded: str, shown: str, *, final: bool) -> tuple[str, str]:
    """Text that is safe to send, and the piece after ``shown``.

    The tokenizer writes U+FFFD when a token is only some of the bytes of a
    character. That mark is held back. The next token replaces it, and the
    slice is the new suffix.
    """
    if not final and decoded.endswith("\ufffd"):
        decoded = decoded[:-1]
    return decoded[len(shown) :], decoded


def _context_length(runtime: Any) -> int:
    live = getattr(runtime, "_context_limit", None)
    if isinstance(live, int) and live > 0:
        return min(live, DEFAULT_CONTEXT_LENGTH)
    return DEFAULT_CONTEXT_LENGTH


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


def _mtp_body(prepared: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if "favlo:mtp" in prepared:
        body["mtp"] = prepared["favlo:mtp"]
    if "favlo:mtp_depth" in prepared:
        body["mtp_depth"] = prepared["favlo:mtp_depth"]
    return body


def _fit_prompt(
    runtime: Any,
    prepared: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: bool,
    keep_thinking: bool,
    prompt: list[int],
) -> list[int]:
    """The opening cache is not the prompt limit. The served window is."""
    ceiling = getattr(runtime, "_context_limit", None)
    if not isinstance(ceiling, int) or ceiling < 1:
        ceiling = DEFAULT_CONTEXT_LENGTH
    if len(prompt) <= ceiling:
        return prompt
    if prepared.get("truncation") != "auto":
        raise RequestError("input does not fit")
    kept = list(messages)
    while len(prompt) > ceiling and len(kept) > 1:
        drop = 1 if kept and kept[0].get("role") == "system" else 0
        if drop >= len(kept):
            break
        del kept[drop]
        prompt = encode_chat(
            runtime.root,
            kept,
            tools,
            thinking,
            preserve_thinking=keep_thinking,
        )
    if len(prompt) > ceiling:
        raise RequestError("input does not fit")
    messages[:] = kept
    return prompt


def _response_payload(
    reply: Any,
    runtime: Any,
    prepared: dict[str, Any],
    search: dict[str, Any] | None,
    reasoning: str,
    content: str | None,
    calls: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    hit_limit: bool,
    response_id: str | None = None,
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
        }
        summary = prepared.get("reasoning")
        if isinstance(summary, dict) and summary.get("summary"):
            item["summary"] = [{"type": "summary_text", "text": reasoning}]
        item["encrypted_content"] = reasoning
        output.append(item)
    annotations: list[dict[str, Any]] = []
    visible = content or ""
    if calls:
        if visible:
            output.append(_message_item(reply, visible, annotations))
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
        output.append(_message_item(reply, visible, annotations))
        if hit_limit:
            status = "incomplete"
            incomplete = "max_output_tokens"
        else:
            status = "completed"
            incomplete = None
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    payload = response_object(
        model=getattr(reply, "model_id", "") or prepared.get("model") or "",
        output=output,
        usage=usage_object(prompt_tokens, completion_tokens, cached, 0),
        body=prepared,
        status=status,
        incomplete=incomplete,
        response_id=response_id or new_id("resp_"),
    )
    chat_usage = _usage(runtime, prompt_tokens, completion_tokens)
    chat_usage["context_length"] = _context_length(runtime)
    payload["_chat_usage"] = chat_usage
    return payload


def _message_item(reply: Any, text: str, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    del reply
    return {
        "type": "message",
        "id": new_id("msg_"),
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": annotations,
            }
        ],
    }


def _stream(
    reply: Any,
    runtime: Any,
    prompt: list[int],
    forced: list[int],
    stop_after_forced: bool,
    limit: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    thinking: bool,
    think_close: list[int],
    think_end: int | None,
    emit_tools: bool = True,
    forbid: set[int] | None = None,
    frequency: float = 0.0,
    *,
    mtp: bool = False,
    mtp_depth: int | None = None,
    prepared: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    search: dict[str, Any] | None = None,
) -> None:
    decode = getattr(runtime, "decode", None)
    if not callable(decode):
        decode = load_tokenizer(runtime.root).decode
    request = prepared if isinstance(prepared, dict) else {"model": getattr(reply, "model_id", "") or ""}
    sequence = 0
    try:
        reply.begin_events()
    except OSError:
        raise RequestClosed() from None

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
        model=getattr(reply, "model_id", "") or request.get("model") or "",
        output=[],
        usage=usage_object(0, 0, 0, 0),
        body=request,
        status="in_progress",
        incomplete=None,
        response_id=response_id,
    )
    event({"type": "response.created", "response": shell})
    event({"type": "response.in_progress", "response": shell})
    pending: queue.Queue[int | None] = queue.Queue()
    failed: list[BaseException] = []
    text = ""

    def write() -> None:
        nonlocal text
        produced: list[int] = []
        previous = ""
        gate = StreamText(thinking, keep_calls=not emit_tools)
        opened: set[str] = set()

        def emit_piece(channel: str, visible: str) -> None:
            if not visible:
                return
            if channel not in opened:
                opened.add(channel)
                _open_item(event, channel, reasoning_id, message_id)
            _emit_response(event, channel, visible, reasoning_id, message_id)

        try:
            while True:
                token = pending.get()
                final = token is None
                if not final:
                    produced.append(int(token))
                decoded = decode(produced) if produced else ""
                piece, previous = _stream_delta(decoded, previous, final=final)
                for channel, visible in gate.push(piece):
                    emit_piece(channel, visible)
                if final:
                    for channel, visible in gate.finish():
                        emit_piece(channel, visible)
                    break
        except BaseException as exc:
            failed.append(exc)
        text = previous

    writer = threading.Thread(target=write, name="favlo-stream", daemon=True)
    writer.start()

    def enqueue(token: int) -> None:
        if failed:
            raise RequestClosed()
        pending.put(token)

    try:
        runtime.generate(
            prompt,
            limit,
            temperature,
            top_p,
            top_k,
            presence,
            STOP,
            forced=forced,
            stop_after_forced=stop_after_forced,
            on_token=enqueue,
            think_close=think_close,
            think_end=think_end,
            thinking_budget=thinking_token_budget(thinking),
            forbid=forbid,
            frequency=frequency,
            mtp=mtp,
            mtp_depth=mtp_depth,
        )
    finally:
        pending.put(None)
        writer.join()
    if failed:
        raise failed[0]
    reasoning, visible = completion_fields(text, thinking)
    content, calls = parse_tool_calls(visible)
    if not emit_tools:
        content = decode_shown_text(visible if thinking else text)
        calls = []
    elif not thinking:
        content = _shown_reply(text, content, calls)
    if reasoning:
        reasoning = decode_unicode_escapes(reasoning)
    produced = getattr(runtime, "_output_ids", None)
    completion_tokens = len(produced) if isinstance(produced, list) else 0
    payload = _response_payload(
        reply,
        runtime,
        request,
        search,
        reasoning if thinking else "",
        content,
        calls if emit_tools else [],
        len(prompt),
        completion_tokens,
        bool(getattr(runtime, "_hit_limit", False)),
        response_id,
    )
    for index, item in enumerate(payload["output"]):
        kind = item.get("type")
        if kind == "reasoning":
            item["id"] = reasoning_id
            event({"type": "response.output_item.done", "output_index": index, "item": item})
        elif kind == "function_call":
            arguments = item.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            item_id = item.get("id") or ""
            opened = dict(item)
            opened["arguments"] = ""
            event({"type": "response.output_item.added", "output_index": index, "item": opened})
            event(
                {
                    "type": "response.function_call_arguments.done",
                    "arguments": arguments,
                    "item_id": item_id,
                    "output_index": index,
                }
            )
            event({"type": "response.output_item.done", "output_index": index, "item": item})
    event({"type": "response.completed", "response": payload})
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
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
        index = 1
    event({"type": "response.output_item.added", "output_index": index, "item": item})


def _emit_response(event: Any, channel: str, visible: str, reasoning_id: str, message_id: str) -> None:
    if not visible:
        return
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

