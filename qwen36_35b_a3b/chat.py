"""Chat template and one completion on the text forward."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from qwen36_35b_a3b.cache import DEFAULT_CONTEXT_LENGTH
from qwen36_35b_a3b.forward import RequestClosed, letter_token_ids, long_token_ids
from qwen36_35b_a3b.sampling import VARIANTS, mode_card, requested_thinking
from qwen36_35b_a3b.tools import (
    StreamText,
    function_has_parameters,
    normalize_messages,
    parse_tool_calls,
    required_function,
    split_thinking,
)
from qwen36_35b_a3b.vision import prepare_prompt

STOP = {248046, 248044}
_LONG_IDS: dict[Path, frozenset[int]] = {}


def _long_ids(root: Path) -> frozenset[int]:
    cached = _LONG_IDS.get(root)
    if cached is None:
        cached = long_token_ids(load_tokenizer(root).get_vocab())
        _LONG_IDS[root] = cached
    return cached


_WORD_IDS: dict[Path, frozenset[int]] = {}


def _word_ids(root: Path) -> frozenset[int]:
    cached = _WORD_IDS.get(root)
    if cached is None:
        cached = letter_token_ids(load_tokenizer(root).get_vocab())
        _WORD_IDS[root] = cached
    return cached


def _sampling(
    body: dict[str, Any], thinking: bool, variant: dict[str, Any] | None
) -> tuple[float, float, int]:
    card = variant if variant is not None else VARIANTS["thinking" if thinking else "instruct"]
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
    card = variant if variant is not None else VARIANTS["thinking" if thinking else "instruct"]
    if "presence_penalty" in card:
        return float(card["presence_penalty"])
    return 0.0 if thinking else 1.5


def _frequency(body: dict[str, Any]) -> float:
    penalty = body.get("frequency_penalty")
    if penalty is None:
        return 0.0
    return float(penalty)


def requested_limit(body: dict[str, Any]) -> int | None:
    """Request or settings cap. None when neither set one."""
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


def completion_limit(max_position: int, prompt_tokens: int, requested: int | None) -> int:
    """Tokens still available in the checkpoint context, cut by a request cap."""
    room = max(0, int(max_position) - int(prompt_tokens))
    if requested is None:
        return room
    return max(0, min(int(requested), room))


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def _bridge_omitted_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put an empty assistant turn where the previous prompt already closed thinking.

    A follow-up that omits that turn renders the next user in its place, so the
    stored tokens are no longer a prefix. The empty turn is the closed think.
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
) -> list[int]:
    """Render the checkpoint template and encode it.

    A passed reasoning effort is ignored. This checkpoint opens or closes
    thinking only through ``enable_thinking``.
    """
    del reasoning_effort
    template = (root / "chat_template.jinja").read_text()
    env = Environment()
    env.globals["raise_exception"] = _raise
    rendered_args: dict[str, Any] = {
        "messages": _bridge_omitted_assistant(normalize_messages(messages)),
        "tools": tools or None,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        "add_vision_id": False,
        "preserve_thinking": True,
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


def handle_chat(
    handler: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    """One non-streaming or streaming chat completion."""
    del args, state
    from qwen36_35b_a3b.firing import finish_request, open_request

    runtime._halted = False
    runtime._stop = getattr(handler, "_client_stopped", None)
    open_request(runtime)
    finished = False
    try:
        _handle_chat(handler, runtime, body, selection)
        finished = True
    except RequestClosed:
        handler.close_connection = True
        runtime._status("client closed")
    except OSError as exc:
        handler.close_connection = True
        if str(exc) != "client closed":
            raise
        runtime._status("client closed")
    finally:
        runtime._vision_at = None
        runtime._stop = None
        finish_request(runtime, selection, complete=finished)


def _handle_chat(handler: Any, runtime: Any, body: dict[str, Any], selection: Any) -> None:
    messages = body["messages"]
    requested = requested_thinking(body)
    required = required_function(body.get("tool_choice"))
    thinking, card = mode_card(selection, requested)
    # A named call is the whole completion. The closed think is the template's
    # empty reasoning turn, encoded with the prompt so the newline stays one token.
    # A :thinking id would otherwise keep the block open around that call.
    if required:
        thinking = False
    temperature, top_p, top_k = _sampling(body, thinking, card)
    presence = _presence(body, thinking, card)
    frequency = _frequency(body)
    prompt = encode_chat(runtime.root, messages, body.get("tools"), thinking)
    prompt = prepare_prompt(runtime, messages, prompt)
    forced, stop_after_forced = _forced_call(runtime.root, body.get("tools"), required)
    close, end = think_close_ids(runtime.root) if thinking else ([], None)
    tool_open = encode_text(runtime.root, "<tool_call>")[0] if thinking else None
    long_ids = _long_ids(runtime.root) if thinking else None
    word_ids = _word_ids(runtime.root) if thinking else None
    forbid = None if thinking else instruct_forbid(runtime.root)
    emit_tools = body.get("tool_choice") != "none"
    stream = bool(body.get("stream", False))
    # Grow happens inside generate. A cap taken from the previous cache is 0
    # once the prompt is longer than that cache, and decode never samples.
    requested = requested_limit(body)
    limit = runtime._context_limit if requested is None else requested
    if stream:
        _stream(
            handler,
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
            tool_open,
            emit_tools,
            long_ids,
            forbid,
            word_ids,
            frequency=frequency,
        )
        return
    started = time.perf_counter()
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
        tool_open=tool_open,
        long_ids=long_ids,
        forbid=forbid,
        word_ids=word_ids,
        frequency=frequency,
    )
    text = decode_text(runtime.root, tokens)
    reasoning, reply = split_thinking(text, thinking)
    content, calls = parse_tool_calls(reply if reply else text)
    if not emit_tools:
        calls = []
        content = reply if reply else text
    if calls:
        reason = "tool_calls"
    else:
        reason = "length" if runtime._hit_limit else "stop"
        if not thinking:
            content = text
    handler._json(
        200,
        _completion(
            content,
            len(prompt),
            len(tokens),
            reason,
            calls,
            reasoning if thinking else "",
            time.perf_counter() - started,
            handler._model_id,
        ),
    )


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


def _frame(
    handler: Any,
    choices: list[dict[str, Any]],
    created: int,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``chat.completion.chunk``. The id, object, created, and model fields are required."""
    payload: dict[str, Any] = {
        "id": "chatcmpl-favlo",
        "object": "chat.completion.chunk",
        "created": created,
        "model": getattr(handler, "_model_id", "") or "",
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _stream(
    handler: Any,
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
    tool_open: int | None = None,
    emit_tools: bool = True,
    long_ids: set[int] | None = None,
    forbid: set[int] | None = None,
    word_ids: set[int] | None = None,
    frequency: float = 0.0,
) -> None:
    decode = getattr(runtime, "decode", None)
    if not callable(decode):
        decode = load_tokenizer(runtime.root).decode
    created = int(time.time())
    handler._started = True
    handler.close_connection = True
    handler.connection.settimeout(2.0)
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
    except OSError:
        raise RequestClosed() from None

    def event(payload: dict[str, Any]) -> None:
        try:
            handler._event(payload)
        except OSError as exc:
            raise RequestClosed() from exc

    event(
        _frame(
            handler,
            [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            created,
        )
    )
    pending: queue.Queue[int | None] = queue.Queue()
    failed: list[BaseException] = []
    text = ""

    def write() -> None:
        nonlocal text
        produced: list[int] = []
        previous = ""
        gate = StreamText(thinking, keep_calls=not emit_tools)
        try:
            while True:
                token = pending.get()
                final = token is None
                if not final:
                    produced.append(int(token))
                decoded = decode(produced) if produced else ""
                piece, previous = _stream_delta(decoded, previous, final=final)
                for channel, visible in gate.push(piece):
                    _emit(event, handler, channel, visible, created)
                if final:
                    for channel, visible in gate.finish():
                        _emit(event, handler, channel, visible, created)
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
            tool_open=tool_open,
            long_ids=long_ids,
            forbid=forbid,
            word_ids=word_ids,
            frequency=frequency,
        )
    finally:
        pending.put(None)
        writer.join()
    if failed:
        raise failed[0]
    _reasoning, reply = split_thinking(text, thinking)
    _content, calls = parse_tool_calls(reply if reply else text)
    if emit_tools and calls:
        for index, call in enumerate(calls):
            event(
                _frame(
                    handler,
                    [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": call["function"]["name"],
                                            "arguments": call["function"]["arguments"],
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                    created,
                )
            )
        reason = "tool_calls"
    else:
        reason = "length" if runtime._hit_limit else "stop"
    event(_frame(handler, [{"index": 0, "delta": {}, "finish_reason": reason}], created))
    produced = getattr(runtime, "_output_ids", None)
    completion_tokens = len(produced) if isinstance(produced, list) else 0
    event(_frame(handler, [], created, _usage(runtime, len(prompt), completion_tokens)))
    try:
        handler._send(b"data: [DONE]\n\n")
    except OSError as exc:
        raise RequestClosed() from exc


def _emit(event: Any, handler: Any, channel: str, visible: str, created: int) -> None:
    if not visible:
        return
    field = "reasoning_content" if channel == "reasoning" else "content"
    event(
        _frame(
            handler,
            [{"index": 0, "delta": {field: visible}, "finish_reason": None}],
            created,
        )
    )


def _completion(
    content: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    tool_calls: list[dict[str, Any]],
    reasoning: str,
    seconds: float,
    model: str,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl-favlo",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "timings": {
                "predicted_per_second": 0.0 if seconds <= 0 else completion_tokens / seconds,
                "predicted_n": completion_tokens,
            },
        },
    }
