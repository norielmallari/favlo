"""Chat template and one completion.

The registry stores the thinking and instruct card. A request field overrides
that card. Temperature 0 is an argmax. Presence, frequency, and logit bias
use the OpenAI logit update.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment
from tokenizers import Tokenizer

from qwen38_27b.cache import DEFAULT_CONTEXT_LENGTH
from qwen38_27b.forward import RequestClosed
from qwen38_27b.sampling import VARIANTS, mode_card, requested_thinking
from qwen38_27b.tools import (
    StreamText,
    function_has_parameters,
    normalize_messages,
    parse_tool_calls,
    required_function,
    split_thinking,
)

STOP = {248046, 248044}


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))


_CHECKOUT_EMPTY_THINK = (
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content"
    " + '\\n</think>\\n\\n' + content }}"
)
_EMPTY_THINK = """{%- if reasoning_content %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
            {%- elif enable_thinking is defined and enable_thinking is false %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n</think>\\n\\n' + content }}
            {%- endif %}"""


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    tools: Any,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
) -> list[int]:
    """Render the checkpoint template and encode it."""
    template = (root / "chat_template.jinja").read_text()
    if _CHECKOUT_EMPTY_THINK not in template:
        message = "chat template has no empty think block to align"
        raise ValueError(message)
    template = template.replace(_CHECKOUT_EMPTY_THINK, _EMPTY_THINK, 1)
    env = Environment()
    env.globals["raise_exception"] = _raise
    rendered_args = {
        "messages": normalize_messages(messages),
        "tools": tools or None,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        "add_vision_id": False,
        "preserve_thinking": True,
    }
    if reasoning_effort:
        rendered_args["reasoning_effort"] = reasoning_effort
    rendered = env.from_string(template).render(**rendered_args)
    return list(load_tokenizer(root).encode(rendered).ids)


def encode_text(root: Path, text: str) -> list[int]:
    return list(load_tokenizer(root).encode(text).ids)


def decode_text(root: Path, ids: list[int]) -> str:
    return load_tokenizer(root).decode(ids, skip_special_tokens=False)


def _raise(message: str) -> None:
    raise ValueError(message)


def _reasoning_effort(body: dict[str, Any]) -> str | None:
    """Template effort. ``none`` is omitted, so the template keeps xhigh."""
    effort = body.get("reasoning_effort")
    if not isinstance(effort, str):
        extra = body.get("chat_template_kwargs")
        if isinstance(extra, dict):
            effort = extra.get("reasoning_effort")
    if not isinstance(effort, str) or effort.strip().lower() == "none":
        return None
    return effort


def _sampling(
    body: dict[str, Any], thinking: bool, variant: dict[str, Any] | None
) -> tuple[float, float, int]:
    """Card values unless the request sets the field. Temperature 0 is an argmax."""
    card = variant if variant is not None else VARIANTS["thinking" if thinking else "instruct"]
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    return (
        float(card["temperature"]) if temperature is None else float(temperature),
        float(card["top_p"]) if top_p is None else float(top_p),
        int(card["top_k"]) if top_k is None else int(top_k),
    )


def _presence(body: dict[str, Any], thinking: bool, variant: dict[str, Any] | None) -> float:
    penalty = body.get("presence_penalty")
    if penalty is not None:
        return float(penalty)
    card = variant if variant is not None else VARIANTS["thinking" if thinking else "instruct"]
    if "presence_penalty" in card:
        return float(card["presence_penalty"])
    return 0.0


def _frequency(body: dict[str, Any]) -> float:
    penalty = body.get("frequency_penalty")
    if penalty is None:
        return 0.0
    return float(penalty)


def _logit_bias(body: dict[str, Any]) -> dict[int, float]:
    raw = body.get("logit_bias")
    if not isinstance(raw, dict):
        return {}
    return {int(token): float(bias) for token, bias in raw.items()}


def _rng(body: dict[str, Any]) -> np.random.Generator:
    seed = body.get("seed")
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _limit(body: dict[str, Any]) -> int | None:
    """Request or settings cap. None means decode until a stop or the context ends."""
    from favlo.logs import output_token_limit

    return output_token_limit(body)


def _forced_call(root: Path, tools: Any, required: str | None) -> tuple[list[int], bool]:
    if not required:
        return [], False
    if function_has_parameters(tools, required):
        return encode_text(root, f"<tool_call>\n<function={required}>\n"), False
    text = f"<tool_call>\n<function={required}>\n</function>\n</tool_call>"
    return encode_text(root, text), True


def handle_chat(
    reply: Any,
    runtime: Any,
    body: dict[str, Any],
    selection: Any,
    args: Any,
    state: dict[str, Any],
) -> None:
    """One non-streaming or streaming chat completion. The next token is an argmax."""
    del args, state
    runtime._halted = False
    runtime._stop = reply.stopped
    try:
        _handle_chat(reply, runtime, body, selection)
    except RequestClosed:
        runtime._status("client closed")
    finally:
        runtime._stop = None


def _handle_chat(reply: Any, runtime: Any, body: dict[str, Any], selection: Any) -> None:
    messages = body["messages"]
    requested = requested_thinking(body)
    required = required_function(body.get("tool_choice"))
    if required and "enable_thinking" not in body and (
        selection is None or getattr(selection, "variant", None) is None
    ):
        requested = False
    thinking, card = mode_card(selection, requested)
    temperature, top_p, top_k = _sampling(body, thinking, card)
    presence = _presence(body, thinking, card)
    frequency = _frequency(body)
    bias = _logit_bias(body)
    draw = _rng(body)
    from qwen38_27b.vision import collect_images, expand_image_tokens, merged_grid, prepare_image

    images = collect_images(messages)
    prompt = encode_chat(
        runtime.root, messages, body.get("tools"), thinking, _reasoning_effort(body)
    )
    prepared = [prepare_image(blob) for blob in images]
    if prepared:
        counts = []
        for _patches, grid in prepared:
            height, width = merged_grid(grid)
            counts.append(height * width)
        prompt = expand_image_tokens(prompt, counts)
    runtime._vision_images = images
    runtime._vision_prepared = prepared
    emit_tools = body.get("tool_choice") != "none"
    forced, stop_after_forced = _forced_call(runtime.root, body.get("tools"), required)
    if bool(body.get("stream", False)):
        _stream(
            reply,
            runtime,
            prompt,
            forced,
            stop_after_forced,
            _limit(body),
            thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
            logit_bias=bias,
            rng=draw,
            emit_tools=emit_tools,
        )
        return
    tokens = runtime.generate(
        prompt,
        _limit(body),
        STOP,
        forced=forced,
        stop_after_forced=stop_after_forced,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        frequency=frequency,
        logit_bias=bias,
        rng=draw,
    )
    text = decode_text(runtime.root, tokens)
    reasoning, visible = split_thinking(text, thinking)
    content, calls = parse_tool_calls(visible if visible else text)
    if not emit_tools:
        calls = []
        content = text if not thinking else (visible if visible else text)
    if calls:
        reason = "tool_calls"
    else:
        reason = "length" if runtime._hit_limit else "stop"
        if not thinking:
            content = text
    reply.json(
        200,
        _completion(
            content,
            len(prompt),
            len(tokens),
            reason,
            calls,
            reasoning if thinking else "",
            runtime,
            reply.model_id,
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
    """The served window. The live cache length is not the model context."""
    window = getattr(runtime, "_limit", None)
    if isinstance(window, int) and window > 0:
        return min(window, DEFAULT_CONTEXT_LENGTH)
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
    reply: Any,
    choices: list[dict[str, Any]],
    created: int,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``chat.completion.chunk``. ``id``, ``object``, ``created``, and ``model`` are required."""
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


def _stream(
    reply: Any,
    runtime: Any,
    prompt: list[int],
    forced: list[int],
    stop_after_forced: bool,
    limit: int,
    thinking: bool,
    *,
    emit_tools: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence: float = 0.0,
    frequency: float = 0.0,
    logit_bias: dict[int, float] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    decode = getattr(runtime, "decode", None)
    if not callable(decode):
        tokenizer = load_tokenizer(runtime.root)

        def decode(ids: list[int]) -> str:
            return tokenizer.decode(ids, skip_special_tokens=False)

    created = int(time.time())
    try:
        reply.begin_events()
    except OSError:
        raise RequestClosed() from None

    def event(payload: dict[str, Any]) -> None:
        try:
            reply.event(payload)
        except OSError as exc:
            raise RequestClosed() from exc

    saw_tool = False

    def emit_pieces(pieces: list[tuple[str, Any]]) -> None:
        nonlocal saw_tool
        for channel, payload in pieces:
            if channel == "tool_start":
                saw_tool = True
                _emit_tool(event, reply, created, payload, started=True)
            elif channel == "tool_args":
                saw_tool = True
                _emit_tool(event, reply, created, payload, started=False)
            else:
                _emit(event, reply, channel, payload, created)

    event(
        _frame(
            reply,
            [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            created,
        )
    )
    produced: list[int] = []
    previous = ""
    gate = StreamText(thinking, tools=emit_tools)

    def on_token(token: int) -> None:
        nonlocal previous
        produced.append(int(token))
        decoded = decode(produced)
        piece, previous = _stream_delta(decoded, previous, final=False)
        emit_pieces(gate.push(piece))

    def on_prefill(_done: int) -> None:
        event(_frame(reply, [{"index": 0, "delta": {}, "finish_reason": None}], created))

    try:
        runtime.generate(
            prompt,
            limit,
            STOP,
            forced=forced,
            stop_after_forced=stop_after_forced,
            on_token=on_token,
            on_prefill=on_prefill,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
            logit_bias=logit_bias,
            rng=rng,
        )
        decoded = decode(produced) if produced else ""
        piece, previous = _stream_delta(decoded, previous, final=True)
        emit_pieces(gate.push(piece))
        emit_pieces(gate.finish())
        reason = "tool_calls" if saw_tool else ("length" if runtime._hit_limit else "stop")
        event(_frame(reply, [{"index": 0, "delta": {}, "finish_reason": reason}], created))
        completion_tokens = len(getattr(runtime, "_output_ids", None) or produced)
        event(_frame(reply, [], created, _usage(runtime, len(prompt), completion_tokens)))
        reply.done()
    except OSError as exc:
        raise RequestClosed() from exc


def _emit(event: Any, reply: Any, channel: str, visible: str, created: int) -> None:
    if not visible:
        return
    field = "reasoning_content" if channel == "reasoning" else "content"
    event(
        _frame(
            reply,
            [{"index": 0, "delta": {field: visible}, "finish_reason": None}],
            created,
        )
    )


def _emit_tool(event: Any, reply: Any, created: int, payload: dict[str, Any], *, started: bool) -> None:
    function: dict[str, Any] = {}
    if started:
        function["name"] = payload["name"]
        call: dict[str, Any] = {
            "index": payload["index"],
            "id": payload["id"],
            "type": "function",
            "function": function,
        }
    else:
        function["arguments"] = payload["arguments"]
        call = {"index": payload["index"], "function": function}
    event(
        _frame(
            reply,
            [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
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
    runtime: Any,
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
        "usage": _usage(runtime, prompt_tokens, completion_tokens),
    }
