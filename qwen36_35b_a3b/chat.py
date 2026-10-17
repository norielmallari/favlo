"""Chat template and one completion on the text forward."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from qwen36_35b_a3b.sampling import VARIANTS, mode_card, requested_thinking
from qwen36_35b_a3b.tools import (
    function_has_parameters,
    normalize_messages,
    parse_tool_calls,
    required_function,
    split_thinking,
)

STOP = {248046, 248044}


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


def _limit(body: dict[str, Any]) -> int:
    if body.get("max_tokens") is not None:
        return int(body["max_tokens"])
    if body.get("max_completion_tokens") is not None:
        return int(body["max_completion_tokens"])
    return 2048


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    tools: Any,
    enable_thinking: bool,
) -> list[int]:
    """Render the checkpoint template and encode it."""
    template = (root / "chat_template.jinja").read_text()
    env = Environment()
    env.globals["raise_exception"] = _raise
    rendered = env.from_string(template).render(
        messages=normalize_messages(messages),
        tools=tools or None,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        add_vision_id=False,
        preserve_thinking=True,
    )
    return list(load_tokenizer(root).encode(rendered).ids)


def encode_text(root: Path, text: str) -> list[int]:
    return list(load_tokenizer(root).encode(text).ids)


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
    prompt = encode_chat(runtime.root, messages, body.get("tools"), thinking)
    forced, stop_after_forced = _forced_call(runtime.root, body.get("tools"), required)
    stream = bool(body.get("stream", False))
    if stream:
        _stream(
            handler,
            runtime,
            prompt,
            forced,
            stop_after_forced,
            _limit(body),
            temperature,
            top_p,
            top_k,
            presence,
            thinking,
        )
        return
    started = time.perf_counter()
    tokens = runtime.generate(
        prompt,
        _limit(body),
        temperature,
        top_p,
        top_k,
        presence,
        STOP,
        forced=forced,
        stop_after_forced=stop_after_forced,
    )
    text = decode_text(runtime.root, tokens)
    reasoning, reply = split_thinking(text, thinking)
    content, calls = parse_tool_calls(reply if reply else text)
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
) -> None:
    handler._started = True
    handler.close_connection = True
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler._event(
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    )
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
    )
    text = decode_text(runtime.root, tokens)
    reasoning, reply = split_thinking(text, thinking)
    content, calls = parse_tool_calls(reply if reply else text)
    if thinking and reasoning:
        handler._event(
            {
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}
                ]
            }
        )
    if content:
        handler._event(
            {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
        )
    reason = "tool_calls" if calls else ("length" if runtime._hit_limit else "stop")
    delta: dict[str, Any] = {}
    if calls:
        delta["tool_calls"] = calls
    handler._event({"choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})
    handler._send(b"data: [DONE]\n\n")


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
        "id": "chatcmpl-madmetal",
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
