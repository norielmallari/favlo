"""Chat template, markup, and one completion."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.usage import complete_chat_usage

from spark_x26_1_7b.closed import RequestClosed
from spark_x26_1_7b.layout import EOS
from spark_x26_1_7b.sampling import VARIANTS

STOP = {EOS}
_TOOL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
_THINK_END = "</think>"
_TOOL_MARK = "<tool_call>"


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))


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


class Pipeline(LanguagePipeline):
    """One completion. The sampler and the markup stay in this pipeline."""

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        del selection
        runtime._halted = False
        runtime._stop = reply.stopped

    def complete(
        self,
        reply: Any,
        runtime: Any,
        body: dict[str, Any],
        selection: Any,
        args: Any,
        state: dict[str, Any],
    ) -> None:
        del args, state
        _handle_chat(reply, runtime, body, selection)


handle_chat = Pipeline()


def _handle_chat(reply: Any, runtime: Any, body: dict[str, Any], selection: Any) -> None:
    thinking, effort, card = _mode(selection, body)
    temperature = _number(body, "temperature", card["temperature"])
    top_p = _number(body, "top_p", card["top_p"])
    top_k = int(body.get("top_k", card["top_k"]))
    if top_k < 0:
        top_k = 0
    presence = _number(body, "presence_penalty", card["presence_penalty"])
    frequency = _number(body, "frequency_penalty", 0.0)
    seed = body.get("seed")
    seed_value = int(seed) if isinstance(seed, int) else None
    prompt = encode_chat(
        runtime.root,
        body["messages"],
        body.get("tools"),
        thinking,
        effort,
    )
    limit = _limit(body)
    if bool(body.get("stream", False)):
        _stream(
            reply,
            runtime,
            prompt,
            limit,
            thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
            seed=seed_value,
        )
        return
    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        frequency=frequency,
        seed=seed_value,
    )
    text = decode_text(runtime.root, tokens)
    reasoning, visible = split_thinking(text, thinking)
    content, calls = parse_tool_calls(visible)
    reason = "tool_calls" if calls else ("length" if runtime._hit_limit else "stop")
    payload = _completion(
        content,
        len(prompt),
        len(tokens),
        reason,
        calls,
        reasoning if thinking or reasoning else "",
        runtime,
        reply.model_id,
    )
    complete_chat_usage(payload)
    reply.json(200, payload)


def _stream(
    reply: Any,
    runtime: Any,
    prompt: list[int],
    limit: int | None,
    thinking: bool,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    frequency: float,
    seed: int | None,
) -> None:
    created = int(time.time())
    try:
        reply.begin_events()
    except OSError:
        raise RequestClosed() from None
    produced: list[int] = []
    shown_reason = ""
    shown_content = ""

    def event(payload: dict[str, Any]) -> None:
        try:
            reply.event(payload)
        except OSError as exc:
            raise RequestClosed() from exc

    def emit(field: str, text: str) -> None:
        if text:
            event(_frame(reply, [{"index": 0, "delta": {field: text}, "finish_reason": None}], created))

    def on_token(token: int) -> None:
        nonlocal shown_reason, shown_content
        produced.append(int(token))
        text = decode_text(runtime.root, produced)
        if text.endswith("\ufffd"):
            text = text[:-1]
        reasoning, visible = split_thinking(text, thinking)
        if reasoning.startswith(shown_reason):
            emit("reasoning_content", reasoning[len(shown_reason) :])
        shown_reason = reasoning
        content, _calls = parse_tool_calls(visible)
        if content.startswith(shown_content):
            emit("content", content[len(shown_content) :])
        shown_content = content

    event(_frame(reply, [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}], created))
    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        on_token=on_token,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        frequency=frequency,
        seed=seed,
    )
    text = decode_text(runtime.root, tokens)
    reasoning, visible = split_thinking(text, thinking)
    content, calls = parse_tool_calls(visible)
    if calls:
        for call in calls:
            event(
                _frame(
                    reply,
                    [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
                    created,
                )
            )
    reason = "tool_calls" if calls else ("length" if runtime._hit_limit else "stop")
    event(_frame(reply, [{"index": 0, "delta": {}, "finish_reason": reason}], created))
    usage = _usage(runtime, len(prompt), len(tokens))
    payload = _frame(reply, [], created, usage)
    complete_chat_usage(payload)
    event(payload)
    reply.done()


def _mode(selection: Any, body: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    variant = getattr(selection, "variant", None) if selection is not None else None
    if variant in VARIANTS:
        thinking = variant != "none"
        effort = "low" if variant == "low" else ("high" if thinking else None)
        return thinking, effort, dict(VARIANTS[variant])
    enabled = _flag(body, "enable_thinking")
    if enabled is False:
        return False, None, dict(VARIANTS["none"])
    effort = _text(body, "reasoning_effort") or "high"
    if effort not in ("high", "low"):
        message = "reasoning_effort must be high or low"
        raise RuntimeError(message)
    return True, effort, dict(VARIANTS[effort])


def _flag(body: dict[str, Any], name: str) -> bool | None:
    if name in body:
        value = body.get(name)
    else:
        kwargs = body.get("chat_template_kwargs")
        if not isinstance(kwargs, dict) or name not in kwargs:
            return None
        value = kwargs.get(name)
    if isinstance(value, bool):
        return value
    message = f"{name} must be true or false"
    raise RuntimeError(message)


def _text(body: dict[str, Any], name: str) -> str | None:
    if name in body and isinstance(body.get(name), str):
        return str(body[name])
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and isinstance(kwargs.get(name), str):
        return str(kwargs[name])
    return None


def _number(body: dict[str, Any], name: str, default: float) -> float:
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{name} must be a number"
        raise RuntimeError(message)
    return float(value)


def _limit(body: dict[str, Any]) -> int | None:
    for name in ("max_completion_tokens", "max_tokens"):
        value = body.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _usage(runtime: Any, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    cached = int(getattr(runtime, "_cached_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "context_length": int(getattr(runtime, "limit", 0) or 0),
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def _completion(
    content: str,
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
