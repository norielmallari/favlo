"""Chat completion for Gemma 4 26B-A4B."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.pipeline import ClientClosed as RequestClosed
from gemma4_26b_a4b.sampling import mode_card, requested_thinking
from gemma4_26b_a4b.tools import parse_tool_calls

_TEMPLATE = Path(__file__).with_name("chat_template.jinja")
_THOUGHT = "<|channel>thought"


def load_tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def render_chat(
    messages: list[dict[str, Any]],
    thinking: bool,
    tools: Any = None,
) -> str:
    """The canonical Gemma 4 template. Tool arguments are objects before render."""
    environment = Environment(autoescape=False, keep_trailing_newline=True)
    template = environment.from_string(_TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        messages=_messages(messages),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=thinking,
        preserve_thinking=False,
        bos_token="<bos>",
        eos_token="<eos>",
    )


def encode_chat(
    root: Path,
    messages: list[dict[str, Any]],
    thinking: bool,
    tools: Any = None,
) -> list[int]:
    """Render the canonical template and encode it, specials included."""
    text = render_chat(messages, thinking, tools)
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


def reply_text(text: str) -> tuple[str, str]:
    """Split a thought channel from the answer. The answer is what the carrot check reads."""
    marker = "<channel|>"
    if marker not in text:
        return "", text
    reasoning, answer = text.rsplit(marker, 1)
    if _THOUGHT in reasoning:
        reasoning = reasoning.rsplit(_THOUGHT, 1)[-1]
    return reasoning.strip("\n"), answer


class Gemma4Chat(LanguagePipeline):
    """One completion. The sampler and the stream stay in this package."""

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        del reply, selection
        runtime._stop = False

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
        thinking, card = mode_card(selection, requested_thinking(body))
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            reply.json(
                400,
                {"error": {"message": "messages are required", "type": "invalid_request_error"}},
            )
            return
        tools = body.get("tools")
        prompt = encode_chat(runtime.root, messages, thinking, tools)
        limit = body.get("max_tokens", body.get("max_completion_tokens"))
        cap = int(limit) if isinstance(limit, int) and limit > 0 else 4096
        if body.get("temperature") is not None:
            temperature = float(body["temperature"])
        else:
            temperature = float(card["temperature"])
        top_p = float(body["top_p"]) if body.get("top_p") is not None else float(card["top_p"])
        top_k = int(body["top_k"]) if body.get("top_k") is not None else int(card["top_k"])
        presence = (
            float(body["presence_penalty"])
            if body.get("presence_penalty") is not None
            else float(card["presence_penalty"])
        )
        if not bool(body.get("stream", False)):
            self._json(
                reply,
                runtime,
                prompt,
                cap,
                thinking,
                temperature,
                top_p,
                top_k,
                presence,
                emit_tools=body.get("tool_choice") != "none",
            )
            return
        created = int(time.time())
        try:
            reply.begin_events()
        except OSError as exc:
            raise RequestClosed() from exc
        self._event(
            reply,
            _frame(
                reply,
                [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                created,
            ),
        )
        tokens, prefill_line, decode_line = runtime.generate(
            prompt,
            cap,
            temperature,
            top_p,
            top_k,
            presence,
        )
        print(prefill_line, flush=True)
        print(decode_line, flush=True)
        text = decode_text(runtime.root, tokens)
        reasoning, answer = reply_text(text)
        if thinking and reasoning:
            self._event(
                reply,
                _frame(
                    reply,
                    [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": reasoning},
                            "finish_reason": None,
                        }
                    ],
                    created,
                ),
            )
        if answer:
            self._event(
                reply,
                _frame(
                    reply,
                    [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
                    created,
                ),
            )
        self._event(
            reply,
            _frame(
                reply,
                [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                created,
                _usage(len(prompt), len(tokens)),
            ),
        )
        try:
            reply.done()
        except OSError as exc:
            raise RequestClosed() from exc

    def _json(
        self,
        reply: Any,
        runtime: Any,
        prompt: list[int],
        cap: int,
        thinking: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        presence: float,
        *,
        emit_tools: bool,
    ) -> None:
        tokens, prefill_line, decode_line = runtime.generate(
            prompt,
            cap,
            temperature,
            top_p,
            top_k,
            presence,
        )
        print(prefill_line, flush=True)
        print(decode_line, flush=True)
        text = decode_text(runtime.root, tokens)
        reasoning, answer = reply_text(text)
        visible = answer if thinking else text
        content, calls = parse_tool_calls(visible)
        if not calls:
            content, calls = parse_tool_calls(text)
        if not emit_tools:
            calls = []
            content = visible or text
        reason = "tool_calls" if calls else "stop"
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if thinking and reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        reply.json(
            200,
            {
                "id": "chatcmpl-favlo",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": getattr(reply, "model_id", "") or "",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": _usage(len(prompt), len(tokens)),
            },
        )

    def _event(self, reply: Any, payload: dict[str, Any]) -> None:
        try:
            reply.event(payload)
        except OSError as exc:
            raise RequestClosed() from exc


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


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


handle_chat = Gemma4Chat()
