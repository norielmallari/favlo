"""Chat completion for Gemma 4 26B-A4B."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.pipeline import ClientClosed as RequestClosed
from gemma4_26b_a4b.sampling import mode_card, requested_thinking
from gemma4_26b_a4b.tools import StreamText, parse_tool_calls
from gemma4_26b_a4b.vision import collect_images

_TEMPLATE = Path(__file__).with_name("chat_template.jinja")
_THOUGHT = "<|channel>thought"
_THOUGHT_CLOSE = "<channel|>"
_TOOL_OPEN = "<|tool_call>"


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
        rendered = render_chat(messages, thinking, tools)
        prompt = list(load_tokenizer(runtime.root).encode(rendered).ids)
        pictures = collect_images(messages)
        stops = _stop_strings(body)
        channel_open = rendered.endswith("<|channel>thought\n")
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
        emit_tools = body.get("tool_choice") != "none"
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
                emit_tools=emit_tools,
                channel_open=channel_open,
                images=pictures,
                stops=stops,
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
        pending: queue.Queue[int | None] = queue.Queue()
        failed: list[BaseException] = []
        produced: list[int] = []
        sent = {"reasoning": "", "content": ""}

        def write() -> None:
            gate = StreamText(channel_open=channel_open)
            shown = ""
            try:
                while True:
                    token = pending.get()
                    final = token is None
                    if not final:
                        produced.append(int(token))
                    decoded = decode_text(runtime.root, produced) if produced else ""
                    piece, shown = _stream_delta(decoded, shown, final=final)
                    for channel, visible in gate.push(piece):
                        _emit(self._event, reply, channel, visible, created, sent)
                    if final:
                        for channel, visible in gate.finish():
                            _emit(self._event, reply, channel, visible, created, sent)
                        break
            except BaseException as exc:
                failed.append(exc)

        writer = threading.Thread(target=write, name="favlo-gemma-stream", daemon=True)
        writer.start()

        def enqueue(token: int) -> None:
            if failed:
                raise RequestClosed()
            pending.put(token)

        try:
            tokens, _prefill_line, _decode_line = runtime.generate(
                prompt,
                cap,
                temperature,
                top_p,
                top_k,
                presence,
                on_token=enqueue,
                images=pictures,
                stops=stops,
            )
        finally:
            pending.put(None)
            writer.join()
        if failed:
            raise failed[0]
        text = cut_stop(decode_text(runtime.root, tokens), stops)
        reasoning, content, calls = completion_fields(
            text, channel_open=channel_open, emit_tools=emit_tools
        )
        _emit_unsent(self._event, reply, created, sent, reasoning, content)
        if calls:
            for index, call in enumerate(calls):
                self._event(
                    reply,
                    _frame(
                        reply,
                        [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call["id"],
                                            "type": "function",
                                            "function": call["function"],
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                        created,
                    ),
                )
        self._event(
            reply,
            _frame(
                reply,
                [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": _finish_reason(runtime, calls),
                    }
                ],
                created,
                _usage(int(runtime.prompt_tokens), len(tokens)),
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
        channel_open: bool,
        images: list[bytes],
        stops: list[str],
    ) -> None:
        del thinking
        tokens, _prefill_line, _decode_line = runtime.generate(
            prompt,
            cap,
            temperature,
            top_p,
            top_k,
            presence,
            images=images,
            stops=stops,
        )
        text = cut_stop(decode_text(runtime.root, tokens), stops)
        reasoning, content, calls = completion_fields(
            text, channel_open=channel_open, emit_tools=emit_tools
        )
        reason = _finish_reason(runtime, calls)
        message = openai_message(reasoning, content, calls)
        reply.json(
            200,
            {
                "id": "chatcmpl-favlo",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": getattr(reply, "model_id", "") or "",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": _usage(int(runtime.prompt_tokens), len(tokens)),
            },
        )

    def _event(self, reply: Any, payload: dict[str, Any]) -> None:
        try:
            reply.event(payload)
        except OSError as exc:
            raise RequestClosed() from exc


def _stream_delta(decoded: str, shown: str, *, final: bool) -> tuple[str, str]:
    """Text that is safe to send, and the piece after ``shown``.

    The tokenizer writes U+FFFD when a token is only some of the bytes of a
    character. That mark is held back. The next token replaces it.
    """
    if not final and decoded.endswith("\ufffd"):
        decoded = decoded[:-1]
    return decoded[len(shown) :], decoded


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


handle_chat = Gemma4Chat()
