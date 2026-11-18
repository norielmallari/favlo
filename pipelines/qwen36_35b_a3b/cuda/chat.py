"""Responses generation for the CUDA runtime."""

from __future__ import annotations

import json
import queue
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from favlo.responses import (
    RequestError,
    function_tools,
    new_id,
    prepare_request,
    response_object,
    template_messages,
    usage_object,
    web_search_context,
)
from qwen36_35b_a3b.cuda.closed import RequestClosed
from qwen36_35b_a3b.cuda.layout import CONTEXT_CAP, EOS, IM_END
from qwen36_35b_a3b.cuda.sampling import mode_card, resolve_effort, thinking_on

STOP = {EOS, IM_END}
_THINK_END = "</think>"
_GENERATION = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}")
_TOOL = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


class Pipeline(LanguagePipeline):
    """One Responses generation. The sampler and the token loop stay here."""

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        super().on_open(runtime, reply, selection)
        runtime._halted = False

    def main(self, argv: Sequence[str]) -> int:
        from qwen36_35b_a3b.cuda.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from favlo.pipelines import bind_open_runtime
        from qwen36_35b_a3b.cuda.runtime import open_runtime as open_rt

        return bind_open_runtime(open_rt, path, **kwargs)

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
        prepared = prepare_request(body, state)
        if _has_image(prepared):
            raise RequestError("an image part is not served by this CUDA opening")
        card = mode_card(getattr(selection, "variant", None), getattr(selection, "sampling", None))
        effort = resolve_effort(_effort(prepared))
        if effort is None:
            effort = resolve_effort(card.get("reasoning_effort"))
        thinking = thinking_on(effort)
        messages, _saw = template_messages(prepared)
        search, note = web_search_context(prepared)
        if note:
            messages.append({"role": "user", "content": note})
        tools = function_tools(prepared.get("tools"))
        prompt = encode_chat(runtime.root, messages, tools, thinking)
        if len(prompt) > CONTEXT_CAP:
            raise RequestError("prompt is past the context window")
        temperature = float(prepared.get("temperature", card["temperature"]))
        top_p = float(prepared.get("top_p", card["top_p"]))
        top_k = int(card["top_k"])
        presence = float(prepared.get("presence_penalty", card["presence_penalty"]))
        frequency = float(prepared.get("frequency_penalty", 0.0) or 0.0)
        limit = _limit(prepared, len(prompt))
        if prepared.get("stream"):
            _stream(
                reply, runtime, prepared, prompt, limit, thinking, search,
                temperature, top_p, top_k, presence, frequency,
            )
            return
        tokens = runtime.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
            limit=limit,
            on_token=None,
        )
        reply.json(200, _object(reply, prepared, prompt, tokens, runtime, thinking, search, limit))


pipeline = Pipeline()
handle_responses = pipeline


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
    thinking only through ``enable_thinking``.
    """
    del reasoning_effort
    messages = _template_messages(messages)
    template_path = root / "chat_template.jinja"
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if not template_path.is_file():
        text = "\n".join(str(message.get("content") or "") for message in messages)
        return list(tokenizer.encode(text).ids)
    env = Environment(autoescape=False)
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(str(message)))
    source = _GENERATION.sub("", template_path.read_text(encoding="utf-8"))
    text = env.from_string(source).render(
        messages=messages,
        tools=tools or None,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        add_vision_id=False,
        preserve_thinking=preserve_thinking,
    )
    return list(tokenizer.encode(text).ids)


def _stream(
    reply: Any,
    runtime: Any,
    prepared: dict[str, Any],
    prompt: list[int],
    limit: int,
    thinking: bool,
    search: dict[str, Any] | None,
    temperature: float,
    top_p: float,
    top_k: int,
    presence: float,
    frequency: float,
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
        except OSError as err:
            raise RequestClosed() from err

    response_id = new_id("resp_")
    created = response_object(
        model=getattr(reply, "model_id", "") or prepared.get("model") or "",
        output=[],
        usage=usage_object(0, 0, 0, 0),
        body=prepared,
        status="in_progress",
        incomplete=None,
        response_id=response_id,
    )
    event({"type": "response.created", "response": created})
    pending: queue.Queue[int | None] = queue.Queue()
    failed: list[BaseException] = []
    produced: list[int] = []

    def produce() -> None:
        try:
            runtime.generate(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence=presence,
                frequency=frequency,
                limit=limit,
                on_token=pending.put,
            )
        except BaseException as exc:
            failed.append(exc)
        finally:
            pending.put(None)

    worker = threading.Thread(target=produce, name="qwen36-cuda-decode")
    worker.start()
    tokenizer = Tokenizer.from_file(str(runtime.root / "tokenizer.json"))
    shown = ""
    try:
        while True:
            token = pending.get()
            if token is None:
                break
            produced.append(int(token))
            text = tokenizer.decode(produced)
            delta = text[len(shown) :]
            shown = text
            if not delta:
                continue
            event({"type": "response.output_text.delta", "delta": delta})
        worker.join()
        if failed:
            raise failed[0]
    except OSError as exc:
        runtime._halted = True
        worker.join()
        raise RequestClosed() from exc
    finished = _object(reply, prepared, prompt, produced, runtime, thinking, search, limit)
    for index, item in enumerate(finished.get("output") or []):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        opened = dict(item)
        opened["arguments"] = ""
        event({"type": "response.output_item.added", "output_index": index, "item": opened})
        event(
            {
                "type": "response.function_call_arguments.done",
                "arguments": item.get("arguments") or "{}",
                "item_id": item.get("id") or "",
                "output_index": index,
            }
        )
        event({"type": "response.output_item.done", "output_index": index, "item": item})
    event({"type": "response.completed", "response": finished})
    try:
        reply.done()
    except OSError as exc:
        raise RequestClosed() from exc


def _object(
    reply: Any,
    prepared: dict[str, Any],
    prompt: list[int],
    tokens: list[int],
    runtime: Any,
    thinking: bool,
    search: dict[str, Any] | None,
    limit: int,
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(runtime.root / "tokenizer.json"))
    text = tokenizer.decode(tokens)
    reasoning, content, calls = _split(text, thinking)
    output: list[dict[str, Any]] = []
    if search is not None:
        output.append(search)
    if reasoning:
        output.append(
            {
                "id": new_id("rs_"),
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": reasoning}],
            }
        )
    for call in calls:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        output.append(
            {
                "type": "function_call",
                "id": new_id("fc_"),
                "call_id": call.get("id") or new_id("call_"),
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )
    if content or not calls:
        output.append(
            {
                "id": new_id("msg_"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    hit_limit = len(tokens) >= limit and not calls
    status = "incomplete" if hit_limit else "completed"
    usage = usage_object(len(prompt), len(tokens), int(getattr(runtime, "_cached_tokens", 0) or 0),
        0)
    usage["prompt_tokens"] = len(prompt)
    usage["completion_tokens"] = len(tokens)
    usage["total_tokens"] = len(prompt) + len(tokens)
    return response_object(
        model=getattr(reply, "model_id", "") or prepared.get("model") or "",
        output=output,
        usage=usage,
        body=prepared,
        status=status,
        incomplete="max_output_tokens" if hit_limit else None,
        response_id=new_id("resp_"),
    )


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The template walks ``arguments`` with ``|items``, so keep a mapping."""
    shaped: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        copied = dict(message)
        calls = copied.get("tool_calls")
        if isinstance(calls, list):
            copied["tool_calls"] = [
                _template_tool_call(call) for call in calls if isinstance(call, dict)
            ]
        shaped.append(copied)
    return shaped


def _template_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments") if isinstance(function, dict) else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    name = ""
    if isinstance(function, dict):
        name = str(function.get("name") or "")
    return {
        "id": call.get("id"),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split Qwen ``<tool_call>`` markup out of the visible reply."""
    calls: list[dict[str, Any]] = []
    for match in _TOOL.finditer(text):
        arguments: dict[str, Any] = {}
        for key, value in _PARAM.findall(match.group(2)):
            arguments[key.strip()] = _coerce(value.strip())
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": match.group(1).strip(),
                    "arguments": json.dumps(arguments, separators=(",", ":"), ensure_ascii=False),
                },
            }
        )
    return _TOOL.sub("", text).strip(), calls


def _coerce(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _split(text: str, thinking: bool) -> tuple[str, str, list[dict[str, Any]]]:
    if not thinking or _THINK_END not in text:
        visible, calls = parse_tool_calls(text)
        return "", visible, calls
    reasoning, _, content = text.partition(_THINK_END)
    visible, calls = parse_tool_calls(content.lstrip("\n"))
    return reasoning, visible, calls


def _effort(body: dict[str, Any]) -> str | None:
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str) or not effort.strip():
        return None
    return effort.strip().lower()


def _limit(body: dict[str, Any], prompt: int) -> int:
    raw = body.get("max_output_tokens")
    room = max(1, CONTEXT_CAP - prompt)
    if raw is None:
        return room
    return max(1, min(int(raw), room))


def _has_image(body: dict[str, Any]) -> bool:
    raw = body.get("input")
    if not isinstance(raw, list):
        return False
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"input_image", "image_url"}:
                return True
    return False
