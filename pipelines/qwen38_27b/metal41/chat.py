"""Chat template and one completion.

The registry stores the thinking and instruct card. A request field overrides
that card. Temperature 0 is an argmax. Presence, frequency, and logit bias
use the OpenAI logit update.
"""

from __future__ import annotations

from collections.abc import Sequence

import json
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment
from tokenizers import Tokenizer

from favlo.language_pipeline import LanguagePipeline
from . import sampling as _sampling_mod
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
from qwen38_27b.metal41.cache import DEFAULT_CONTEXT_LENGTH
from qwen38_27b.metal41.forward import RequestClosed
from qwen38_27b.metal41.sampling import (
    DEFAULT_VARIANT,
    VARIANTS,
    mode_card,
    resolve_effort,
    template_effort,
)
from qwen38_27b.metal41.tools import (
    StreamText,
    function_has_parameters,
    normalize_messages,
    parse_tool_calls,
    required_function,
    split_thinking,
)

STOP = {248046, 248044}
THINK_END = 248069
TOOL_CALL_FORBID = {248058, 248059}


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
            {%- elif message.tool_calls %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n</think>\\n\\n' + content }}
            {%- endif %}"""


def preserve_thinking(body: dict[str, Any], registry: bool | None = None) -> bool:
    """Whether historical reasoning stays in the prompt.

    A top-level ``preserve_thinking`` wins over ``chat_template_kwargs``.
    Either wins over this pipeline's registry row. Omitted is off: only the
    think block for the latest user message stays. True keeps historical
    traces so a later prompt can extend the stored prefix.
    """
    flag = _preserve_flag(body.get("preserve_thinking")) if "preserve_thinking" in body else None
    if flag is None:
        kwargs = body.get("chat_template_kwargs")
        if isinstance(kwargs, dict) and "preserve_thinking" in kwargs:
            flag = _preserve_flag(kwargs.get("preserve_thinking"))
    if flag is not None:
        return flag
    if registry is None:
        return False
    return bool(registry)


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


def _preserve_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    message = "preserve_thinking must be true or false"
    raise RuntimeError(message)


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

    ``preserve_thinking`` false drops reasoning on assistant turns before the
    latest user message. True keeps every assistant think block.
    """
    template = (root / "chat_template.jinja").read_text()
    if _CHECKOUT_EMPTY_THINK not in template:
        message = "chat template has no empty think block to align"
        raise ValueError(message)
    template = template.replace(_CHECKOUT_EMPTY_THINK, _EMPTY_THINK, 1)
    prepared = normalize_messages(messages)
    if preserve_thinking:
        prepared = _bridge_omitted_assistant(prepared)
    env = Environment()
    env.globals["raise_exception"] = _raise
    rendered_args = {
        "messages": prepared,
        "tools": tools or None,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        "add_vision_id": False,
        "preserve_thinking": preserve_thinking,
    }
    if reasoning_effort:
        rendered_args["reasoning_effort"] = reasoning_effort
    rendered = env.from_string(template).render(**rendered_args)
    return list(load_tokenizer(root).encode(rendered).ids)


def encode_text(root: Path, text: str) -> list[int]:
    return list(load_tokenizer(root).encode(text).ids)


_GENERATION_SUFFIX: dict[Path, tuple[list[int], list[int]]] = {}


def think_closes(root: Path) -> tuple[list[int], list[int], int]:
    """Close sequences for an open think, and the ``<tool_call>`` token.

    Empty reasoning replays ``</think>\\n\\n``. Reasoning text replays
    ``\\n</think>\\n\\n``. A tool call drawn before that close is stored as
    the matching sequence, then sampled again.
    """
    opener = encode_text(root, "<tool_call>")
    if len(opener) != 1:
        raise RuntimeError("the tool call open is not one token")
    return encode_text(root, "</think>\n\n"), encode_text(root, "\n</think>\n\n"), opener[0]


def redirect_open_think(
    sampled: int,
    emitted: list[int],
    *,
    closed: bool,
    end: int | None,
    tool_open: int | None,
    empty_close: list[int],
    text_close: list[int],
) -> list[int] | None:
    """Tokens to store instead of ``sampled`` so the next prompt stays a prefix.

    None keeps ``sampled``. A bare ``</think>`` or a ``<tool_call>`` while the
    block is open becomes the template close. The tool token is not kept; the
    next draw happens after the close.
    """
    if closed or end is None or tool_open is None:
        return None
    if sampled != tool_open and sampled != end:
        return None
    close = list(empty_close if not emitted else text_close)
    if emitted and close and emitted[-1] == close[0] and len(close) > 1:
        close = close[1:]
    if not close:
        return None
    return close


def generation_cut(root: Path, prompt: list[int]) -> int | None:
    """Index where the generation suffix starts, when this prompt ends with one.

    Thinking leaves ``<think>\\n``. Instruct leaves the closed empty block.
    A later prompt that omits historical thinking does not repeat that suffix,
    so the tokens before it are the prefix the follow-up still shares.
    """
    pair = _GENERATION_SUFFIX.get(root)
    if pair is None:
        pair = (
            encode_text(root, "<think>\n\n</think>\n\n"),
            encode_text(root, "<think>\n"),
        )
        _GENERATION_SUFFIX[root] = pair
    for suffix in pair:
        count = len(suffix)
        if count and len(prompt) > count and prompt[-count:] == suffix:
            return len(prompt) - count
    return None


def user_turn_cut(
    root: Path,
    messages: list[dict[str, Any]],
    tools: Any,
    thinking: bool,
    effort: str | None = None,
) -> int | None:
    """Tokens through the latest user message, before the generation suffix.

    Historical thinking is omitted on the next user message, including the
    traces this turn kept after that message. The file that stops here is the
    one that message still shares. A prompt that already ends on the user
    message uses ``generation_cut``.
    """
    last = None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last = index
    if last is None or last + 1 == len(messages):
        return None
    head = encode_chat(
        root,
        messages[: last + 1],
        tools,
        thinking,
        effort,
        preserve_thinking=False,
    )
    return generation_cut(root, head)


def _expanded_cut(text_prompt: list[int], cut: int | None, counts: list[int]) -> int | None:
    """Map a text-token cut through the image pads expanded after the fit."""
    if cut is None:
        return None
    if not counts:
        return cut
    from qwen38_27b.metal41.vision import IMAGE_PAD

    extra = 0
    seen = 0
    for position, token in enumerate(text_prompt):
        if position >= cut:
            break
        if token != IMAGE_PAD:
            continue
        if seen >= len(counts):
            break
        extra += int(counts[seen]) - 1
        seen += 1
    return cut + extra


def decode_text(root: Path, ids: list[int]) -> str:
    return load_tokenizer(root).decode(ids, skip_special_tokens=False)


def keeps_token_ids(tokenizer: Tokenizer, produced: list[int], token: int) -> bool:
    """True when ``produced + [token]`` is the encoding of its own text.

    A later prompt encodes that text again. A split such as ``/P`` + ``LE``
    for ``/PLE`` is not that encoding, so the stored prefix would stop matching.
    """
    trial = [*produced, int(token)]
    text = tokenizer.decode(trial, skip_special_tokens=False)
    return list(tokenizer.encode(text).ids) == trial


def _raise(message: str) -> None:
    raise ValueError(message)


def _reasoning_effort(body: dict[str, Any]) -> str | None:
    """Template effort after ``resolve_effort``. ``none`` is omitted."""
    effort = body.get("reasoning_effort")
    if not isinstance(effort, str):
        extra = body.get("chat_template_kwargs")
        if isinstance(extra, dict):
            effort = extra.get("reasoning_effort")
    if not isinstance(effort, str):
        return None
    resolved = resolve_effort(effort)
    if resolved is None or resolved == "none":
        return None
    return resolved


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


def _sampling(
    body: dict[str, Any], thinking: bool, variant: dict[str, Any] | None
) -> tuple[float, float, int]:
    """Card values unless the request sets the field. Temperature 0 is an argmax."""
    del thinking
    card = variant if variant is not None else VARIANTS[DEFAULT_VARIANT]
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


def _limit(body: dict[str, Any]) -> int | None:
    """Request or settings cap. None means decode until a stop or the context ends."""
    from favlo.logs import output_token_limit

    return output_token_limit(body)


def tool_call_forbid(root: Path) -> set[int]:
    """Tool markup. ``tool_choice`` ``none`` keeps those tokens out of the reply."""
    del root
    return set(TOOL_CALL_FORBID)


def _forced_call(root: Path, tools: Any, required: str | None) -> tuple[list[int], bool]:
    if not required:
        return [], False
    if function_has_parameters(tools, required):
        return encode_text(root, f"<tool_call>\n<function={required}>\n"), False
    text = f"<tool_call>\n<function={required}>\n</function>\n</tool_call>"
    return encode_text(root, text), True


class Pipeline(LanguagePipeline):
    """One Responses generation for this package."""

    variants = _sampling_mod.VARIANTS
    default_variant = _sampling_mod.DEFAULT_VARIANT
    default_thinking = _sampling_mod.DEFAULT_THINKING
    default_reasoning_effort = _sampling_mod.DEFAULT_REASONING_EFFORT

    def has_vision(self) -> bool:
        return True

    def on_open(self, runtime: Any, reply: Any, selection: Any) -> None:
        super().on_open(runtime, reply, selection)
        runtime._halted = False

    def client_closed(self, exc: OSError) -> bool:
        return str(exc) == "client closed"

    def on_finish(self, runtime: Any, selection: Any, *, complete: bool) -> None:
        from qwen38_27b.metal41.tower import release_vision

        release_vision(runtime)
        super().on_finish(runtime, selection, complete=complete)

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



    def main(self, argv: Sequence[str]) -> int:
        from qwen38_27b.metal41.commands import main as cli_main

        return int(cli_main(argv))

    def open_runtime(self, path: Path, **kwargs: Any) -> Any:
        from qwen38_27b.metal41.runtime import open_runtime as open_rt
        from favlo.pipelines import bind_open_runtime

        return bind_open_runtime(open_rt, path, **kwargs)

pipeline = Pipeline()
handle_responses = pipeline
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
    requested = _request_effort(prepared)
    required = required_function(prepared.get("tool_choice"))
    thinking, card = mode_card(selection, requested)
    if required:
        thinking = False
    temperature, top_p, top_k = _sampling(prepared, thinking, card)
    presence = _presence(prepared, thinking, card)
    frequency = _frequency(prepared)
    messages, _ = template_messages(prepared)
    extra = format_instruction(prepared)
    if extra:
        messages = [{"role": "system", "content": extra}, *messages]
    search, note = web_search_context(prepared)
    if note:
        messages.append({"role": "user", "content": note})
    tools = function_tools(prepared.get("tools"))
    effort = template_effort(selection, requested) if thinking else None
    try:
        row = getattr(selection, "preserve_thinking", None)
        keep_thinking = preserve_thinking(prepared, row if isinstance(row, bool) else None)
    except RuntimeError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    try:
        prompt = _text_prompt(runtime, messages, tools, thinking, effort, keep_thinking)
        prompt = _fit_prompt(
            runtime, prepared, messages, tools, thinking, effort, keep_thinking, prompt
        )
        text_cut = (
            None
            if keep_thinking
            else user_turn_cut(runtime.root, messages, tools, thinking, effort)
        )
        text_prompt = prompt
        prompt, counts = _attach_vision(runtime, messages, prompt)
        runtime._stable_cut = _expanded_cut(text_prompt, text_cut, counts)
    except RequestError as exc:
        reply.json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        return
    forced, stop_after_forced = _forced_call(runtime.root, tools, required)
    emit_tools = prepared.get("tool_choice") != "none"
    bias = _logit_bias(prepared)
    if not emit_tools:
        for token in tool_call_forbid(runtime.root):
            bias[token] = -1e9
    think_end = THINK_END if thinking else None
    empty_close: list[int] = []
    text_close: list[int] = []
    tool_open: int | None = None
    if thinking:
        empty_close, text_close, tool_open = think_closes(runtime.root)
        # A required tool call is forced, so the sampler never sees the opener.
        # The template still writes the empty close before that call.
        if forced:
            forced = [*empty_close, *forced]
    fill = bool(prepared.get("favlo:fill_decode"))
    stream = bool(prepared.get("stream", False))
    _prefix_run(runtime, prepared)
    mtp_body = _mtp_body(prepared)
    mtp = _mtp_flag(mtp_body, selection)
    mtp_depth = _mtp_depth(mtp_body, selection)
    limit = _output_limit(prepared)
    if stream:
        _stream(
            reply,
            runtime,
            prompt,
            forced,
            stop_after_forced,
            limit,
            thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
            logit_bias=bias,
            think_end=think_end,
            think_empty_close=empty_close,
            think_text_close=text_close,
            tool_open=tool_open,
            emit_tools=emit_tools,
            mtp=mtp,
            mtp_depth=mtp_depth,
            fill=fill,
            prepared=prepared,
            state=state,
            search=search,
        )
        return
    tokens = runtime.generate(
        prompt,
        limit,
        STOP,
        forced=forced,
        stop_after_forced=stop_after_forced,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence=presence,
        frequency=frequency,
        logit_bias=bias,
        think_end=think_end,
        think_empty_close=empty_close,
        think_text_close=text_close,
        tool_open=tool_open,
        mtp=mtp,
        mtp_depth=mtp_depth,
        fill=fill,
    )
    text = decode_text(runtime.root, tokens)
    payload = _response_payload(
        reply,
        runtime,
        prepared,
        search,
        text,
        thinking,
        emit_tools,
        len(prompt),
        len(getattr(runtime, "_output_ids", None) or tokens),
        bool(getattr(runtime, "_hit_limit", False)),
    )
    reply.json(200, payload)


def _output_limit(prepared: dict[str, Any]) -> int | None:
    """``max_output_tokens`` when the body sets it, otherwise the settings cap."""
    cap = prepared.get("max_output_tokens")
    if cap is not None:
        return int(cap)
    return _limit({})


def _text_prompt(
    runtime: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    thinking: bool,
    effort: str | None,
    keep_thinking: bool,
) -> list[int]:
    return encode_chat(
        runtime.root,
        messages,
        tools,
        thinking,
        effort,
        preserve_thinking=keep_thinking,
    )


def _attach_vision(
    runtime: Any, messages: list[dict[str, Any]], prompt: list[int]
) -> tuple[list[int], list[int]]:
    """Expand image pads after the text cut is known. Counts are merged tokens."""
    from qwen38_27b.metal41.vision import (
        collect_images,
        expand_image_tokens,
        log_prepared,
        merged_grid,
        prepare_image,
    )

    noted = getattr(runtime, "_status", None)

    def _vision_note(text: str) -> None:
        if callable(noted):
            noted(text)

    images = collect_images(messages, note=_vision_note)
    pixels = [prepare_image(blob, min_pixels=low, max_pixels=high) for blob, low, high in images]
    log_prepared(runtime, pixels)
    runtime._vision_images = images
    runtime._vision_prepared = pixels
    if not pixels:
        return prompt, []
    counts = []
    for _patches, grid in pixels:
        height, width = merged_grid(grid)
        counts.append(height * width)
    return expand_image_tokens(prompt, counts), counts


def _prefix_run(runtime: Any, prepared: dict[str, Any]) -> None:
    """Copy a scratch prefix budget onto the runtime for this request.

    Absent fields clear a previous scratch run so the next chat uses the home
    partition again.
    """
    budget = prepared.get("favlo:prefix_budget")
    scratch = prepared.get("favlo:prefix_scratch")
    runtime._prefix_budget = (
        budget if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0 else None
    )
    runtime._prefix_scratch = scratch if isinstance(scratch, str) and scratch else None


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
    effort: str | None,
    keep_thinking: bool,
    prompt: list[int],
) -> list[int]:
    ceiling = getattr(runtime, "_limit", None)
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
        prompt = _text_prompt(runtime, kept, tools, thinking, effort, keep_thinking)
    if len(prompt) > ceiling:
        raise RequestError("input does not fit")
    messages[:] = kept
    return prompt


def _shown_reply(text: str, content: str | None, calls: list[dict[str, Any]]) -> str | None:
    """Reply text with a tool call removed. Text after the call stays."""
    if not calls:
        return content
    if content:
        return content
    end = text.rfind("</tool_call>")
    trailing = text[end + len("</tool_call>") :].strip() if end >= 0 else ""
    return trailing or None


def _response_payload(
    reply: Any,
    runtime: Any,
    prepared: dict[str, Any],
    search: dict[str, Any] | None,
    text: str,
    thinking: bool,
    emit_tools: bool,
    prompt_tokens: int,
    completion_tokens: int,
    hit_limit: bool,
    response_id: str | None = None,
) -> dict[str, Any]:
    reasoning, visible = split_thinking(text, thinking)
    content, calls = parse_tool_calls(visible)
    if not emit_tools:
        content = _shown_reply(visible if thinking else text, content, calls)
        calls = []
    elif not thinking:
        content = _shown_reply(text, content, calls)
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
    visible_text = content or ""
    if calls:
        if visible_text:
            output.append(_message_item(visible_text))
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
        output.append(_message_item(visible_text))
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
    payload["_chat_usage"] = chat_usage
    return payload


def _message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": new_id("msg_"),
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


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
    think_end: int | None = None,
    think_empty_close: list[int] | None = None,
    think_text_close: list[int] | None = None,
    tool_open: int | None = None,
    rng: np.random.Generator | None = None,
    mtp: bool = False,
    mtp_depth: int | None = None,
    fill: bool = False,
    prepared: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    search: dict[str, Any] | None = None,
) -> None:
    del rng
    decode = getattr(runtime, "decode", None)
    if not callable(decode):
        tokenizer = load_tokenizer(runtime.root)

        def decode(ids: list[int]) -> str:
            return tokenizer.decode(ids, skip_special_tokens=False)

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
    pending: queue.Queue[int | str | None | object] = queue.Queue()
    failed: list[BaseException] = []
    held: list[tuple[str, int]] = []
    keepalive = object()

    def on_prefill(_done: int) -> None:
        """Queue one empty delta. The stream thread writes it."""
        if failed:
            raise RequestClosed()
        pending.put(keepalive)

    def write() -> None:
        produced: list[int] = []
        previous = ""
        gate = StreamText(thinking, tools=emit_tools)
        opened: set[str] = set()

        def emit_piece(channel: str, visible: str) -> None:
            if not visible or channel in {"tool_start", "tool_args"}:
                return
            if channel not in opened:
                opened.add(channel)
                _open_item(event, channel, reasoning_id, message_id)
            _emit_response(event, channel, visible, reasoning_id, message_id)

        try:
            while True:
                item = pending.get()
                if item is keepalive:
                    event(
                        {
                            "type": "response.output_text.delta",
                            "content_index": 0,
                            "delta": "",
                            "item_id": message_id,
                            "output_index": 1,
                            "logprobs": [],
                        }
                    )
                    continue
                final = item is None
                if not final:
                    produced.append(int(item))
                decoded = decode(produced) if produced else ""
                piece, previous = _stream_delta(decoded, previous, final=final)
                for channel, visible in gate.push(piece):
                    emit_piece(channel, visible)
                if final:
                    for channel, visible in gate.finish():
                        emit_piece(channel, visible)
                    held.append((previous, len(produced)))
                    break
        except OSError:
            failed.append(RequestClosed())
        except BaseException as exc:
            failed.append(exc)

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
            STOP,
            forced=forced,
            stop_after_forced=stop_after_forced,
            on_token=enqueue,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence=presence,
            frequency=frequency,
            logit_bias=logit_bias,
            think_end=think_end,
            think_empty_close=think_empty_close,
            think_text_close=think_text_close,
            tool_open=tool_open,
            mtp=mtp,
            mtp_depth=mtp_depth,
            fill=fill,
            on_prefill=on_prefill,
        )
    finally:
        pending.put(None)
        writer.join()
    if failed:
        raise failed[0]
    text, streamed = held[-1] if held else ("", 0)
    counted = getattr(runtime, "_output_ids", None)
    completion_tokens = len(counted) if isinstance(counted, list) else streamed
    payload = _response_payload(
        reply,
        runtime,
        request,
        search,
        text,
        thinking,
        emit_tools,
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

