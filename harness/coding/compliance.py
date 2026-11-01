"""OpenAI chat contract for the coding harness.

The coding agent is the production-readiness run. These checks share its
listener: a system instruction and a user reply, a thinking follow-up,
``finish_reason`` ``length``, a stop sequence, a required tool call,
``tool_choice`` ``none``, and a streamed reply. The horse still is the first
coding chat when ``--vision`` is set, not a separate check.

The listener's chat object uses ``stop``, ``length``, and ``tool_calls``. It does
not return ``content_filter``. Audio parts, file parts, web search, logprobs, and
``n`` greater than one are not part of that object. ``reasoning_content`` is the
trace this listener stores.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from harness.chat_agent import reply_parts, thinking_model
from harness.image_attachment import QUESTION, sees_horse

_PATCH = 16
_POOL = 3
_SOFT_MAX = 280

WORD = "apple"
CODE = "pebble-7"
LABEL = f"drawer holds {CODE}"
STOP_WORD = "STOPWORD"
_MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>")
_INSTRUCTION = "The reply is the single word apple."
_REPLY = "Name one fruit."
_LENGTH = "Write a sentence of at least twelve words about a horse."
_STOP = "Repeat exactly: alpha STOPWORD omega"
_REASONING = (
    "What vegetable does a horse eat? Name one.",
    "Name a second vegetable a horse eats.",
)


_TOWER = re.compile(r"vision patches \d+")
_CACHE = re.compile(r"vision cache")


def markup_problem(text: str) -> str | None:
    """A reply that still contains think or tool markup."""
    for tag in _MARKUP:
        if tag in text:
            return f"reply contains {tag}"
    return None


def reply_problems(text: str, reasoning: str) -> list[str]:
    """Problems in one finished reply. Reasoning stays off the reply text."""
    problems: list[str] = []
    if not text:
        problems.append("the reply is empty")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    if reasoning and reasoning == text:
        problems.append("the reply repeats the reasoning")
    if WORD not in text.lower():
        problems.append(f"the reply does not contain {WORD}")
    return problems


def reasoning_problems(text: str, reasoning: str, *, stored: str) -> list[str]:
    """Problems when a thinking turn does not keep reasoning for the next request."""
    problems: list[str] = []
    if not reasoning:
        problems.append("the turn returned no reasoning")
    if not text:
        problems.append("the turn returned no reply")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    if reasoning and reasoning == text:
        problems.append("the reply repeats the reasoning")
    if not stored:
        problems.append("the session did not store reasoning")
    return problems


@dataclass(frozen=True)
class ToolRound:
    """One ``locker`` call and the tool result bound to its id."""

    call_id: str
    code: str
    result: str | None


def tool_problems(rounds: list[ToolRound], text: str) -> list[str]:
    """Problems when the required tool did not run or the reply dropped its label."""
    problems: list[str] = []
    if len(rounds) != 1:
        problems.append(f"locker calls {len(rounds)}")
    else:
        round_ = rounds[0]
        if not round_.call_id:
            problems.append("the tool call has no id")
        if round_.code != CODE:
            problems.append(f"locker received {round_.code!r}")
        if round_.result is None:
            problems.append("the tool result is not bound to the call id")
        elif round_.result != LABEL:
            problems.append("the tool result is missing")
    if LABEL not in text:
        problems.append("the reply does not repeat the tool result")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    return problems


def choice_problems(rounds: list[ToolRound], text: str, reasoning: str) -> list[str]:
    """Problems when tools are present and ``tool_choice`` is ``none``."""
    problems = reply_problems(text, reasoning)
    if rounds:
        problems.append("a tool ran when tool_choice was none")
    return problems


def length_problems(text: str) -> list[str]:
    """Problems in a reply that was cut at one completion token."""
    problems: list[str] = []
    if not text:
        problems.append("the reply is empty")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    return problems


def stop_problems(text: str) -> list[str]:
    """Problems when a stop sequence does not end the reply before the next word."""
    problems: list[str] = []
    if "alpha" not in text:
        problems.append("the reply does not contain alpha")
    if STOP_WORD in text or "omega" in text:
        problems.append("the stop sequence did not end the reply")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    return problems


def stream_problems(pieces: list[str], text: str) -> list[str]:
    """Problems when streamed text does not assemble to the reply."""
    if not pieces:
        return ["the stream returned no text"]
    joined = "".join(pieces)
    if text and text not in joined and joined not in text and pieces[-1] != text:
        return ["the streamed text does not match the reply"]
    return []


def finish_problem(reason: object, expected: str) -> str | None:
    """A finish reason other than the one this check requires."""
    if reason == expected:
        return None
    return f"finish_reason {reason} expected {expected}"


def usage_problems(details: object) -> list[str]:
    """Problems when usage is missing prompt, completion, or their sum."""
    if not isinstance(details, Mapping):
        return ["usage is missing"]
    prompt = details.get("input_token_count")
    completion = details.get("output_token_count")
    total = details.get("total_token_count")
    problems: list[str] = []
    if not isinstance(prompt, int) or prompt < 1:
        problems.append("prompt_tokens is missing")
    if not isinstance(completion, int) or completion < 1:
        problems.append("completion_tokens is missing")
    if isinstance(prompt, int) and isinstance(completion, int) and total != prompt + completion:
        problems.append("total_tokens is not prompt_tokens plus completion_tokens")
    return problems


def ended(response: Any, expected: str) -> list[str]:
    """Finish reason and usage problems for one agent response."""
    problems: list[str] = []
    finish = finish_problem(getattr(response, "finish_reason", None), expected)
    if finish is not None:
        problems.append(finish)
    problems.extend(usage_problems(getattr(response, "usage_details", None)))
    return problems


def image_content(png: bytes) -> Any:
    """One PNG part. The framework sends it as an ``image_url`` data URL."""
    from agent_framework import Content

    return Content.from_data(
        data=png,
        media_type="image/png",
        additional_properties={"detail": "auto"},
    )


def image_message(png: bytes) -> Any:
    """The question, then the still. The question does not name the animal."""
    from agent_framework import Message

    return Message("user", [QUESTION, image_content(png)])


def image_floor(width: int, height: int) -> int:
    """Pooled soft tokens the vision turn must cover.

    Patch 16, pool 3, and at most 280. A side that rounds to zero becomes one
    pooled patch, and the longer side shrinks until the count fits. A tower
    that emits more tokens still passes.
    """
    if width < 1 or height < 1:
        message = f"horse image {width}x{height} is empty"
        raise RuntimeError(message)
    pooled = _POOL * _PATCH
    scale = (_SOFT_MAX * pooled * pooled / (height * width)) ** 0.5

    def snap(side: int) -> int:
        snapped = int((scale * side) // pooled) * pooled
        if snapped < pooled:
            return pooled
        return snapped

    target_h = snap(height)
    target_w = snap(width)
    while (target_h // pooled) * (target_w // pooled) > _SOFT_MAX:
        if target_h >= target_w and target_h > pooled:
            target_h -= pooled
        elif target_w > pooled:
            target_w -= pooled
        else:
            break
    return (target_h // pooled) * (target_w // pooled)


def image_turn_problems(text: str, counted: object, width: int, height: int) -> list[str]:
    """Problems when the first chat did not see the still through the tower."""
    needed = image_floor(width, height)
    problems: list[str] = []
    if not isinstance(counted, int) or counted < needed:
        problems.append(f"prompt tokens {counted} are below the pooled tokens {needed}")
    if not sees_horse(text):
        problems.append("the reply does not describe the horse")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    return problems


def vision_problems(log: str) -> list[str]:
    """The tower encodes the still once. A later request reuses that result."""
    tower = [index for index, line in enumerate(log.splitlines()) if _TOWER.search(line)]
    cached = [index for index, line in enumerate(log.splitlines()) if _CACHE.search(line)]
    problems: list[str] = []
    if len(tower) != 1:
        problems.append(f"the vision tower ran {len(tower)} times")
    if not cached:
        problems.append("a later request did not reuse the vision tower")
    elif tower and cached[0] < tower[0]:
        problems.append("the vision cache preceded the tower")
    return problems


def session_messages(session: Any) -> list[Any]:
    """Messages the agent session stored for the next request."""
    state = getattr(session, "state", None)
    memory = state.get("in_memory") if isinstance(state, dict) else None
    messages = memory.get("messages") if isinstance(memory, dict) else None
    if not isinstance(messages, list):
        return []
    return messages


def stored_reasoning(session: Any) -> str:
    """Reasoning parts on stored assistant turns."""
    chunks: list[str] = []
    for message in session_messages(session):
        if getattr(message, "role", None) != "assistant":
            continue
        for part in getattr(message, "contents", ()) or ():
            if getattr(part, "type", None) != "text_reasoning":
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks).strip()


def _argument_map(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def tool_rounds(messages: list[Any]) -> list[ToolRound]:
    """``locker`` calls and the tool result stored under each call id."""
    calls: list[tuple[str, str]] = []
    results: dict[str, str] = {}
    seen_calls: set[object] = set()
    seen_results: set[object] = set()
    for message in messages:
        for part in getattr(message, "contents", ()) or ():
            kind = getattr(part, "type", None)
            if kind == "function_call" and getattr(part, "name", None) == "locker":
                call_id = getattr(part, "call_id", None)
                key = call_id or id(part)
                if key in seen_calls:
                    continue
                seen_calls.add(key)
                code = _argument_map(getattr(part, "arguments", None)).get("code")
                stored_id = call_id if isinstance(call_id, str) else ""
                stored_code = code if isinstance(code, str) else ""
                calls.append((stored_id, stored_code))
            elif kind == "function_result":
                call_id = getattr(part, "call_id", None)
                key = call_id or id(part)
                if key in seen_results or not isinstance(call_id, str) or not call_id:
                    continue
                seen_results.add(key)
                result = getattr(part, "result", None)
                if isinstance(result, str) and result:
                    results[call_id] = result
    return [ToolRound(call_id, code, results.get(call_id)) for call_id, code in calls]


def _tools(ran: list[str]) -> list[object]:
    from agent_framework import tool

    @tool(approval_mode="never_require")
    def locker(code: Annotated[str, Field(description="The code to store.")]) -> str:
        """Store a code and return the drawer label."""
        ran.append(code)
        return f"drawer holds {code}"

    return [locker]


def _limit_tools(client: Any) -> None:
    """One required call, then one reply. A third model turn stops the loop."""
    client.function_invocation_configuration["max_iterations"] = 3
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1


def _quiet(thinking: bool) -> dict[str, object]:
    return {"temperature": 0, "extra_body": {"enable_thinking": thinking}}


async def _run(
    base_url: str,
    model: str,
    prompt: Any,
    options: dict[str, object],
    **agent_kwargs: Any,
) -> tuple[Any, Any, Any]:
    """One agent turn. Returns the client, the session, and the response."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    client = open_chat_client(model, base_url)
    if agent_kwargs.pop("limit_tools", False):
        _limit_tools(client)
    agent = Agent(client=client, name="coding", default_options=options, **agent_kwargs)
    session = agent.create_session()
    try:
        response = await agent.run(prompt, session=session, options=options)
        return client, session, response
    finally:
        await close_chat_client(client)


async def run_reply(base_url: str, model: str) -> list[str]:
    """A system instruction the user text does not repeat. Thinking stays off."""
    options = _quiet(False)
    _client, _session, response = await _run(
        base_url,
        model,
        _REPLY,
        options,
        instructions=_INSTRUCTION,
    )
    reasoning, text = reply_parts(response)
    print(f"reply {text!r}", flush=True)
    return [*reply_problems(text, reasoning), *ended(response, "stop")]


async def run_reasoning(base_url: str, model: str) -> list[str]:
    """Two thinking turns. The first turn's reasoning stays on the session."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    options: dict[str, object] = {"extra_body": {"enable_thinking": True}}
    client = open_chat_client(thinking_model(model), base_url)
    agent = Agent(
        client=client,
        name="coding",
        instructions="Reply in one short sentence.",
        default_options=options,
    )
    session = agent.create_session()
    problems: list[str] = []
    kept = ""
    try:
        for index, prompt in enumerate(_REASONING, start=1):
            response = await agent.run(prompt, session=session, options=options)
            reasoning, text = reply_parts(response)
            stored = stored_reasoning(session)
            print(f"reasoning turn {index} reply {text!r}", flush=True)
            problems.extend(reasoning_problems(text, reasoning, stored=stored))
            problems.extend(ended(response, "stop"))
            if index == 1:
                kept = reasoning
            elif kept and kept not in stored:
                problems.append("the first turn's reasoning left the session")
    finally:
        await close_chat_client(client)
    return problems


async def run_tools(base_url: str, model: str) -> list[str]:
    """A required ``locker`` call, then a reply that repeats the tool result."""
    options: dict[str, object] = {
        **_quiet(False),
        "tool_choice": {"mode": "required", "required_function_name": "locker"},
    }
    _client, session, response = await _run(
        base_url,
        model,
        f"Call locker with code {CODE}. Then repeat the tool result exactly.",
        options,
        instructions="Use the tool, then repeat its result.",
        tools=_tools([]),
        limit_tools=True,
    )
    _reasoning, text = reply_parts(response)
    messages = [*session_messages(session), *getattr(response, "messages", [])]
    rounds = tool_rounds(messages)
    print(f"tools reply {text!r}", flush=True)
    return [*tool_problems(rounds, text), *ended(response, "stop")]


async def run_choice(base_url: str, model: str) -> list[str]:
    """Tools are listed and ``tool_choice`` is ``none``. The reply does not call one."""
    options: dict[str, object] = {**_quiet(False), "tool_choice": {"mode": "none"}}
    _client, session, response = await _run(
        base_url,
        model,
        _REPLY,
        options,
        instructions=_INSTRUCTION,
        tools=_tools([]),
        limit_tools=True,
    )
    reasoning, text = reply_parts(response)
    messages = [*session_messages(session), *getattr(response, "messages", [])]
    rounds = tool_rounds(messages)
    print(f"choice reply {text!r}", flush=True)
    return [*choice_problems(rounds, text, reasoning), *ended(response, "stop")]


async def run_length(base_url: str, model: str) -> list[str]:
    """One completion token. The framework sends that cap as ``max_completion_tokens``."""
    options: dict[str, object] = {**_quiet(False), "max_tokens": 1}
    _client, _session, response = await _run(base_url, model, _LENGTH, options)
    _reasoning, text = reply_parts(response)
    print(f"length reply {text!r}", flush=True)
    return [*length_problems(text), *ended(response, "length")]


async def run_stop(base_url: str, model: str) -> list[str]:
    """A stop sequence ends the reply before the following word."""
    options: dict[str, object] = {**_quiet(False), "stop": [STOP_WORD]}
    _client, _session, response = await _run(base_url, model, _STOP, options)
    _reasoning, text = reply_parts(response)
    print(f"stop reply {text!r}", flush=True)
    return [*stop_problems(text), *ended(response, "stop")]


async def run_stream(base_url: str, model: str) -> list[str]:
    """Stream the reply. Text deltas assemble to the same reply as the final response."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    options = _quiet(False)
    client = open_chat_client(model, base_url)
    agent = Agent(
        client=client,
        name="coding",
        instructions=_INSTRUCTION,
        default_options=options,
    )
    session = agent.create_session()
    pieces: list[str] = []
    try:
        stream = await agent.run(_REPLY, stream=True, session=session, options=options)
        async for update in stream:
            piece = getattr(update, "text", "")
            if isinstance(piece, str) and piece:
                pieces.append(piece)
        response = await stream.get_final_response()
    finally:
        await close_chat_client(client)
    reasoning, text = reply_parts(response)
    print(f"stream reply {text!r}", flush=True)
    problems = reply_problems(text, reasoning)
    problems.extend(stream_problems(pieces, text))
    problems.extend(ended(response, "stop"))
    return problems


async def run_contract(base_url: str, model: str) -> list[str]:
    """Reply, reasoning, tools, choice, length, stop, and stream on one listener."""
    problems: list[str] = []
    cases = (
        ("reply", run_reply),
        ("reasoning", run_reasoning),
        ("tools", run_tools),
        ("choice", run_choice),
        ("length", run_length),
        ("stop", run_stop),
        ("stream", run_stream),
    )
    for name, runner in cases:
        print(f"check {name}", flush=True)
        found = await runner(base_url, model)
        for item in found:
            print(item, flush=True)
        if found:
            problems.extend(found)
        else:
            print(f"{name} held", flush=True)
    return problems
