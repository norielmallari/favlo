"""OpenAI chat contract for the coding harness.

The coding agent is the production-readiness run. These checks share its
listener: a system instruction and a user reply, one short reply at each
text reasoning effort, a stored follow-up, a required tool call,
``tool_choice`` ``none``, a hosted
``web_search``, ``finish_reason`` ``length``, a streamed reply, and a second
streamed chat that keeps thinking on and must show the tool name, then argument
fragments, after the think block. After that contract holds, the command
posts two raw ``/v1/responses`` calls with thinking off, one streamed and one
not. A stream error fails the run. The horse
still is the first coding chat when ``--vision`` is set, not a separate check.

The listener's chat object uses ``stop``, ``length``, and ``tool_calls``. It does
not return ``content_filter``. Audio parts, file parts, logprobs, and ``n`` greater
than one are not checks. ``web_search`` is a hosted tool the listener runs.
``store`` is one value for every check. False is the client replay. True
is ``previous_response_id`` and the listener chain.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from harness.chat_agent import reply_parts
from harness.image_attachment import QUESTION, sees_horse

_PATCH = 16
_POOL = 3
_SOFT_MAX = 280

WORD = "apple"
CODE = "pebble-7"
LABEL = f"drawer holds {CODE}"
STOP_WORD = "STOPWORD"
TEXT_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>")
_INSTRUCTION = "The reply is the single word apple."
_REPLY = "Name one fruit."
_LENGTH = "Write a sentence of at least twelve words about a horse."
_STOP = "Repeat exactly: alpha STOPWORD omega"
_REASONING = "Reply with the word oats."
_REASONING_TOKENS = 256
_SECRET = "The secret word is apple. Reply with ok."
_RECALL = "What is the secret word? Reply with that word only."
_SEARCH = "Search the web for the Qwen3.6-35B-A3B model card and answer in one sentence."


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


def reasoning_level_problems(
    text: str, reasoning: str, *, stored: str, effort: str
) -> list[str]:
    """Problems for one text effort. ``none`` and ``minimal`` may omit the trace.

    ``none`` is the OpenAI off switch. ``minimal`` is the Gemini off switch.
    A trace that is present must be stored and must stay off the reply.
    """
    if effort in {"none", "minimal"}:
        problems: list[str] = []
        if not text:
            problems.append("the turn returned no reply")
        leaked = markup_problem(text)
        if leaked is not None:
            problems.append(leaked)
        if reasoning and reasoning == text:
            problems.append("the reply repeats the reasoning")
        if reasoning and not stored:
            problems.append("the session did not store reasoning")
        return problems
    return reasoning_problems(text, reasoning, stored=stored)


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


def search_calls(response: Any) -> list[str]:
    """Ids of ``web_search`` calls on one agent response."""
    found: list[str] = []
    for message in getattr(response, "messages", None) or []:
        for part in getattr(message, "contents", ()) or ():
            if getattr(part, "type", None) != "search_tool_call":
                continue
            if getattr(part, "tool_name", None) != "web_search":
                continue
            call_id = getattr(part, "call_id", None)
            found.append(call_id if isinstance(call_id, str) else "")
    return found


def web_search_problems(text: str, calls: list[str]) -> list[str]:
    """Problems when the listener did not run ``web_search`` or the reply is empty."""
    problems: list[str] = []
    if not calls:
        problems.append("the response has no web_search call")
    elif not calls[0]:
        problems.append("the web_search call has no id")
    if not text:
        problems.append("the reply is empty")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    return problems


def recall_problems(text: str) -> list[str]:
    """Problems when a stored follow-up drops the secret word."""
    problems: list[str] = []
    if not text:
        problems.append("the reply is empty")
    elif WORD not in text.lower():
        problems.append(f"the stored follow-up does not contain {WORD}")
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


def tool_stream_problems(
    *,
    saw_reasoning: bool,
    name: str,
    pieces: list[str],
) -> list[str]:
    """Problems when a thinking stream does not show the call being built.

    Reasoning comes first. The first argument piece is not the finished object.
    Later pieces complete that object, and it carries the locker code.
    """
    problems: list[str] = []
    if not saw_reasoning:
        problems.append("the stream returned no reasoning before the tool call")
    if name != "locker":
        problems.append(f"streamed tool {name!r}")
    first = pieces[0] if pieces else ""
    try:
        opened = json.loads(first) if first else None
    except json.JSONDecodeError:
        opened = None
    if isinstance(opened, dict):
        problems.append("the tool call arrived in one piece")
    parsed = _first_object(pieces)
    if not isinstance(parsed, dict) or parsed.get("code") != CODE:
        problems.append("the streamed arguments are not the locker code")
    return problems


def _first_object(pieces: list[str]) -> Any:
    joined = ""
    for piece in pieces:
        joined += piece
        try:
            value = json.loads(joined)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


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
    """Problems when the reply does not describe the still."""
    del counted, width, height
    problems: list[str] = []
    if not sees_horse(text):
        problems.append("the reply does not describe the horse")
    leaked = markup_problem(text)
    if leaked is not None:
        problems.append(leaked)
    return problems


def vision_problems(log: str) -> list[str]:
    """The tower encodes the still once. A later request reuses that result.

    ``vision cached`` is a copy already inside a loaded prefix. It may precede
    the tower that encodes a later copy of that still. A disk ``vision cache``
    line before the tower means the row file was read before the tower ran.
    """
    lines = log.splitlines()
    tower = [index for index, line in enumerate(lines) if _TOWER.search(line)]
    reuse = [index for index, line in enumerate(lines) if line.startswith("vision cache")]
    problems: list[str] = []
    if len(tower) != 1:
        problems.append(f"the vision tower ran {len(tower)} times")
    if not reuse:
        problems.append("a later request did not reuse the vision tower")
    elif tower:
        for index in reuse:
            if index >= tower[0] or lines[index].startswith("vision cached"):
                continue
            problems.append("the vision cache preceded the tower")
            break
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
        cleaned = code.strip()
        ran.append(cleaned)
        return f"drawer holds {cleaned}"

    return [locker]


def _limit_tools(client: Any) -> None:
    """One required call, then one reply. A third model turn stops the loop."""
    client.function_invocation_configuration["max_iterations"] = 3
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1


def _quiet(thinking: bool, *, store: bool) -> dict[str, object]:
    effort = "medium" if thinking else "none"
    return {"store": store, "temperature": 0, "reasoning": {"effort": effort}}


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


async def run_reply(base_url: str, model: str, *, store: bool) -> list[str]:
    """A system instruction the user text does not repeat. Thinking stays off."""
    options = _quiet(False, store=store)
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


async def run_reasoning(base_url: str, model: str, *, store: bool) -> list[str]:
    """One short reply at each text effort. A trace stays on the session."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    client = open_chat_client(model.partition(":")[0], base_url)
    agent = Agent(client=client, name="coding")
    problems: list[str] = []
    try:
        for effort in TEXT_EFFORTS:
            print(f"reasoning {effort}", flush=True)
            options: dict[str, object] = {
                **_quiet(effort != "none", store=store),
                "max_tokens": _REASONING_TOKENS,
            }
            options["reasoning"] = {"effort": effort}
            session = agent.create_session()
            response = await agent.run(_REASONING, session=session, options=options)
            reasoning, text = reply_parts(response)
            stored = stored_reasoning(session)
            if store and not stored:
                stored = reasoning
            print(f"reasoning {effort} reply {text!r}", flush=True)
            found = reasoning_level_problems(text, reasoning, stored=stored, effort=effort)
            found.extend(ended(response, "stop"))
            problems.extend(f"{effort}: {item}" for item in found)
    finally:
        await close_chat_client(client)
    return problems


async def run_tools(base_url: str, model: str, *, store: bool) -> list[str]:
    """A required ``locker`` call, then a reply that repeats the tool result."""
    options: dict[str, object] = {
        **_quiet(False, store=store),
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


async def run_choice(base_url: str, model: str, *, store: bool) -> list[str]:
    """Tools are listed and ``tool_choice`` is ``none``. The reply does not call one."""
    options: dict[str, object] = {
        **_quiet(False, store=store),
        "tool_choice": {"mode": "none"},
    }
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


async def run_length(base_url: str, model: str, *, store: bool) -> list[str]:
    """Sixteen output tokens. The framework sends that cap as ``max_output_tokens``."""
    options: dict[str, object] = {**_quiet(False, store=store), "max_tokens": 16}
    _client, _session, response = await _run(base_url, model, _LENGTH, options)
    _reasoning, text = reply_parts(response)
    print(f"length reply {text!r}", flush=True)
    return [*length_problems(text), *ended(response, "length")]


async def run_stop(base_url: str, model: str, *, store: bool) -> list[str]:
    """A stop sequence ends the reply before the following word."""
    options: dict[str, object] = {**_quiet(False, store=store), "stop": [STOP_WORD]}
    _client, _session, response = await _run(base_url, model, _STOP, options)
    _reasoning, text = reply_parts(response)
    print(f"stop reply {text!r}", flush=True)
    return [*stop_problems(text), *ended(response, "stop")]


async def run_stream(base_url: str, model: str, *, store: bool) -> list[str]:
    """Stream the reply. Text deltas assemble to the same reply as the final response."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    options = _quiet(False, store=store)
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


async def run_stored(base_url: str, model: str, *, store: bool) -> list[str]:
    """Two turns at this run's ``store`` value. The follow-up repeats the secret."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    options = _quiet(False, store=store)
    client = open_chat_client(model, base_url)
    agent = Agent(client=client, name="coding", default_options=options)
    session = agent.create_session()
    problems: list[str] = []
    try:
        first = await agent.run(_SECRET, session=session, options=options)
        _reasoning, text = reply_parts(first)
        print(f"stored turn 1 reply {text!r}", flush=True)
        problems.extend(ended(first, "stop"))
        if not text:
            problems.append("the reply is empty")
        second = await agent.run(_RECALL, session=session, options=options)
        _reasoning, text = reply_parts(second)
        print(f"stored turn 2 reply {text!r}", flush=True)
        problems.extend(recall_problems(text))
        problems.extend(ended(second, "stop"))
    finally:
        await close_chat_client(client)
    return problems


async def run_web_search(base_url: str, model: str, *, store: bool) -> list[str]:
    """The listener runs ``web_search``. The client only names the tool."""
    from agent_framework_openai import OpenAIChatClient

    options = _quiet(False, store=store)
    tool = OpenAIChatClient.get_web_search_tool(search_context_size="low")
    _client, _session, response = await _run(
        base_url,
        model,
        _SEARCH,
        options,
        tools=[tool],
    )
    _reasoning, text = reply_parts(response)
    calls = search_calls(response)
    print(f"web search reply {text!r}", flush=True)
    return [*web_search_problems(text, calls), *ended(response, "stop")]


def _stream_reasoning(update: Any) -> str:
    parts = [
        item.text
        for item in getattr(update, "contents", None) or []
        if getattr(item, "type", None) == "text_reasoning"
        and isinstance(getattr(item, "text", None), str)
        and item.text
    ]
    return "".join(parts)


async def run_tool_stream(base_url: str, model: str, *, store: bool) -> list[str]:
    """Thinking stays on. The tool name streams, then the argument fragments."""
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    options = _quiet(True, store=store)
    client = open_chat_client(model, base_url)
    agent = Agent(
        client=client,
        name="coding",
        instructions="Call locker. Do not describe the call in the reply.",
        tools=_tools([]),
        default_options=options,
    )
    session = agent.create_session()
    saw_reasoning = False
    name = ""
    pieces: list[str] = []
    prompt = f"Think in one sentence, then call locker with code {CODE}."
    try:
        stream = await agent.run(prompt, stream=True, session=session, options=options)
        async for update in stream:
            if _stream_reasoning(update):
                if not name:
                    saw_reasoning = True
            for part in getattr(update, "contents", None) or []:
                if getattr(part, "type", None) != "function_call":
                    continue
                part_name = getattr(part, "name", "") or ""
                part_args = getattr(part, "arguments", "") or ""
                if isinstance(part_args, dict):
                    part_args = json.dumps(part_args)
                if not isinstance(part_args, str):
                    part_args = ""
                if part_name and not name:
                    name = part_name
                if part_args:
                    pieces.append(part_args)
            if getattr(update, "finish_reason", None):
                break
    finally:
        await close_chat_client(client)
    print(f"tool stream {name!r} pieces {len(pieces)}", flush=True)
    return tool_stream_problems(
        saw_reasoning=saw_reasoning,
        name=name,
        pieces=pieces,
    )


async def run_contract(base_url: str, model: str, *, store: bool) -> list[str]:
    """Reply, reasoning, store, tools, choice, web search, length, stream, and tool stream."""
    problems: list[str] = []
    cases = (
        ("reply", run_reply),
        ("reasoning", run_reasoning),
        ("stored", run_stored),
        ("tools", run_tools),
        ("choice", run_choice),
        ("web", run_web_search),
        ("length", run_length),
        ("stream", run_stream),
        ("tool_stream", run_tool_stream),
    )
    for name, runner in cases:
        print(f"check {name}", flush=True)
        found = await runner(base_url, model, store=store)
        for item in found:
            print(item, flush=True)
        if found:
            problems.extend(found)
        else:
            print(f"{name} held", flush=True)
    return problems


def raw_reply_problems(*, deltas: list[str], text: str) -> list[str]:
    """Problems when a thinking-off reply did not arrive as text deltas."""
    if len(deltas) < 2:
        return ["the reply did not stream"]
    if deltas[0] == text:
        return ["the reply arrived in one piece"]
    if "".join(deltas) != text:
        return ["the streamed text is not the reply"]
    return []


def raw_closed_problems(body: object) -> list[str]:
    """Problems when a non-stream response is not one finished reply."""
    if not isinstance(body, dict):
        return ["the response was not one object"]
    status = body.get("status")
    if status not in {"completed", "incomplete"}:
        return [f"status {status}"]
    if not _output_text(body):
        return ["the reply was empty"]
    return []


def run_raw_responses(base_url: str, model: str) -> list[str]:
    """Thinking-off Responses calls, stream on then stream off."""
    problems: list[str] = []
    print("check raw_stream", flush=True)
    found = _raw_stream(base_url, model)
    problems.extend(_report("raw_stream", found))
    print("check raw_closed", flush=True)
    found = _raw_closed(base_url, model)
    problems.extend(_report("raw_closed", found))
    return problems


def _report(name: str, found: list[str]) -> list[str]:
    for item in found:
        print(item, flush=True)
    if found:
        return found
    print(f"{name} held", flush=True)
    return []


def _raw_body(model: str, *, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "input": "Count from one to five in one sentence.",
        "stream": stream,
        "store": False,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 64,
    }


def _raw_stream(base_url: str, model: str) -> list[str]:
    try:
        raw = _post(base_url, _raw_body(model, stream=True))
    except RuntimeError as exc:
        return [str(exc)]
    events = _sse_events(raw)
    if not events:
        return ["the stream returned no events"]
    deltas, text, error = _consume_responses(events)
    if error:
        return [error]
    print(f"raw stream deltas {len(deltas)}", flush=True)
    return raw_reply_problems(deltas=deltas, text=text)


def _raw_closed(base_url: str, model: str) -> list[str]:
    try:
        raw = _post(base_url, _raw_body(model, stream=False))
    except RuntimeError as exc:
        return [str(exc)]
    text = raw.decode("utf-8", errors="replace").lstrip()
    if text.startswith("data:"):
        return ["stream false returned a stream"]
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return ["the response was not one object"]
    return raw_closed_problems(body)


def _post(base_url: str, body: dict[str, Any]) -> bytes:
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        message = f"HTTP {exc.code} {detail}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        message = f"responses request failed: {exc.reason}"
        raise RuntimeError(message) from exc


def _sse_events(raw: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _output_text(body: Mapping[str, Any]) -> str:
    output = body.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                parts.append(str(part.get("text") or ""))
    return "".join(parts)


def _consume_responses(events: list[dict[str, Any]]) -> tuple[list[str], str, str]:
    from openai._models import construct_type_unchecked
    from openai._types import omit
    from openai.lib.streaming.responses._responses import ResponseStreamState
    from openai.types.responses import ResponseStreamEvent as RawResponseStreamEvent

    state = ResponseStreamState(input_tools=omit, text_format=omit)
    deltas: list[str] = []
    for payload in events:
        try:
            event = construct_type_unchecked(type_=RawResponseStreamEvent, value=payload)
            fired = state.handle_event(event)
        except Exception as exc:
            return deltas, "", f"stream error: {exc}"
        for item in fired:
            if getattr(item, "type", "") == "response.output_text.delta":
                deltas.append(str(getattr(item, "delta", "") or ""))
    text = ""
    for payload in reversed(events):
        if payload.get("type") == "response.completed":
            response = payload.get("response")
            if isinstance(response, dict):
                text = _output_text(response)
            break
    return deltas, text, ""
