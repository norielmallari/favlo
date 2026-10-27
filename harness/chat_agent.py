"""Chat scenarios on the shared chat client.

Each scenario is a short conversation. A later bug adds another scenario.
A follow-up must extend the prefix stored by the turn before it, and every
turn must return a reply. The listener on ``127.0.0.1:8129`` is reused.
This command does not take the machine lock.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from harness.file_session import RequestRate, parse_requests
from harness.harness_model import judge_model
from harness.server import close_listener, open_listener

INSTRUCTIONS = "Reply in one short sentence."
TURN_LIMIT = 1024


@dataclass(frozen=True)
class Scenario:
    """One conversation. ``thinking`` selects that sampling row."""

    name: str
    turns: tuple[str, ...]
    thinking: bool


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="thinking-follow-up",
        turns=(
            "What vegetable does a horse eat? Name one.",
            "Name a second vegetable a horse eats.",
        ),
        thinking=True,
    ),
)


def thinking_model(model: str) -> str:
    """The thinking row of a registry id."""
    stem, _sep, _variant = model.partition(":")
    return f"{stem}:thinking"


def prefix_holds(requests: list[RequestRate]) -> list[str]:
    """Problems when a later request does not extend the stored prefix."""
    if len(requests) < 2:
        return ["the conversation made fewer than two requests"]
    problems: list[str] = []
    stored = 0
    for index, request in enumerate(requests):
        if index == 0:
            if request.decode_tokens <= 0:
                problems.append("turn 1 drew no tokens")
            stored = request.prompt_tokens + request.decode_tokens
            continue
        turn = index + 1
        if request.missed or request.prefix_hit != stored:
            seen = "miss" if request.missed else str(request.prefix_hit)
            problems.append(f"turn {turn} prefix {seen} expected {stored}")
        if request.prompt_tokens <= stored:
            problems.append(f"turn {turn} prompt did not grow past {stored}")
        stored = request.prompt_tokens + request.decode_tokens
    return problems


def scenario_named(name: str | None) -> tuple[Scenario, ...]:
    if name is None:
        return SCENARIOS
    chosen = [item for item in SCENARIOS if item.name == name]
    if not chosen:
        known = ", ".join(item.name for item in SCENARIOS)
        message = f"unknown scenario {name}; known scenarios: {known}"
        raise SystemExit(message)
    return tuple(chosen)


def _log_text() -> str:
    from favlo.logs import favlo_home, log_path

    path = log_path(favlo_home())
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def log_mark(text: str) -> str:
    """A tail the next read can find after the log drops older lines."""
    return text[-240:]


def requests_since(mark: str, text: str) -> list[RequestRate]:
    """Requests in ``text`` that follow ``mark``.

    The server log keeps a fixed number of bytes, so a count of older records
    does not stay put. A missing mark means the tail was rolled off.
    """
    if mark and mark in text:
        fresh = text[text.rfind(mark) + len(mark) :]
    else:
        fresh = text
    return parse_requests(fresh)


def reply_parts(response: Any) -> tuple[str, str]:
    """Reasoning and reply text from one agent response."""
    reasoning: list[str] = []
    text: list[str] = []
    for message in getattr(response, "messages", None) or []:
        if getattr(message, "role", None) != "assistant":
            continue
        for part in getattr(message, "contents", ()) or ():
            value = getattr(part, "text", None)
            if not isinstance(value, str) or not value:
                continue
            if getattr(part, "type", None) == "text_reasoning":
                reasoning.append(value)
            elif getattr(part, "type", None) == "text":
                text.append(value)
    if not text:
        fallback = getattr(response, "text", None)
        if isinstance(fallback, str) and fallback.strip():
            text.append(fallback)
    return "".join(reasoning).strip(), "".join(text).strip()


async def run_scenario(base_url: str, model: str, scenario: Scenario) -> list[str]:
    """Run one conversation. Problems are returned; the caller prints them."""
    from agent_framework import Agent, InMemoryHistoryProvider
    from openai import AsyncOpenAI

    from favlo.chat_client import history_message, open_chat_client

    chat_model = thinking_model(model) if scenario.thinking else model
    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=3600.0)
    client = open_chat_client(chat_model, base_url, async_client=http)
    options: dict[str, object] = {
        "max_tokens": TURN_LIMIT,
        "extra_body": {"enable_thinking": scenario.thinking},
    }
    agent = Agent(
        client=client,
        name="chat-agent",
        instructions=INSTRUCTIONS,
        default_options=options,
    )
    prior: list[tuple[str, str, str]] = []
    problems: list[str] = []
    try:
        for index, prompt in enumerate(scenario.turns, start=1):
            session = agent.create_session()
            bucket = session.state.setdefault(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
            messages = bucket.setdefault("messages", [])
            messages.extend(
                history_message(role, text, reasoning) for role, text, reasoning in prior
            )
            response = await agent.run(prompt, session=session, options=options)
            reasoning, text = reply_parts(response)
            print(f"{scenario.name} turn {index} reply {text!r}", flush=True)
            if scenario.thinking and not reasoning:
                problems.append(f"turn {index} returned no reasoning")
            if not text:
                problems.append(f"turn {index} returned no reply")
            prior.append(("user", prompt, ""))
            prior.append(("assistant", text, reasoning))
    finally:
        closer = getattr(client, "close", None)
        if closer is not None:
            closed = closer()
            if hasattr(closed, "__await__"):
                await closed
        await http.close()
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run accumulated chat scenarios.")
    parser.add_argument(
        "--model",
        default=None,
        help="chat model id; default is the registry default",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="one scenario name; default runs every scenario",
    )
    args = parser.parse_args(argv)
    model = args.model or judge_model()
    chosen = scenario_named(args.scenario)
    listener = open_listener()
    base_url = f"http://{listener.host}:{listener.port}/v1"
    failed = False
    try:
        for scenario in chosen:
            mark = log_mark(_log_text())
            chat_model = thinking_model(model) if scenario.thinking else model
            print(f"scenario {scenario.name} model {chat_model}", flush=True)
            problems = asyncio.run(run_scenario(base_url, model, scenario))
            requests = requests_since(mark, _log_text())
            for index, request in enumerate(requests, start=1):
                hit = "miss" if request.missed else str(request.prefix_hit)
                print(
                    f"{scenario.name} turn {index} prompt {request.prompt_tokens} "
                    f"prefix {hit} decode {request.decode_tokens}",
                    flush=True,
                )
            if len(requests) != len(scenario.turns):
                problems.append(
                    f"{scenario.name} logged {len(requests)} requests, "
                    f"expected {len(scenario.turns)}"
                )
            else:
                problems.extend(prefix_holds(requests))
            for problem in problems:
                print(problem, flush=True)
            if problems:
                failed = True
    finally:
        close_listener(listener)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
