"""One Agent Framework run against ``serve``. The score is the hidden suite."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from harness.benchmarks.serve import Request
from harness.coding.agent_tools import coding_tools
from harness.coding.session import session_for
from harness.coding.workspace import PROMPT, prepare_workspace

# One model call per tool step, then status polls, then a stop.
SESSION_ROUNDS = 16


def agent_options(request: Request, *, mtp: bool, mtp_depth: int) -> dict[str, object]:
    """Fields the framework client sends. Sampling stays off the body.

    ``store`` is false so the client replays function calls and reasoning.
    A one-turn check and a server continuation use ``store`` true instead.
    """
    return {
        "store": False,
        "reasoning": {"effort": "medium" if request.enable_thinking else "none"},
        "extra_body": {
            "favlo:mtp": bool(mtp),
            "favlo:mtp_depth": int(mtp_depth),
        },
    }


def run_session(
    port: int,
    model: str,
    request: Request,
    workspace: Path,
    *,
    mtp: bool,
    mtp_depth: int = 3,
    vision: bytes | None = None,
) -> tuple[list[str], list[str]]:
    """Let the agent use the tools. Return the calls that succeeded and any problems."""
    prepare_workspace(workspace)
    session = session_for(workspace)
    problems: list[str] = []
    try:
        asyncio.run(
            _run(
                port,
                model,
                request,
                session,
                mtp=mtp,
                mtp_depth=mtp_depth,
                vision=vision,
                problems=problems,
            )
        )
        return list(session.calls), problems
    finally:
        session.close()


async def _run(
    port: int,
    model: str,
    request: Request,
    session: Any,
    *,
    mtp: bool,
    mtp_depth: int,
    vision: bytes | None,
    problems: list[str],
) -> None:
    from agent_framework import Agent

    from favlo.chat_client import close_chat_client, open_chat_client

    client = open_chat_client(model.partition(":")[0], f"http://127.0.0.1:{port}/v1")
    client.function_invocation_configuration["max_iterations"] = SESSION_ROUNDS
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = SESSION_ROUNDS
    client.function_invocation_configuration["allow_concurrent_invocation"] = False
    options = agent_options(request, mtp=mtp, mtp_depth=mtp_depth)
    agent = Agent(
        client=client,
        name="coding",
        instructions=(
            "You are a coding assistant. Carry out the next step, then wait "
            "for its result. Leave the reply empty."
        ),
        tools=coding_tools(session),
        default_options=options,
    )
    printer = TurnPrinter(sys.stdout)
    try:
        chat = agent.create_session() if vision else None
        if vision is not None and chat is not None:
            await _describe(agent, client, vision, options, chat, printer, problems)
        stream = agent.run(PROMPT, stream=True, session=chat, options=options)
        async for update in stream:
            printer.take(update)
        printer.close()
    finally:
        await close_chat_client(client)


async def _describe(
    agent: Any,
    client: Any,
    png: bytes,
    options: dict[str, object],
    chat: Any,
    printer: TurnPrinter,
    problems: list[str],
) -> None:
    """The first chat is the still. Tools stay off so the tower runs before the ledger."""
    from harness.chat_agent import reply_parts
    from harness.coding.compliance import ended, image_message, image_turn_problems
    from harness.image_attachment import png_size

    first = {**options, "tool_choice": {"mode": "none"}}
    stream = await agent.run(image_message(png), stream=True, session=chat, options=first)
    async for update in stream:
        printer.take(update)
    response = await stream.get_final_response()
    printer.close()
    _reasoning, text = reply_parts(response)
    width, height = png_size(png)
    counted = getattr(client, "prompt_tokens", None)
    problems.extend(image_turn_problems(text, counted, width, height))
    problems.extend(ended(response, "stop"))


class TurnPrinter:
    """Print reasoning, the reply, and tool calls for each model call."""

    def __init__(self, sink: TextIO) -> None:
        self.sink = sink
        self.turn = 0
        self.response_id = ""
        self.ended = True
        self.saw_reply = False
        self.section = ""
        self.calls: list[dict[str, str]] = []

    def take(self, update: Any) -> None:
        response_id = getattr(update, "response_id", None) or ""
        if self.ended and _has_body(update):
            self._start(response_id)
        else:
            self._roll(response_id)
        role = getattr(update, "role", None)
        if role in (None, "assistant"):
            reason = _reasoning(update)
            if reason:
                self._reasoning(reason)
            text = getattr(update, "text", "")
            if isinstance(text, str) and text:
                self._reply(text)
        for item in getattr(update, "contents", None) or []:
            if getattr(item, "type", None) == "function_call":
                self._call(item)
        if getattr(update, "finish_reason", None):
            self._end()

    def close(self) -> None:
        self._end()

    def _roll(self, response_id: str) -> None:
        if self.turn == 0:
            self._start(response_id)
            return
        if response_id and self.response_id and response_id != self.response_id:
            self._end()
            self._start(response_id)

    def _start(self, response_id: str) -> None:
        self.turn += 1
        self.response_id = response_id
        self.ended = False
        self.saw_reply = False
        self.section = "reasoning"
        self.calls = []
        print(f"--- request {self.turn} reasoning ---", file=self.sink, flush=True)

    def _reasoning(self, text: str) -> None:
        if self.section != "reasoning":
            print(f"\n--- request {self.turn} reasoning ---", file=self.sink, flush=True)
            self.section = "reasoning"
        print(text, end="", file=self.sink, flush=True)

    def _reply(self, text: str) -> None:
        if not self.saw_reply:
            print(f"\n--- request {self.turn} reply ---", file=self.sink, flush=True)
            self.saw_reply = True
            self.section = "reply"
        print(text, end="", file=self.sink, flush=True)

    def _call(self, item: Any) -> None:
        call_id = getattr(item, "call_id", None) or ""
        if not isinstance(call_id, str):
            call_id = str(call_id)
        found = next((row for row in self.calls if row["id"] == call_id), None) if call_id else None
        if found is None:
            found = {"id": call_id or f"call_{len(self.calls)}", "name": "", "arguments": ""}
            self.calls.append(found)
        name = getattr(item, "name", None) or ""
        if isinstance(name, str) and name:
            found["name"] += name
        arguments = getattr(item, "arguments", None)
        if isinstance(arguments, dict):
            found["arguments"] = json.dumps(arguments)
        elif isinstance(arguments, str) and arguments:
            found["arguments"] += arguments

    def _end(self) -> None:
        if self.ended:
            return
        self.ended = True
        if not self.saw_reply:
            print(f"\n--- request {self.turn} reply ---", file=self.sink, flush=True)
        named = [row for row in self.calls if row["name"]]
        if named:
            print(f"\n--- request {self.turn} tool_calls ---", file=self.sink, flush=True)
            for row in named:
                print(f"{row['name']} {row['arguments']}", file=self.sink, flush=True)
        print(file=self.sink, flush=True)


def print_updates(updates: Iterable[Any], sink: TextIO) -> None:
    """Print one stream of framework updates. Tests use this without a listener."""
    printer = TurnPrinter(sink)
    for update in updates:
        printer.take(update)
    printer.close()


def _has_body(update: Any) -> bool:
    if _reasoning(update):
        return True
    text = getattr(update, "text", "")
    if isinstance(text, str) and text:
        return True
    return any(
        getattr(item, "type", None) == "function_call"
        for item in getattr(update, "contents", None) or []
    )


def _reasoning(update: Any) -> str:
    parts = [
        item.text
        for item in getattr(update, "contents", None) or []
        if getattr(item, "type", None) == "text_reasoning"
        and isinstance(getattr(item, "text", None), str)
        and item.text
    ]
    return "".join(parts)
