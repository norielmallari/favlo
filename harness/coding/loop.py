"""One Agent Framework run against ``serve``. The score is the hidden suite."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from harness.benchmarks.serve import Request, Server
from harness.coding.agent_tools import coding_tools
from harness.coding.session import session_for
from harness.coding.workspace import PROMPT, prepare_workspace
from harness.file_session import RequestRate, parse_requests, pipeline_log
from harness.turn_printer import TurnPrinter

# One model call per tool step, then status polls, then a stop.
SESSION_ROUNDS = 16


def agent_options(
    request: Request, *, mtp: bool, mtp_depth: int, store: bool = False
) -> dict[str, object]:
    """Fields the framework client sends. Sampling stays off the body.

    ``store`` is the value for the whole run. False makes the client replay
    the transcript. True sends ``previous_response_id`` and the listener
    chain rebuilds that transcript before the pipeline encodes.
    """
    return {
        "store": store,
        "reasoning": {"effort": "medium" if request.enable_thinking else "none"},
        "extra_body": {
            "favlo:mtp": bool(mtp),
            "favlo:mtp_depth": int(mtp_depth),
        },
    }


_PREAMBLE = (
    "vision ",
    "prefix store ",
    "prefix load ",
    "waiting ",
    "loaded ",
    "released ",
)


def _prompt_count(line: str) -> int:
    parts = line.split()
    if len(parts) == 3 and parts[2].isdigit():
        return int(parts[2])
    return 0


def ledger_log(text: str) -> str:
    """The growing transcript on a shared model log.

    A second upload of one still logs that still twice, as ``vision cached`` for
    the copy already in the prefix and ``vision cache`` for the copy past the
    hit. The chain runs from that still's tower through the repeated upload. A later request
    with a shorter prompt is another client. Lines after one prompt that
    belong to the next request stay with that next request.
    """
    buckets: list[list[str]] = []
    bucket: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("prompt tokens ") and bucket:
            buckets.append(bucket)
            bucket = [line]
        else:
            bucket.append(line)
    if bucket:
        buckets.append(bucket)
    blocks: list[tuple[int, list[str]]] = []
    carry: list[str] = []
    for item in buckets:
        if not item[0].startswith("prompt tokens "):
            carry = item
            continue
        body = carry + item
        carry = []
        end = len(body)
        while end > 1 and body[end - 1].startswith(_PREAMBLE):
            end -= 1
        blocks.append((_prompt_count(item[0]), body[:end]))
        carry = body[end:]
    if carry and blocks:
        blocks[-1][1].extend(carry)
    if not blocks:
        return ""
    start, end = _still_span(blocks)

    chain = [blocks[start]]
    for index in range(start + 1, end + 1):
        prompt, lines = blocks[index]
        if prompt < chain[-1][0]:
            continue
        chain.append((prompt, lines))
    return "\n".join(line for _prompt, lines in chain for line in lines)


def _cache_counts(lines: list[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in lines:
        token = ""
        for prefix in ("vision cached ", "vision cache "):
            if line.startswith(prefix):
                token = line.removeprefix(prefix).strip()
                break
        if not token.isdigit():
            continue
        count = int(token)
        counts[count] = counts.get(count, 0) + 1
    return counts


def _still_span(blocks: list[tuple[int, list[str]]]) -> tuple[int, int]:
    """Tower index and repeated-upload index for one still.

    The repeated upload names the same still twice, as ``vision cached`` or
    ``vision cache``. The tower is the
    latest earlier block that merged that still. Without a repeated upload,
    the span is the whole log.
    """
    end: int | None = None
    still: int | None = None
    for index, (_prompt, lines) in enumerate(blocks):
        for count, times in _cache_counts(lines).items():
            if times >= 2:
                end = index
                still = count
    if end is None or still is None:
        return 0, len(blocks) - 1
    start = end
    marker = f"vision merged {still}"
    for index in range(end + 1):
        if any(line == marker for line in blocks[index][1]):
            start = index
    return start, end


def prefix_problems(requests: list[RequestRate]) -> list[str]:
    """Problems when the restart did not load the prefix or later turns drifted.

    A later turn may reload the prompt-boundary file when the chat template
    re-encodes a tool call differently from the drawn tokens. That reload is
    still a held prefix when the hit equals that boundary or the prior applied
    length. A cache miss after the first request is not.
    """
    if len(requests) < 2:
        return ["the ledger made fewer than two requests"]
    if not any(request.prefix_load is not None for request in requests[1:]):
        return ["prefix was not loaded from disk"]
    if not coding_prefix_holds(requests):
        return ["prefix did not extend"]
    return []


def coding_prefix_holds(requests: list[RequestRate]) -> bool:
    """True when the restart loaded from disk and later turns kept a cache hit."""
    from harness.file_session import _applied, _cold_start, _extends, _stored_load

    if not requests:
        return False
    first = requests[0]
    if not (_cold_start(first) or _stored_load(first)):
        return False
    for previous, current in zip(requests, requests[1:], strict=False):
        if current.missed or current.prefix_hit is None:
            return False
        if current.prefix_load is not None:
            if not _stored_load(current):
                return False
            if current.prefix_hit not in {_applied(previous), previous.prompt_tokens}:
                return False
            continue
        if not _extends(previous, current):
            return False
    return True


def rebind_chat_client(client: Any, port: int) -> None:
    """Point the framework client at the restarted listener."""
    url = f"http://127.0.0.1:{port}/v1"
    client.base_url = url
    http = getattr(client, "client", None)
    if http is not None:
        http.base_url = url


def run_session(
    server: Server,
    model: str,
    request: Request,
    workspace: Path,
    *,
    mtp: bool,
    mtp_depth: int = 3,
    vision: bytes | None = None,
    store: bool = False,
) -> tuple[list[str], list[str]]:
    """Let the agent use the tools. Return the calls that succeeded and any problems."""
    prepare_workspace(workspace)
    session = session_for(workspace)
    problems: list[str] = []
    try:
        asyncio.run(
            _run(
                server,
                model,
                request,
                session,
                mtp=mtp,
                mtp_depth=mtp_depth,
                vision=vision,
                store=store,
                problems=problems,
            )
        )
        text = ledger_log(pipeline_log(server.log.text(), model))
        problems.extend(prefix_problems(parse_requests(text)))
        if vision is not None:
            from harness.coding.compliance import vision_problems

            problems.extend(vision_problems(text))
        return list(session.calls), problems
    finally:
        session.close()


async def _run(
    server: Server,
    model: str,
    request: Request,
    session: Any,
    *,
    mtp: bool,
    mtp_depth: int,
    vision: bytes | None,
    store: bool,
    problems: list[str],
) -> None:
    from agent_framework import Agent, ChatMiddleware

    from favlo.chat_client import close_chat_client, open_chat_client

    client = open_chat_client(model.partition(":")[0], f"http://127.0.0.1:{server.port}/v1")
    client.function_invocation_configuration["max_iterations"] = SESSION_ROUNDS
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = SESSION_ROUNDS
    client.function_invocation_configuration["allow_concurrent_invocation"] = False
    client.function_invocation_configuration["include_detailed_errors"] = True
    options = agent_options(request, mtp=mtp, mtp_depth=mtp_depth, store=store)
    restarter = _PrefixRestart(server, client, model)

    class PrefixRestartMiddleware(ChatMiddleware):
        """Unload the resident model after the first turn so the next loads from SSD."""

        async def process(self, _context: Any, call_next: Any) -> None:
            if restarter.calls == 1:
                print("unloading model", flush=True)
                restarter.restart()
            await call_next()
            restarter.calls += 1

    agent = Agent(
        client=client,
        name="coding",
        instructions=(
            "You are a coding assistant. Carry out the next step, then wait "
            "for its result. Leave the reply empty."
        ),
        tools=coding_tools(session),
        default_options=options,
        middleware=[PrefixRestartMiddleware()],
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
        if vision is not None and chat is not None:
            await _describe(agent, client, vision, options, chat, printer, problems)
    finally:
        await close_chat_client(client)


class _PrefixRestart:
    """Unload the resident model once, keeping the listener and the followed log."""

    def __init__(self, server: Server, client: Any, model: str) -> None:
        self.server = server
        self.client = client
        self.model = model
        self.calls = 0

    def restart(self) -> None:
        _wait_latest_decode(self.server, self.model)
        self.server.restart()
        rebind_chat_client(self.client, self.server.port)


def decode_ready(requests: list[RequestRate]) -> bool:
    """True when the finished request has its decode line.

    A later request that has not logged prefill is another client still
    inside prefill. Its open row does not hold the restart.
    """
    if not requests:
        return False
    if requests[-1].decode:
        return True
    return len(requests) >= 2 and bool(requests[-2].decode) and not requests[-1].prefill


def _wait_latest_decode(server: Server, model: str, *, timeout: float = 30.0) -> None:
    """Block until this model's finished request has a decode line in the log.

    The prefix file is written before that line. Stopping earlier can drop the
    decode telemetry the extension check needs for the applied token count.
    Another pipeline's decode line does not count.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if decode_ready(parse_requests(pipeline_log(server.log.text(), model))):
            return
        time.sleep(0.05)
    message = "server log did not record decode before restart"
    raise TimeoutError(message)


async def _describe(
    agent: Any,
    client: Any,
    png: bytes,
    options: dict[str, object],
    chat: Any,
    printer: TurnPrinter,
    problems: list[str],
) -> None:
    """One still chat. Tools stay off so the tower is not mixed with a tool call."""
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
