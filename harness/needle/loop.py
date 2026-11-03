"""One Agent Framework run. The model may read the haystack and then report the number."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agent_framework import Agent

from favlo.chat_client import close_chat_client, open_chat_client
from harness.benchmarks.serve import Request, Server
from harness.needle.grade import problems
from harness.needle.haystack import FILE_NAME, QUESTION, Haystack
from harness.needle.read_tool import read_tool
from harness.turn_printer import TurnPrinter

_PAGE = (
    "You look up one number in a file. Use read_file on the file named in the "
    "question. A read returns at most a window of the file. Do not set limit. "
    "When the result includes next_offset and the window has no line that "
    "starts with 'The special magic number is ', read again from that offset. "
    "Reply with the number only after a read has shown that line. Do not guess."
)
_RETRIEVE = (
    "You look up one number in a file. Use read_file on the file named in the "
    "question. A read returns at most a window of the file. When the result "
    "includes next_offset, read again from that offset. Read until a result "
    "has no next_offset. Only then reply with the number."
)


def agent_options(request: Request, *, mtp: bool, mtp_depth: int) -> dict[str, object]:
    """Fields the framework client sends. Sampling stays off the body.

    ``store`` is false so each later read replays the windows already returned.
    """
    return {
        "store": False,
        "reasoning": {"effort": "medium" if request.enable_thinking else "none"},
        "extra_body": {
            "favlo:mtp": bool(mtp),
            "favlo:mtp_depth": int(mtp_depth),
        },
    }


def continue_offset(reads: list[dict[str, Any]], *, retrieve: bool, seen: set[int]) -> int | None:
    """Next page-mode offset after a miss.

    Retrieve stays one transcript. A page-mode miss continues from the last
    ``next_offset`` once. A repeated offset means the scan did not advance.
    """
    if retrieve or not reads:
        return None
    nxt = reads[-1].get("next_offset")
    if not isinstance(nxt, int) or nxt < 0 or nxt in seen:
        return None
    return nxt


def rounds_for(haystack: Haystack) -> int:
    """Tool rounds that can walk the file once, plus a few for the reply."""
    from harness.coding.tools.limits import READ_MAX_BYTES, READ_MAX_LINES

    lines = haystack.text.count("\n")
    by_lines = (lines + READ_MAX_LINES - 1) // READ_MAX_LINES
    by_bytes = (len(haystack.text.encode("utf-8")) + READ_MAX_BYTES - 1) // READ_MAX_BYTES
    return max(by_lines, by_bytes, 1) + 8


async def run_needle(
    server: Server,
    model: str,
    request: Request,
    workspace: Path,
    haystack: Haystack,
    *,
    mtp: bool,
    mtp_depth: int,
    mode: str = "page",
) -> list[str]:
    """Ask for the passkey. Return the grade problems."""
    retrieve = mode == "retrieve"
    reads: list[dict[str, Any]] = []
    client = open_chat_client(model, f"http://127.0.0.1:{server.port}/v1")
    rounds = rounds_for(haystack)
    client.function_invocation_configuration["max_iterations"] = rounds
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = rounds
    client.function_invocation_configuration["allow_concurrent_invocation"] = False
    client.function_invocation_configuration["include_detailed_errors"] = True
    options = agent_options(request, mtp=mtp, mtp_depth=mtp_depth)
    agent = Agent(
        client=client,
        name="needle",
        instructions=_RETRIEVE if retrieve else _PAGE,
        tools=[read_tool(workspace, reads)],
        default_options=options,
    )
    printer = TurnPrinter(sys.stdout)
    prompt = QUESTION
    seen: set[int] = set()
    found = ["reply missed the number"]
    try:
        while True:
            try:
                stream = agent.run(prompt, stream=True, options=agent.default_options)
                async for update in stream:
                    printer.take(update)
                printer.close()
                response = await stream.get_final_response()
            except Exception as exc:
                printer.close()
                if "input does not fit" not in str(exc):
                    raise
                response = None
            if response is not None:
                from harness.chat_agent import reply_parts

                _reasoning, reply = reply_parts(response)
                found = problems(
                    reply, haystack.number, haystack.line, reads, retrieve=retrieve
                )
                if not found:
                    break
            nxt = continue_offset(reads, retrieve=retrieve, seen=seen)
            if nxt is None or len(seen) >= rounds:
                if response is None:
                    found = ["the transcript did not fit"]
                break
            seen.add(nxt)
            prompt = (
                f"The number was not in the windows already read. "
                f"Read {FILE_NAME} from offset {nxt}. Do not set limit. "
                "Reply with the number only after a read shows the line "
                "'The special magic number is '."
            )
            agent = Agent(
                client=client,
                name="needle",
                instructions=_PAGE,
                tools=[read_tool(workspace, reads)],
                default_options=options,
            )
    finally:
        await close_chat_client(client)
    print(
        f"needle line {haystack.line} reads {len(reads)} file {FILE_NAME}",
        flush=True,
    )
    return found
