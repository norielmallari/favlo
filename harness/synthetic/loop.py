"""One Agent Framework completion of a prepared prompt."""

from __future__ import annotations

import asyncio

from agent_framework import Agent

from favlo.chat_client import close_chat_client, open_chat_client
from harness.benchmarks.serve import Server, card_model_id


def agent_options(*, thinking: bool, mtp: bool, mtp_depth: int, decode: int) -> dict[str, object]:
    """Fields the framework client sends.

    Sampling stays on the card. ``max_tokens`` is the decode cap. ``store``
    is false so the next length is a new prompt, not a continuation.
    """
    return {
        "store": False,
        "max_tokens": int(decode),
        "reasoning": {"effort": "medium" if thinking else "none"},
        "extra_body": {
            "favlo:mtp": bool(mtp),
            "favlo:mtp_depth": int(mtp_depth),
        },
    }


async def complete(
    server: Server,
    model: str,
    text: str,
    *,
    thinking: bool,
    mtp: bool,
    mtp_depth: int,
    decode: int,
) -> None:
    """Send one user message and wait until the listener finishes it."""
    chat_model = card_model_id(model, thinking=thinking)
    client = open_chat_client(chat_model, f"http://127.0.0.1:{server.port}/v1")
    options = agent_options(thinking=thinking, mtp=mtp, mtp_depth=mtp_depth, decode=decode)
    agent = Agent(client=client, name="synthetic")
    try:
        await agent.run(text, options=options)
    finally:
        await close_chat_client(client)


def run_complete(
    server: Server,
    model: str,
    text: str,
    *,
    thinking: bool,
    mtp: bool,
    mtp_depth: int,
    decode: int,
) -> None:
    asyncio.run(
        complete(
            server,
            model,
            text,
            thinking=thinking,
            mtp=mtp,
            mtp_depth=mtp_depth,
            decode=decode,
        )
    )
