"""One agent checks the default chat model and the embedding model.

The agent must call ``check_embeddings``. That tool posts the six sentences
from ``harness.embeddings_agent`` to ``POST /v1/embeddings`` and requires
every paraphrase cosine to beat every cross-topic cosine. Exit 0 means the
tool ran and those checks passed. This command does not take the machine
lock and is not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agent_framework import Agent, tool
from openai import AsyncOpenAI

from favlo.chat_client import open_chat_client
from harness.embeddings_agent import (
    MODEL,
    SENTENCES,
    check_payload,
    check_scores,
    post_embedding,
)
from harness.harness_model import judge_model
from harness.server import close_listener, open_listener

HOST = "127.0.0.1"


def _tools(port: int, embed_model: str, ran: list[bool]) -> list[object]:
    @tool(approval_mode="never_require")
    def check_embeddings() -> str:
        """Score the six embedding sentences. Return when the pairs separate."""
        vectors: dict[str, list[float]] = {}
        for name, text in SENTENCES:
            payload = post_embedding(port, text, embed_model)
            check_payload(payload)
            row = payload["data"][0]["embedding"]
            if not isinstance(row, list):
                message = "embedding row is not a list"
                raise RuntimeError(message)
            vectors[name] = row
        scores = check_scores(vectors)
        for kind, left, right, score in scores:
            print(f"{kind} {left} {right} {score:.6f}", flush=True)
        ran.append(True)
        return "separated"

    return [check_embeddings]


def _agent(base_url: str, tools: list[object], model: str) -> int:
    try:
        asyncio.run(_run(base_url, tools, model))
    except Exception as exc:
        print(exc, flush=True)
        return 1
    return 0


async def _run(base_url: str, tools: list[object], model: str) -> None:
    http = AsyncOpenAI(api_key="favlo", base_url=base_url, timeout=None, max_retries=0)
    client = open_chat_client(model, base_url, async_client=http)
    client.function_invocation_configuration["max_iterations"] = 1
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    agent = Agent(
        client=client,
        name="language-pipeline",
        instructions="Call the named tool once. Then stop.",
        tools=tools,
        default_options={
            "temperature": 0,
            "extra_body": {"enable_thinking": False},
        },
    )
    await agent.run(
        "Check the embedding sentences.",
        options={
            "temperature": 0,
            "tool_choice": {"mode": "required", "required_function_name": "check_embeddings"},
            "extra_body": {"enable_thinking": False},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the default chat model and the embedding model."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--judge", default=None, help="chat model id; default is the registry default")
    parser.add_argument("--embed", default=MODEL, help="embedding model id")
    args = parser.parse_args(argv)
    judge_id = args.judge or judge_model(args.root.resolve())
    listener = None
    ran: list[bool] = []
    try:
        listener = open_listener()
        base = f"http://{HOST}:{listener.port}/v1"
        print("start", flush=True)
        failed = _agent(base, _tools(listener.port, args.embed, ran), judge_id)
        if failed != 0 or not ran:
            print("the language pipeline did not check embeddings", flush=True)
            return 1
        print("checked", flush=True)
        return 0
    finally:
        if listener is not None:
            close_listener(listener)


if __name__ == "__main__":
    raise SystemExit(main())
