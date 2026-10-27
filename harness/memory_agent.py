"""Exercise ``remember``, ``recall``, and ``forget`` through the default LLM.

A Microsoft Agent Framework agent on the registry default calls memory tools
that forward to ``POST /mcp``. The command does not take the machine lock. It
reuses ``127.0.0.1:8129`` or starts that listener and stops the process it
started. When this session starts the daemon, it sets ``FAVLO_MEMORY_HOME`` to a
temporary directory so the run does not write the default ``~/.favlo/memory.db``.
A reused listener keeps that process's memory file; sentences are tagged with a
run id and forgotten at the end. Embeddings use the same ``bge-m3-int8`` row as
``harness.embeddings_agent``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, tool
from pydantic import Field

from favlo.chat_client import open_chat_client
from harness.harness_model import judge_model
from harness.mcp_media import call_tool
from harness.server import close_listener, open_listener

CLIENT = "memory-harness"
HOST = "127.0.0.1"


def tool_value(result: dict[str, Any]) -> Any:
    """The payload from a completed memory tool call."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    if structured is not None:
        return structured
    content = result.get("content")
    if not isinstance(content, list):
        message = "the tool result has no content"
        raise RuntimeError(message)
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text.strip()
    message = "the tool result has no text block"
    raise RuntimeError(message)


def recall_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ``recall`` into a list of observation rows."""
    value = tool_value(result)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            message = value
            raise RuntimeError(message) from None
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
    message = "recall did not return a list"
    raise RuntimeError(message)


def remember_id(port: int, text: str) -> str:
    result = call_tool(
        port,
        "remember",
        {"text": text, "client": CLIENT},
    )
    if result.get("isError"):
        message = json.dumps(result)
        raise RuntimeError(message)
    mem_id = tool_value(result)
    if not isinstance(mem_id, str) or not mem_id.startswith("mem_"):
        message = f"remember returned {mem_id!r}"
        raise RuntimeError(message)
    return mem_id


def recall(port: int, query: str) -> list[dict[str, Any]]:
    result = call_tool(port, "recall", {"query": query})
    if result.get("isError"):
        message = json.dumps(result)
        raise RuntimeError(message)
    return recall_rows(result)


def forget(port: int, mem_id: str) -> None:
    result = call_tool(port, "forget", {"id": mem_id})
    if result.get("isError"):
        message = json.dumps(result)
        raise RuntimeError(message)
    text = tool_value(result)
    if not isinstance(text, str) or "forgot" not in text:
        message = f"forget returned {text!r}"
        raise RuntimeError(message)


def check_recall(
    rows: list[dict[str, Any]],
    *,
    want: str,
    absent: str,
) -> float:
    """``want`` is present and ranks above ``absent`` when both appear."""
    ids = [str(row["id"]) for row in rows if isinstance(row.get("id"), str)]
    if want not in ids:
        message = f"recall did not return {want}"
        raise RuntimeError(message)
    want_row = next(row for row in rows if row.get("id") == want)
    score = want_row.get("score")
    if absent not in ids:
        if not isinstance(score, (int, float)):
            return 0.0
        return float(score)
    absent_row = next(row for row in rows if row.get("id") == absent)
    other = absent_row.get("score")
    if isinstance(score, (int, float)) and isinstance(other, (int, float)):
        if float(score) <= float(other):
            message = "the matching memory did not outrank the other topic"
            raise RuntimeError(message)
        return float(score)
    if ids.index(want) > ids.index(absent):
        message = "the matching memory did not outrank the other topic"
        raise RuntimeError(message)
    return float(score) if isinstance(score, (int, float)) else 0.0


def memory_tools(port: int, trace: dict[str, Any]) -> list[object]:
    @tool(approval_mode="never_require")
    def remember(
        text: Annotated[str, Field(description="One sentence to store.")],
    ) -> str:
        """Store a fact for later recall."""
        mem_id = remember_id(port, text)
        trace.setdefault("remembered", []).append(mem_id)
        return mem_id

    @tool(approval_mode="never_require")
    def recall_memories(
        query: Annotated[str, Field(description="Natural-language search query.")],
    ) -> str:
        """Return matching memories as JSON."""
        rows = recall(port, query)
        trace.setdefault("recalls", []).append(rows)
        return json.dumps(rows)

    @tool(approval_mode="never_require")
    def forget_memory(
        mem_id: Annotated[str, Field(description="Id from remember or recall.")],
    ) -> str:
        """Soft-delete one memory by id."""
        forget(port, mem_id)
        trace.setdefault("forgot", []).append(mem_id)
        return f"forgot {mem_id}"

    return [remember, recall_memories, forget_memory]


async def _run_agent(
    base_url: str,
    model: str,
    tools: list[object],
    *,
    fact_a: str,
    fact_b: str,
    query: str,
) -> None:
    client = open_chat_client(model, base_url)
    client.function_invocation_configuration["max_iterations"] = 12
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 2
    instructions = (
        "You test long-term memory tools. Call remember, recall_memories, and "
        "forget_memory only. Use temperature 0 behavior: follow the steps exactly."
    )
    prompt = (
        f"1. remember this sentence: {fact_a}\n"
        f"2. remember the same sentence again and note whether the id matches\n"
        f"3. remember this different sentence: {fact_b}\n"
        f"4. recall_memories with query: {query}\n"
        f"5. forget_memory the id for the first sentence\n"
        f"6. recall_memories again with the same query\n"
        f"7. forget_memory the id for the second sentence\n"
        "Reply done when finished."
    )
    agent = Agent(
        client=client,
        name="memory-agent",
        instructions=instructions,
        tools=tools,
        default_options={"temperature": 0},
    )
    await agent.run(prompt, options={"temperature": 0})


def verify_run(port: int, run_id: str, trace: dict[str, Any]) -> tuple[str, str, float]:
    """Check the agent trace and confirm recall ranking on the harness side."""
    remembered = trace.get("remembered")
    if not isinstance(remembered, list) or len(remembered) < 3:
        message = "the agent did not call remember enough times"
        raise RuntimeError(message)
    mem_a, mem_a_dup, mem_b = remembered[0], remembered[1], remembered[2]
    if mem_a != mem_a_dup:
        message = "remember did not return the same id for duplicate text"
        raise RuntimeError(message)
    recalls = trace.get("recalls")
    if not isinstance(recalls, list) or len(recalls) < 2:
        message = "the agent did not call recall_memories twice"
        raise RuntimeError(message)
    score = check_recall(recalls[0], want=mem_a, absent=mem_b)
    forgot = trace.get("forgot")
    if not isinstance(forgot, list) or mem_a not in forgot:
        message = "the agent did not forget the first memory"
        raise RuntimeError(message)
    if mem_a in {row.get("id") for row in recalls[1]}:
        message = "forgot memory still appears in the agent's second recall"
        raise RuntimeError(message)
    query = f"Harness {run_id}: northern lights in Norway"
    if mem_a in {row.get("id") for row in recall(port, query)}:
        message = "forgot memory still appears in recall"
        raise RuntimeError(message)
    if mem_b not in forgot:
        forget(port, mem_b)
    return mem_a, mem_b, score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.memory_agent")
    parser.add_argument(
        "--keep-home",
        action="store_true",
        help="leave FAVLO_MEMORY_HOME unset and use the listener's existing database",
    )
    parser.add_argument("--model", default=None, help="chat model id; default is the registry default")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    model = args.model or judge_model(root)
    run_id = secrets.token_hex(4)
    fact_a = f"Harness {run_id}: The aurora borealis is visible in Tromsø in winter."
    fact_b = f"Harness {run_id}: Wheat prices rose after the dry season."
    query = f"Harness {run_id}: northern lights in Norway"
    temp_home: Path | None = None
    if not args.keep_home and not os.environ.get("FAVLO_MEMORY_HOME"):
        temp_home = Path(tempfile.mkdtemp(prefix="favlo-memory-harness-"))
        os.environ["FAVLO_MEMORY_HOME"] = str(temp_home)
    trace: dict[str, Any] = {}
    listener = open_listener()
    try:
        tools = memory_tools(listener.port, trace)
        asyncio.run(
            _run_agent(
                f"http://{HOST}:{listener.port}/v1",
                model,
                tools,
                fact_a=fact_a,
                fact_b=fact_b,
                query=query,
            )
        )
        mem_a, mem_b, score = verify_run(listener.port, run_id, trace)
    finally:
        close_listener(listener)
        if temp_home is not None:
            db = temp_home / "memory.db"
            if db.is_file():
                db.unlink()
            temp_home.rmdir()
    print(f"run {run_id} remembered {mem_a} {mem_b} score {score:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
