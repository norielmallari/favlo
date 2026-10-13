"""File tools driven by the MADMetal chat endpoint.

The agent creates a note, replaces its text, and confirms the file. The
workspace argument is the only directory the tools write.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import Field

FILENAME = "note.txt"
ORIGINAL = "The harbor log is open."
EDITED = "The harbor log is closed."


def _locate(workspace: Path, path: str) -> Path:
    root = workspace.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        message = "path leaves the workspace"
        raise ValueError(message)
    return candidate


def file_tools(workspace: Path, calls: list[str]) -> list[object]:
    """Create, edit, and confirm, recorded in ``calls`` in invocation order."""

    @tool(approval_mode="never_require")
    def create_file(
        path: Annotated[str, Field(description="Workspace-relative path of the new file.")],
        content: Annotated[str, Field(description="Exact text to write.")],
    ) -> str:
        """Create a file. Fails when the file already exists."""
        target = _locate(workspace, path)
        if target.exists():
            message = f"{path} already exists"
            raise FileExistsError(message)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        calls.append("create")
        return f"created {path}"

    @tool(approval_mode="never_require")
    def edit_file(
        path: Annotated[str, Field(description="Workspace-relative path of the file to edit.")],
        content: Annotated[
            str, Field(description="Exact text the file must contain after the edit.")
        ],
    ) -> str:
        """Replace the file text. Fails when the file does not exist."""
        target = _locate(workspace, path)
        if not target.is_file():
            message = f"{path} does not exist"
            raise FileNotFoundError(message)
        target.write_text(content)
        calls.append("edit")
        return f"edited {path}"

    @tool(approval_mode="never_require")
    def confirm_file(
        path: Annotated[str, Field(description="Workspace-relative path to read back.")],
    ) -> str:
        """Return the file text so the caller can confirm the edit."""
        text = _locate(workspace, path).read_text()
        calls.append("confirm")
        return text

    return [create_file, edit_file, confirm_file]


def calls_in_order(calls: list[str]) -> bool:
    """True when create, then edit, then confirm each occur."""
    expect = ["create", "edit", "confirm"]
    index = 0
    for name in calls:
        if name == expect[index]:
            index += 1
            if index == len(expect):
                return True
    return False


async def run_file_task(base_url: str, workspace: Path) -> tuple[list[str], str]:
    """Run one agent step per tool. Returns the tool order and the file text.

    Each step sees only the tool for that step. The sampler then has one legal
    function name once a call starts.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []
    tools = file_tools(workspace, calls)
    prompts = [
        (
            "create_file",
            f"Create {FILENAME} containing exactly this text and no extra characters: {ORIGINAL}",
        ),
        (
            "edit_file",
            f"Edit {FILENAME} so it contains exactly this text and no extra characters: {EDITED}",
        ),
        (
            "confirm_file",
            f"Confirm {FILENAME} containing exactly this path and no extra characters: {FILENAME}",
        ),
    ]
    client = OpenAIChatCompletionClient(
        model="qwen38-flash-next",
        api_key="madmetal",
        base_url=base_url,
    )
    client.function_invocation_configuration["max_iterations"] = 1
    client.function_invocation_configuration["max_consecutive_errors_per_request"] = 1
    for step_tool, (name, prompt) in zip(tools, prompts, strict=True):
        agent = Agent(
            client=client,
            name="file-agent",
            instructions="Call the tool once. Then stop.",
            tools=[step_tool],
            default_options={"temperature": 0, "max_tokens": 128},
        )
        await agent.run(
            prompt,
            options={
                "temperature": 0,
                "max_tokens": 128,
                "tool_choice": {"mode": "required", "required_function_name": name},
            },
        )
    note = workspace / FILENAME
    text = note.read_text() if note.is_file() else ""
    return calls, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create, edit, and confirm one file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8129/v1")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    calls, text = asyncio.run(run_file_task(args.base_url, args.workspace))
    if calls_in_order(calls) and text == EDITED:
        print(f"confirmed {FILENAME}")
        return 0
    print(f"calls {calls}")
    print(f"file {text!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
