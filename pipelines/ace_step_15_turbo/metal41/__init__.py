"""ACE-Step 1.5 XL Turbo. Neural work stays in the Metal shaders."""

from __future__ import annotations

from ace_step_15_turbo.metal41.runtime import handle_chat, pipeline

PIPELINE = "ace_step_15_turbo"

handle_music = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = ["pipeline", "handle_chat", "handle_music", "main", "mcp_tools", "open_runtime"]
