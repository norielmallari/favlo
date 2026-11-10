"""VoxCPM2 voice pipeline."""

from __future__ import annotations

from voxcpm2.metal41.runtime import handle_chat, pipeline

PIPELINE = "voxcpm2"

handle_voice = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = ["pipeline", "handle_chat", "handle_voice", "main", "mcp_tools", "open_runtime"]
