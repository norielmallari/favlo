"""Z-Image (base) pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from z_image_base.portable.chat import handle_chat, pipeline

PIPELINE = "z_image_base"

handle_image = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = ["pipeline", "handle_chat", "handle_image", "main", "mcp_tools", "open_runtime"]
