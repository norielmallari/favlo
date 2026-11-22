"""FLUX.2 [klein] 4B pipeline entry. Device packages open from ``open_runtime``."""

from __future__ import annotations

from klein_4b.portable.pipeline import handle_image, pipeline

PIPELINE = "klein_4b"

main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = ["pipeline", "handle_image", "main", "mcp_tools", "open_runtime"]
