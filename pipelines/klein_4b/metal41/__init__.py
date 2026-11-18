"""FLUX.2 [klein] 4B image pipeline."""

from __future__ import annotations

from klein_4b.metal41.pipeline import handle_image, pipeline

PIPELINE = "klein_4b"

main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = ["pipeline", "handle_image", "main", "mcp_tools", "open_runtime"]
