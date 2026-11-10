"""Wan2.2 I2V pipeline."""

from __future__ import annotations

from wan22_i2v.metal41.chat import handle_chat, pipeline
from wan22_i2v.metal41.schedule import list_resolutions, video_modes

PIPELINE = "wan22_i2v"

handle_video = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = [
    "pipeline",
    "handle_chat",
    "handle_video",
    "list_resolutions",
    "main",
    "mcp_tools",
    "open_runtime",
    "video_modes",
]
