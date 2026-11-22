"""Wan2.2 TI2V-5B CUDA pipeline."""

from __future__ import annotations

from wan22_ti2v_5b.cuda.runtime import list_resolutions, pipeline, video_modes

PIPELINE = "wan22_ti2v_5b"

handle_video = pipeline
main = pipeline.main
open_runtime = pipeline.open_runtime


def mcp_tools():
    return pipeline.mcp_tools()


__all__ = [
    "PIPELINE",
    "handle_video",
    "list_resolutions",
    "main",
    "mcp_tools",
    "open_runtime",
    "pipeline",
    "video_modes",
]
