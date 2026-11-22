"""Wan2.2 I2V Metal device.

The pipeline entry is ``wan22_i2v.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from wan22_i2v.portable import (
    handle_chat,
    handle_video,
    list_resolutions,
    main,
    mcp_tools,
    open_runtime,
    pipeline,
    video_modes,
)

PIPELINE = "wan22_i2v"

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
