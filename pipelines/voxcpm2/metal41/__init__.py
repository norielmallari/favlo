"""VoxCPM2 Metal device.

The pipeline entry is ``voxcpm2.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from voxcpm2.portable import (
    handle_chat,
    handle_voice,
    main,
    mcp_tools,
    open_runtime,
    pipeline,
)

PIPELINE = "voxcpm2"

__all__ = ["pipeline", "handle_chat", "handle_voice", "main", "mcp_tools", "open_runtime"]
