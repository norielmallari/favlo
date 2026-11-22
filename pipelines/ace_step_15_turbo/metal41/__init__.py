"""ACE-Step 1.5 XL Turbo Metal device.

The pipeline entry is ``ace_step_15_turbo.portable``. This package re-exports
that pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from ace_step_15_turbo.portable import (
    handle_chat,
    handle_music,
    main,
    mcp_tools,
    open_runtime,
    pipeline,
)

PIPELINE = "ace_step_15_turbo"

__all__ = ["pipeline", "handle_chat", "handle_music", "main", "mcp_tools", "open_runtime"]
