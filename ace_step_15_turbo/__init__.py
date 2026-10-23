"""ACE-Step 1.5 XL Turbo. Neural work stays in the Metal shaders."""

from __future__ import annotations

from typing import Any

__all__ = ["main", "open_runtime", "handle_chat", "handle_music"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from ace_step_15_turbo.commands import main

        return main
    if name in {"open_runtime", "handle_chat", "handle_music"}:
        from ace_step_15_turbo import runtime

        return getattr(runtime, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
