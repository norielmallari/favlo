"""Qwen3.8-27B pipeline. It does not import the Flash Next package."""

from ternary_bonsai_2.commands import main
from ternary_bonsai_2.runtime import handle_chat, open_runtime

__all__ = ["handle_chat", "main", "open_runtime"]
