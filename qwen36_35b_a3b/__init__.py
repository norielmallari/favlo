"""Qwen3.6-35B-A3B pipeline. It does not import another pipeline package."""

from qwen36_35b_a3b.commands import main
from qwen36_35b_a3b.runtime import handle_chat, open_runtime

__all__ = ["handle_chat", "main", "open_runtime"]
