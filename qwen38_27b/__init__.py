"""Qwen3.8-27B pipeline. It does not import the Flash Next package."""

from qwen38_27b.commands import main
from qwen38_27b.runtime import open_runtime

__all__ = ["main", "open_runtime"]
