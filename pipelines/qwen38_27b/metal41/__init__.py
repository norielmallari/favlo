"""Qwen3.8-27B pipeline. It does not import the Flash Next package."""


PIPELINE = "qwen38_27b"


from qwen38_27b.metal41.commands import main
from qwen38_27b.metal41.runtime import handle_responses, open_runtime

__all__ = ["handle_responses", "main", "open_runtime"]
