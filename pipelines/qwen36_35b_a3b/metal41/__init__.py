"""Qwen3.6-35B-A3B pipeline. It does not import another pipeline package."""


PIPELINE = "qwen36_35b_a3b"


from qwen36_35b_a3b.metal41.commands import main
from qwen36_35b_a3b.metal41.firing import idle_heartbeat
from qwen36_35b_a3b.metal41.runtime import handle_responses, open_runtime

__all__ = ["handle_responses", "idle_heartbeat", "main", "open_runtime"]
