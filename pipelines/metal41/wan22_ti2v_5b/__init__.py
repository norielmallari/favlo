"""Wan2.2 TI2V-5B text-to-video and image-to-video."""

from wan22_ti2v_5b.commands import main
from wan22_ti2v_5b.runtime import handle_chat, handle_video, open_runtime

__all__ = ["handle_chat", "handle_video", "main", "open_runtime"]
