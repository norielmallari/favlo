"""Qwen3.8-27B Metal device.

The pipeline entry is ``qwen38_27b.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from qwen38_27b.portable import pipeline

__all__ = ["pipeline"]
