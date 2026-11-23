"""Qwen3.5-4B CUDA device.

The pipeline entry is ``qwen35_4b.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from qwen35_4b.portable import pipeline

__all__ = ["pipeline"]
