"""Qwen3.6-35B-A3B CUDA device.

The pipeline entry is ``qwen36_35b_a3b.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from qwen36_35b_a3b.portable import pipeline

__all__ = ["pipeline"]
