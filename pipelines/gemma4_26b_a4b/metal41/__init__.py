"""Gemma 4 26B-A4B Metal device.

The pipeline entry is ``gemma4_26b_a4b.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from gemma4_26b_a4b.portable import pipeline

__all__ = ["pipeline"]
