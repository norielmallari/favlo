"""Inkling 2 Mini Metal device.

The pipeline entry is ``inkling_2_mini.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from inkling_2_mini.portable import pipeline

__all__ = ["pipeline"]
