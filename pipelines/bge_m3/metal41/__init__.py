"""BGE-M3 Metal device.

The pipeline entry is ``bge_m3.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from bge_m3.portable import pipeline

__all__ = ["pipeline"]
