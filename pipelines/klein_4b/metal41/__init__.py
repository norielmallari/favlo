"""FLUX.2 [klein] 4B Metal device.

The pipeline entry is ``klein_4b.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from klein_4b.portable import pipeline

__all__ = ["pipeline"]
