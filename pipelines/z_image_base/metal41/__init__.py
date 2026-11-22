"""Z-Image (base) Metal device.

The pipeline entry is ``z_image_base.portable``. This package re-exports that
pipeline so a listener that still imports the device name serves the same
object. The device opens from ``portable`` ``open_runtime``.
"""

from z_image_base.portable import pipeline

__all__ = ["pipeline"]
