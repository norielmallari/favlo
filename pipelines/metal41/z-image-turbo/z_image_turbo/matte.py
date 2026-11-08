"""BiRefNet cutout for this pipeline.

The shared implementation is ``favlo.birefnet``. This module keeps the
import path the rest of the pipeline already uses.
"""

from favlo.birefnet import (
    EXPECTED,
    FILE,
    INPUT,
    MEAN,
    REPO,
    STD,
    URL,
    cutout,
    ensure_matte,
    matte_dir,
)

__all__ = [
    "EXPECTED",
    "FILE",
    "INPUT",
    "MEAN",
    "REPO",
    "STD",
    "URL",
    "cutout",
    "ensure_matte",
    "matte_dir",
]
