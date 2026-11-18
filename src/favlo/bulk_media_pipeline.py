"""Lowest media admission rank.

A served video handler inherits this class. Image, music, and voice stay on
``MediaPipeline``. The HTTP kind, hold key, and slice loop stay those of the
parent. The scheduler puts this instance on the bulk deque.
"""

from __future__ import annotations

from favlo.media_pipeline import MediaPipeline


class BulkMediaPipeline(MediaPipeline):
    """Interactive media stays ahead. A second bulk job waits on the deque."""

    _seam_only = True
    admission = frozenset({"bulk"})
    preempts_media = False
    preempts_bulk = False
    drop_if_disconnected = False
    debounce_after_release = False
    resident = "exclusive"
    waits_for_memory = True
