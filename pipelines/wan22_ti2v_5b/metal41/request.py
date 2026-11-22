"""Request types live in ``wan22_ti2v_5b.portable.request``."""

from wan22_ti2v_5b.portable.request import (
    VideoRequest,
    VideoRequestError,
    list_resolutions,
    parse_request,
    video_modes,
)

__all__ = [
    "VideoRequest",
    "VideoRequestError",
    "list_resolutions",
    "parse_request",
    "video_modes",
]
