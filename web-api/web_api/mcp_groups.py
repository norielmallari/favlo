"""Favlo media tools, grouped for one enable switch each."""

from __future__ import annotations

_GROUPS: dict[str, tuple[str, str]] = {
    "list_image_models": ("image", "Image"),
    "generate_image": ("image", "Image"),
    "list_video_models_i2v": ("video", "Video"),
    "list_video_models_t2v": ("video", "Video"),
    "generate_video_i2v": ("video", "Video"),
    "generate_video_t2v": ("video", "Video"),
    "list_music_models": ("music", "Music"),
    "generate_music": ("music", "Music"),
    "list_voice_models": ("voice", "Voice"),
    "generate_voice": ("voice", "Voice"),
    "media_status": ("media", "Media tasks"),
    "media_cancel": ("media", "Media tasks"),
    "remember": ("memory", "Memory"),
    "recall": ("memory", "Memory"),
    "forget": ("memory", "Memory"),
}

ORDER = ("image", "video", "music", "voice", "media", "memory", "tools")

LIST_BEFORE = {
    "generate_image": "list_image_models",
    "generate_video_i2v": "list_video_models_i2v",
    "generate_video_t2v": "list_video_models_t2v",
    "generate_music": "list_music_models",
    "generate_voice": "list_voice_models",
}


def group_for(name: str) -> tuple[str, str]:
    return _GROUPS.get(name, ("tools", "Tools"))
