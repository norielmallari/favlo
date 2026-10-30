"""Favlo media tools, grouped for one enable switch each."""

from __future__ import annotations

_GROUPS: dict[str, tuple[str, str]] = {
    "list_image_models": ("image", "Image"),
    "set_image_model": ("image", "Image"),
    "generate_image": ("image", "Image"),
    "list_video_models_i2v": ("video", "Video"),
    "list_video_models_t2v": ("video", "Video"),
    "set_video_i2v_model": ("video", "Video"),
    "set_video_t2v_model": ("video", "Video"),
    "generate_video_i2v": ("video", "Video"),
    "generate_video_t2v": ("video", "Video"),
    "list_music_models": ("music", "Music"),
    "set_music_model": ("music", "Music"),
    "generate_music": ("music", "Music"),
    "list_voice_models": ("voice", "Voice"),
    "set_voice_model": ("voice", "Voice"),
    "generate_voice": ("voice", "Voice"),
    "media_status": ("media", "Media tasks"),
    "media_cancel": ("media", "Media tasks"),
    "remember": ("memory", "Memory"),
    "recall": ("memory", "Memory"),
    "forget": ("memory", "Memory"),
}

ORDER = ("image", "video", "music", "voice", "media", "memory", "tools")

GENERATE = {
    "generate_image": ("list_image_models", "set_image_model"),
    "generate_video_i2v": ("list_video_models_i2v", "set_video_i2v_model"),
    "generate_video_t2v": ("list_video_models_t2v", "set_video_t2v_model"),
    "generate_music": ("list_music_models", "set_music_model"),
    "generate_voice": ("list_voice_models", "set_voice_model"),
}


def group_for(name: str) -> tuple[str, str]:
    return _GROUPS.get(name, ("tools", "Tools"))
