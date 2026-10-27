"""Stream a stored image, video, or audio file named by a favlo:// link."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

from favlo.images import image_path
from favlo.logs import favlo_home
from favlo.music import load_music
from favlo.videos import video_path
from favlo.voice import load_voice
from web_api.http import json_body

router = APIRouter()

_AUDIO = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}


@router.get("/media/{kind}/{filename}")
async def read_media(kind: str, filename: str):
    response = media_response(kind, filename, favlo_home())
    if response is None:
        return json_body(404, {"error": "not_found", "message": "That file is missing."})
    return response


def media_response(kind: str, filename: str, home: Path) -> Response | None:
    parsed = _stem(filename)
    if parsed is None:
        return None
    stem, ext = parsed
    if kind == "images" and ext == "png":
        return _file(image_path, home, stem, "image/png")
    if kind == "videos" and ext == "mp4":
        return _file(video_path, home, stem, "video/mp4")
    if kind == "voice" and ext in _AUDIO:
        return _audio(load_voice, home, stem, ext)
    if kind == "music" and ext in _AUDIO:
        return _audio(load_music, home, stem, ext)
    return None


def _file(path_for, home: Path, stem: str, media_type: str) -> FileResponse | None:
    try:
        path = path_for(home, stem)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return FileResponse(path, media_type=media_type)


def _audio(load, home: Path, stem: str, ext: str) -> Response | None:
    try:
        loaded = load(stem, ext, home=home)
    except ValueError:
        return None
    if loaded is None:
        return None
    body, media_type = loaded
    return Response(content=body, media_type=media_type)


def _stem(filename: str) -> tuple[str, str] | None:
    stem, dot, ext = filename.rpartition(".")
    if (
        not dot
        or not stem
        or not ext
        or stem in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "." in stem
        or "\x00" in filename
    ):
        return None
    return stem, ext
