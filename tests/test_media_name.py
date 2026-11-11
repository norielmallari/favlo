"""A bare filename is a file in the Favlo kind folder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from favlo.media_name import (
    MissingMediaFile,
    is_plain_filename,
    locate_plain_filename,
    read_plain_filename,
    resolve_media_input,
)


def test_a_plain_filename_reads_the_kind_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("favlo.logs.favlo_home", lambda: tmp_path)
    images = tmp_path / "images"
    videos = tmp_path / "videos"
    audio = tmp_path / "audio"
    images.mkdir()
    videos.mkdir()
    audio.mkdir()
    (images / "still.png").write_bytes(b"png")
    (videos / "clip.mp4").write_bytes(b"mp4")
    (audio / "line.wav").write_bytes(b"wav")
    (audio / "song.mp3").write_bytes(b"mp3")
    assert read_plain_filename("still.png") == b"png"
    assert read_plain_filename("clip.mp4") == b"mp4"
    assert read_plain_filename("line.wav") == b"wav"
    assert read_plain_filename("song.mp3") == b"mp3"
    assert locate_plain_filename("still.png") == (images / "still.png").resolve()
    assert resolve_media_input(Path("clip.mp4")) == (videos / "clip.mp4").resolve()
    assert resolve_media_input(Path("/tmp/still.png")) == Path("/tmp/still.png")
    with pytest.raises(MissingMediaFile, match="missing"):
        read_plain_filename("gone.png")


def test_a_music_sweep_leaves_a_voice_clip(tmp_path: Path) -> None:
    from favlo.music import sweep_music
    from favlo.voice import sweep_voice

    audio = tmp_path / "audio"
    audio.mkdir()
    music = audio / "music_old.wav"
    voice = audio / "voice_old.wav"
    music.write_bytes(b"m")
    voice.write_bytes(b"v")
    old = 0.0
    os.utime(music, (old, old))
    os.utime(voice, (old, old))
    sweep_music(home=tmp_path, now=10_000_000)
    assert not music.exists()
    assert voice.is_file()
    sweep_voice(home=tmp_path, now=10_000_000)
    assert not voice.exists()


def test_a_path_or_an_id_is_not_a_plain_filename() -> None:
    assert is_plain_filename("still.png")
    assert not is_plain_filename("image_abc")
    assert not is_plain_filename("file:///absolute/still.png")
    assert not is_plain_filename("/tmp/still.png")
    assert not is_plain_filename("./still.png")
    assert not is_plain_filename("dir/still.png")
    assert not is_plain_filename("favlo://images/still.png")
    assert read_plain_filename("image_abc") is None
