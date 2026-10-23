"""A music resource read returns the job's format and does not encode another."""

from __future__ import annotations

from favlo.mcp_server import stored_music


def test_wav_job_is_not_read_back_as_mp3(tmp_path) -> None:
    music_id = "music_clip"
    folder = tmp_path / "music"
    folder.mkdir()
    (folder / f"{music_id}.wav").write_bytes(b"RIFF")
    state = {
        "music_home": tmp_path,
        "music": {music_id: {"response_format": "wav"}},
    }
    loaded = stored_music(state, music_id)
    assert loaded == (b"RIFF", "audio/wav")
    assert not (folder / f"{music_id}.mp3").exists()
