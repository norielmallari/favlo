"""Stored MP4 expiry. No device and no checkpoint."""

import json
import os

from favlo.logs import settings_path
from favlo.videos import (
    load_video,
    source_path,
    store_source,
    store_video,
    sweep_videos,
    video_path,
)


def test_a_fresh_clip_stays_and_an_old_one_is_removed(tmp_path) -> None:
    settings_path(tmp_path).write_text(
        json.dumps({"video_ttl_seconds": 60}),
        encoding="utf-8",
    )
    videos = {"video_old": {"status": "completed"}, "video_new": {"status": "completed"}}
    store_source("video_old", b"RIFF-old", home=tmp_path)
    store_video("video_old", b"old-mp4", home=tmp_path, videos=videos)
    store_video("video_new", b"new-mp4", home=tmp_path, videos=videos)
    old = video_path(tmp_path, "video_old")
    old_avi = source_path(tmp_path, "video_old")
    fresh = video_path(tmp_path, "video_new").stat().st_mtime
    expired = fresh - 120
    variant = old.with_name("video_old.max.mp4")
    variant.write_bytes(b"max")
    os.utime(old, (expired, expired))
    sweep_videos(videos, home=tmp_path, now=fresh)
    assert not old.is_file()
    assert not old_avi.is_file()
    assert not variant.is_file()
    assert "video_old" not in videos
    assert load_video("video_new", home=tmp_path) == b"new-mp4"
    assert "video_new" in videos


def test_a_path_in_the_id_is_refused(tmp_path) -> None:
    try:
        store_video("../clip", b"nope", home=tmp_path)
    except ValueError as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("path id was stored")
    assert load_video("../clip", home=tmp_path) is None


def test_another_quality_encodes_the_saved_avi_once(tmp_path, monkeypatch) -> None:
    store_source("video_a", b"RIFF", home=tmp_path)
    store_video("video_a", b"high-mp4", home=tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        with open(command[-1], "wb") as handle:
            handle.write(b"max-mp4")

        class _Result:
            returncode = 0
            stderr = b""

        return _Result()

    monkeypatch.setattr("favlo.videos.subprocess.run", fake_run)
    assert load_video("video_a", home=tmp_path, quality="max") == b"max-mp4"
    assert load_video("video_a", home=tmp_path, quality="max") == b"max-mp4"
    assert len(calls) == 1
    assert calls[0][calls[0].index("-crf") + 1] == "14"
    assert load_video("video_a", home=tmp_path) == b"high-mp4"
