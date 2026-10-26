"""Recovery rules for the media pipeline harness. They do not open a listener."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

from ace_step_15_turbo.schedule import MIN_SECONDS
from wan22_i2v.schedule import frames_from_duration as i2v_frames
from wan22_ti2v_5b.request import list_resolutions
from wan22_ti2v_5b.schedule import frames_from_duration as ti2v_frames

import pytest

from harness.media_pipeline import (
    CLIP_SECONDS,
    FRAMES,
    HEIGHT,
    MUSIC_CHANNELS,
    MUSIC_RATE,
    TI2V_FRAMES,
    TI2V_HEIGHT,
    TI2V_SECONDS,
    TI2V_WIDTH,
    WIDTH,
    Probe,
    _PROBES,
    recovered,
    select_probes,
    status_record,
    wav_fields,
)


def test_half_a_second_is_five_frames_at_320p() -> None:
    assert i2v_frames(CLIP_SECONDS) == FRAMES
    assert (WIDTH, HEIGHT) == (576, 320)


def test_ti2v_duration_is_five_frames_at_the_served_size() -> None:
    assert ti2v_frames(TI2V_SECONDS) == TI2V_FRAMES
    sizes = {row["size"] for row in list_resolutions()}
    assert f"{TI2V_WIDTH}x{TI2V_HEIGHT}" in sizes


def test_music_smoke_is_the_shortest_clip() -> None:
    note = json.loads(Path("docs/ace-step-15-turbo/smoke.json").read_text(encoding="utf-8"))
    assert float(note["seconds"]) == MIN_SECONDS
    assert (MUSIC_RATE, MUSIC_CHANNELS) == (48000, 2)


def test_one_agent_covers_every_served_media_pipeline() -> None:
    assert _PROBES == ("image", "i2v", "t2v", "music", "voice")


def _catalog() -> list[Probe]:
    return [Probe(name, "", {}, "", "", "") for name in _PROBES]


def test_named_pipelines_keep_catalog_order() -> None:
    cases = _catalog()
    assert [case.name for case in select_probes(cases, ["voice", "i2v"])] == ["i2v", "voice"]
    assert [case.name for case in select_probes(cases, [])] == list(_PROBES)
    assert [case.name for case in select_probes(cases, None)] == list(_PROBES)


def test_an_unknown_pipeline_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown media pipeline"):
        select_probes(_catalog(), ["wan"])


def test_voice_hold_is_exported_with_the_other_media_holds() -> None:
    from favlo.pipeline_worker import _HOLDS as worker_holds
    from favlo.pipelines import _HOLDS as process_holds

    assert worker_holds == process_holds
    assert "voice_hold" in worker_holds


def test_a_pause_followed_by_completion_recovers() -> None:
    samples = [
        {"status": "working"},
        {"status": "working", "stage": "paused", "statusMessage": "task is ongoing"},
        {"status": "working", "stage": "high"},
        {"status": "completed", "taskId": "task"},
    ]
    assert recovered(samples)


def test_completion_without_a_pause_does_not_recover() -> None:
    assert not recovered([{"status": "working"}, {"status": "completed"}])


def test_a_failed_task_does_not_recover() -> None:
    samples = [
        {"status": "working", "stage": "paused"},
        {"status": "failed"},
    ]
    assert not recovered(samples)


def test_status_record_reads_a_paused_media_status() -> None:
    paused = {
        "taskId": "task",
        "status": "working",
        "stage": "paused",
        "statusMessage": "task is ongoing",
    }
    assert status_record({"structuredContent": paused}) == paused
    text = '{"status": "working", "stage": "paused"}'
    recorded = status_record({"content": [{"type": "text", "text": text}]})
    assert recorded["stage"] == "paused"


def test_wav_fields_reads_pcm() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x00\x10" * 4800)
    rate, channels, seconds, rms = wav_fields(buffer.getvalue())
    assert (rate, channels) == (48000, 1)
    assert abs(seconds - 0.1) < 1e-6
    assert rms > 0
