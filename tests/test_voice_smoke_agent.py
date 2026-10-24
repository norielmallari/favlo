"""Format check for the voice harness. The listener is not started."""

from __future__ import annotations

import json

from harness.voice_smoke_agent import report_ok

from favlo.cli import load_pipeline, repo_root
from favlo.mcp_server import media_kinds

_REPORT = "rate 48000\nchannels 1\nseconds 1.200\nrms 0.010000\n"


def test_a_48k_clip_passes() -> None:
    assert report_ok(_REPORT, rate=48000, max_seconds=8)


def test_a_44k_clip_fails() -> None:
    text = _REPORT.replace("rate 48000", "rate 44100")
    assert not report_ok(text, rate=48000, max_seconds=8)


def test_smoke_note_names_the_model_and_rate() -> None:
    note = json.loads((repo_root() / "docs" / "voxcpm2" / "smoke.json").read_text(encoding="utf-8"))
    assert isinstance(note["input"], str) and note["input"].strip()
    assert isinstance(note["instructions"], str)
    assert note["model"] == "VoxCPM2"
    assert note["sample_rate"] == 48000
    assert float(note["max_seconds"]) > 0


def test_the_entry_serves_voice() -> None:
    module = load_pipeline("voxcpm2", runtime="metal41")
    assert media_kinds(module) == ["voice"]
    assert not callable(getattr(module, "list_voices", None))
    assert not callable(getattr(module, "list_emotions", None))
