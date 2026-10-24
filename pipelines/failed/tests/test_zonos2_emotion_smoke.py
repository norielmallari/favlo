"""A few ZONOS2 emotions on the first collection voice. Skips when the record is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.inspect_voice import format_report, report
from zonos2.audio import encode_wav
from zonos2.runtime import open_runtime
from zonos2.signals import voice_report
from zonos2.voices import list_voices, load

_RECORD = Path("models/ZONOS2-fp8-fp4")
_LINE = "The morning train arrived on time."
# The four named directions. Valence and arousal stay available as tags.
_NAMES = ("happy", "sad", "angry", "surprised")


def _voiced(samples) -> tuple[bool, dict[str, float]]:
    """The default level floor is 0.02. A quiet emotion can sit under that and still be speech."""
    report = voice_report(samples)
    ok = (
        0.75 <= report["seconds"] <= 12.0
        and report["rms"] >= 0.01
        and report["clip_fraction"] < 0.001
        and report["frames"] >= 8
        and 0.03 <= report["flatness"] <= 0.55
        and report["voiced"] >= 0.25
        and report["pitched"] >= 8
        and report["pitch_span"] >= 0.5
        and report["band"] >= 0.55
    )
    return ok, report


def test_representative_emotions_are_voiced():
    if not (_RECORD / "manifest.json").is_file() or not (_RECORD / "weights").is_dir():
        pytest.skip("packed record is missing")
    speaker = load(list_voices()[0]["id"])
    runtime = open_runtime(_RECORD, kv_cache="e4m3", console=None, resident_percent=100.0)
    try:
        neutral = runtime.speak(_LINE, speaker, max_frames=400)
        neutral_path = Path("artifacts/zonos2-neutral.wav")
        neutral_path.parent.mkdir(parents=True, exist_ok=True)
        neutral_path.write_bytes(encode_wav(neutral))
        ok, voiced = _voiced(neutral)
        assert ok, ("neutral", voiced)
        print(format_report(neutral_path, report(neutral, 44100)), flush=True)
        for name in _NAMES:
            samples = runtime.speak(f"[{name}] {_LINE}", speaker, max_frames=400)
            path = Path(f"artifacts/zonos2-emotion-{name}.wav")
            path.write_bytes(encode_wav(samples))
            ok, voiced = _voiced(samples)
            assert ok, (name, voiced)
            print(format_report(path, report(samples, 44100, neutral)), flush=True)
    finally:
        runtime.close()
