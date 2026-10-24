"""One spoken line from the packed record. Skips when that record is missing.

The check is the no-reference speech signals in ``zonos2.signals``: level,
clipping, duration, energy from 80 Hz to 4 kHz, spectral flatness in that
band, and a pitch that moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zonos2.audio import encode_wav
from zonos2.runtime import open_runtime
from zonos2.signals import voice_ok

_RECORD = Path("models/ZONOS2-fp8-fp4")
_SENTENCE = Path("docs/zonos2/smoke.json")


def test_smoke_sentence_is_voiced():
    if not (_RECORD / "manifest.json").is_file() or not (_RECORD / "weights").is_dir():
        pytest.skip("packed record is missing")
    note = json.loads(_SENTENCE.read_text(encoding="utf-8"))
    runtime = open_runtime(_RECORD, kv_cache="e4m3", console=None, resident_percent=100.0)
    samples = runtime.speak(note["input"], max_frames=400)
    path = Path("artifacts/zonos2-smoke.wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_wav(samples))
    ok, report = voice_ok(samples)
    assert ok, report
