"""One short line through the VoxCPM2 decoder."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from favlo.registry import choose_complete
from voxcpm2.metal41.runtime import open_device
from voxcpm2.portable.audio import encode_wav

_PIPELINE = "voxcpm2"


def _installed_record() -> Path | None:
    try:
        path, _selection = choose_complete(
            model=None, record=None, pipeline=_PIPELINE
        )
    except LookupError:
        return None
    if not (path / "manifest.json").is_file():
        return None
    if not (path / "weights.bin").is_file() and not (path / "model.safetensors").is_file():
        return None
    return path


def test_basic_line_writes_a_48k_wav(tmp_path: Path) -> None:
    record = _installed_record()
    if record is None:
        pytest.skip("voxcpm2 registry default is not installed")
    runtime = open_device(record, owner="voxcpm2:smoke")
    try:
        samples = runtime.speak(
            "Hello.",
            persona="Clear adult voice.",
            max_len=3,
            min_len=2,
            steps=10,
            seed=0,
        )
        rate = runtime.config.out_sample_rate
        hop = math.prod(runtime.config.decoder_rates)
        frames = 3 * runtime.config.patch_size
    finally:
        runtime.close()
    assert samples.dtype == np.float32
    assert samples.shape == (frames * hop,)
    assert np.isfinite(samples).all()
    assert float(np.max(np.abs(samples))) <= 1.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    assert rms > 1e-3
    path = tmp_path / "line.wav"
    path.write_bytes(encode_wav(samples, rate))
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 48000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == samples.shape[0]
