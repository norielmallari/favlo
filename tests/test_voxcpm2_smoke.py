"""One short line through the VoxCPM2 decoder."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from voxcpm2.audio import encode_wav
from voxcpm2.runtime import open_runtime

_CHECKOUT = Path("models/openbmb/VoxCPM2")


def test_basic_line_writes_a_48k_wav(tmp_path: Path) -> None:
    weights = _CHECKOUT / "model.safetensors"
    if not weights.is_file():
        pytest.skip("checkout missing")
    runtime = open_runtime(_CHECKOUT, owner="voxcpm2:smoke")
    try:
        samples = runtime.speak("Hello.", max_len=3, min_len=2, steps=10, seed=0)
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
