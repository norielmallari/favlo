"""YIN reports the fundamental when a harmonic is louder than it."""

from __future__ import annotations

import numpy as np

from harness.inspect_voice import fundamentals


def test_a_louder_second_harmonic_does_not_replace_the_fundamental():
    rate = 44100
    time = np.arange(rate, dtype=np.float64) / rate
    wave = 0.3 * np.sin(2.0 * np.pi * 100.0 * time)
    wave += 0.8 * np.sin(2.0 * np.pi * 200.0 * time)
    wave += 0.2 * np.sin(2.0 * np.pi * 300.0 * time)
    pitches = fundamentals(wave, rate)
    assert pitches.size > 8
    assert abs(float(np.median(pitches)) - 100.0) < 1.0


def test_a_pure_tone_reports_its_own_frequency():
    rate = 44100
    time = np.arange(rate, dtype=np.float64) / rate
    pitches = fundamentals(np.sin(2.0 * np.pi * 220.0 * time), rate)
    assert abs(float(np.median(pitches)) - 220.0) < 1.0
