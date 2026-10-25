"""Coherence gates for the 3 minute music harness. The listener is not started."""

from __future__ import annotations

import numpy as np
from harness.music_long_agent import SEAM_RATIO, boundary_ratio, report_problems

_REPORT = """\
rate 48000
channels 2
seconds 180.0000
rms 0.1200
spectral_flatness 0.0400
key D major correlation 0.8000 margin 0.2000 runner_up G major
quiet_slices 2/360
requested_key D major match
template_output
seconds 180.0000
key D major
bpm 96.0000
"""


def test_a_settled_report_passes() -> None:
    assert report_problems(_REPORT, 180) == []


def test_noise_and_an_unsettled_key_fail() -> None:
    text = _REPORT.replace("spectral_flatness 0.0400", "spectral_flatness 0.9000")
    text = text.replace("margin 0.2000", "margin 0.0010")
    text = text.replace("requested_key D major match", "requested_key D major miss")
    problems = report_problems(text, 180)
    assert any("noise" in item for item in problems)
    assert any("margin" in item for item in problems)
    assert any("missed" in item for item in problems)


def test_a_smooth_join_stays_under_the_seam_ceiling() -> None:
    rate = 48_000
    mono = np.sin(np.linspace(0, 40, rate * 3, dtype=np.float32))
    assert boundary_ratio(mono, rate) < SEAM_RATIO


def test_a_click_at_the_tile_join_fails() -> None:
    rate = 48_000
    mono = np.sin(np.linspace(0, 40, rate * 3, dtype=np.float32)) * 0.01
    mono[rate] += 1.0
    assert boundary_ratio(mono, rate) > SEAM_RATIO
