"""CUDA opening math and the int4 row pack. No GPU."""

from __future__ import annotations

import numpy as np

from wan22_ti2v_5b.cuda.formats import pack_int4
from wan22_ti2v_5b.cuda.layout import (
    BUDGET,
    dit_bytes,
    fits,
    longest,
    opening_bytes,
    scratch_bytes,
    umt5_bytes,
)


def test_catalog_clip_fits_and_the_next_knot_does_not() -> None:
    frames, time = longest(704, 1280)
    assert frames == 485
    assert time == 122
    assert fits(frames, 704, 1280)
    assert opening_bytes(frames, 704, 1280) <= BUDGET
    assert not fits(489, 704, 1280)


def test_shorter_canvases_run_longer() -> None:
    wide, _wide_time = longest(480, 832)
    small, _small_time = longest(320, 576)
    assert wide == 1097
    assert small == 2385


def test_records_fit_one_at_a_time() -> None:
    assert dit_bytes() < BUDGET
    assert umt5_bytes() < BUDGET
    long_scratch = scratch_bytes(121, 704, 1280)
    assert dit_bytes() + umt5_bytes() + long_scratch > BUDGET


def test_int4_nibble_and_fp16_scale() -> None:
    weight = np.array([[7.0, -7.0, 0.0, 0.0]], dtype=np.float32)
    codes, scale = pack_int4(weight)
    assert codes.dtype == np.uint8
    assert codes.shape == (1, 128)
    assert scale.dtype == np.float16
    assert codes[0, 0] == 0x97
    assert float(scale[0]) == np.float16(1.0)
