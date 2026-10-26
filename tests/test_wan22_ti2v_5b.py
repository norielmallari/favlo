"""Wan2.2 TI2V-5B shapes, knots, and the AVI container. No GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from wan22_ti2v_5b.activations import apply_rope
from wan22_ti2v_5b.avi import encode_avi
from wan22_ti2v_5b.png import decode_png, encode_png
from wan22_ti2v_5b.quant import pack_fp4, pack_int8, relative_rms, score_matrix
from wan22_ti2v_5b.request import VideoRequestError, list_resolutions, parse_request, video_modes
from wan22_ti2v_5b.runtime import _require_int8_pack
from wan22_ti2v_5b.schedule import (
    blend,
    image_mask,
    latent_grid,
    require_knots,
    text_mask,
    timesteps,
    token_count,
)
from wan22_ti2v_5b.vae import _patch, _unpatch


def test_video_modes_cover_text_and_image() -> None:
    assert video_modes() == ("i2v", "t2v")


def test_listed_resolutions_are_the_two_frames() -> None:
    assert list_resolutions() == [
        {"name": "1280x704", "orientation": "landscape", "size": "1280x704"},
        {"name": "704x1280", "orientation": "portrait", "size": "704x1280"},
    ]


def test_pixel_patch_is_color_then_width_then_height() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.float32)
    frame[0, 0] = (1.0, 0.0, 0.0)
    frame[0, 1] = (0.0, 1.0, 0.0)
    frame[1, 0] = (0.0, 0.0, 1.0)
    frame[1, 1] = (1.0, 1.0, 0.0)
    folded = _patch(frame)[:, 0, 0, 0]
    # index = color * 4 + width_offset * 2 + height_offset
    assert folded[0] == 1.0
    assert folded[1 * 4 + 1 * 2 + 0] == 1.0
    assert folded[2 * 4 + 0 * 2 + 1] == 1.0
    assert np.array_equal(_unpatch(folded.reshape(12, 1, 1, 1))[0], frame)


def test_rope_odd_half_keeps_the_original_even() -> None:
    rows = np.random.default_rng(0).standard_normal((8, 24 * 128)).astype(np.float32)
    got = apply_rope(rows, 2, 2)
    assert float(np.max(np.abs(got[0] - rows[0]))) == 0.0
    data = rows.reshape(-1, 24, 64, 2)
    out = got.reshape(-1, 24, 64, 2)
    even = float(data[1, 0, 43, 0])
    odd = float(data[1, 0, 43, 1])
    cosine = float(np.cos(1.0))
    sine = float(np.sin(1.0))
    assert abs(float(out[1, 0, 43, 0]) - (even * cosine - odd * sine)) < 1e-5
    assert abs(float(out[1, 0, 43, 1]) - (even * sine + odd * cosine)) < 1e-5


def test_knots_match_the_shift_5_row() -> None:
    got = timesteps()
    assert [round(value, 3) for value in got] == [1000.0, 960.0, 888.889, 727.273]
    require_knots(got)


def test_smoke_grid_is_540_tokens() -> None:
    assert latent_grid(9, 320, 576) == (3, 20, 36)
    assert token_count(9, 320, 576) == 540


def test_image_mask_is_zero_on_the_first_frame() -> None:
    mask = image_mask(3, 4, 6)
    assert mask.shape == (1, 3, 4, 6)
    assert float(mask[0, 0].max()) == 0.0
    assert float(mask[0, 1].min()) == 1.0
    assert float(text_mask(3, 4, 6).min()) == 1.0


def test_blend_keeps_the_image_on_frame_zero() -> None:
    latent = np.ones((2, 2, 1, 1), dtype=np.float32)
    image = np.full((2, 1, 1, 1), 4.0, dtype=np.float32)
    mask = image_mask(2, 1, 1)
    mixed = blend(latent, image, mask)
    assert float(mixed[0, 0, 0, 0]) == 4.0
    assert float(mixed[0, 1, 0, 0]) == 1.0


def test_pack_roundtrip_is_finite() -> None:
    weight = np.random.default_rng(0).standard_normal((32, 256)).astype(np.float32)
    error, perm = score_matrix(weight)
    assert perm.shape == (256,)
    assert np.isfinite(error)
    packed, scales, dequant = pack_fp4(weight[:, perm])
    assert packed.dtype == np.uint8
    assert scales.shape == (32, 8)
    assert dequant.shape == weight.shape


def test_int8_relative_rms_is_finite() -> None:
    weight = np.random.default_rng(1).standard_normal((128, 256)).astype(np.float32)
    packed, scales, dequant = pack_int8(weight)
    assert packed.dtype == np.int8
    assert packed.shape == (128, 256)
    assert scales.shape == (128,)
    assert int(packed.min()) >= -127
    assert np.isfinite(relative_rms(weight, dequant))


def test_duration_floors_to_four_n_plus_one() -> None:
    from wan22_ti2v_5b.schedule import frames_from_duration

    assert frames_from_duration(5.0) == 117
    body = parse_request({"prompt": "a pond", "duration": 5.0})
    assert body.frames == 117


def test_request_defaults_and_the_step_guard() -> None:
    body = parse_request({"prompt": "a pond"})
    assert (body.width, body.height, body.frames) == (1280, 704, 121)
    assert body.image is None
    short = parse_request({"prompt": "a pond", "size": "576x320", "frames": 9, "seconds": "5"})
    assert short.frames == 9
    try:
        parse_request({"prompt": "a pond", "steps": 40})
    except VideoRequestError as exc:
        assert "steps" in str(exc)
    else:
        raise AssertionError("steps 40 should fail")


def test_png_and_avi_roundtrip(tmp_path) -> None:
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    image[1, 2] = (10, 20, 30)
    restored = decode_png(encode_png(image))
    assert restored.shape == image.shape
    assert int(restored[1, 2, 0]) == 10
    frames = np.stack([image, image, image, image, image])
    path = tmp_path / "clip.avi"
    path.write_bytes(encode_avi(frames, fps=24))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    width, height, count = probe.stdout.strip().split(",")
    assert (width, height, count) == ("16", "8", "5")


def test_umt5_pack_must_be_int8(tmp_path: Path) -> None:
    pack = tmp_path / "umt5-xxl-int8"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps({"weight": "int8"}), encoding="utf-8")
    assert _require_int8_pack(pack) == pack
    bf16 = tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"
    bf16.write_bytes(b"")
    try:
        _require_int8_pack(bf16)
    except RuntimeError as exc:
        assert "int8" in str(exc)
    else:
        raise AssertionError("bf16 encoder was accepted")
