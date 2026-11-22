"""Request, schedule, and pack checks. They do not dispatch Metal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from favlo.media_pipeline import MediaRequestError
from klein_4b.portable import mcp_tools
from klein_4b.portable.formats import pack_matrix, reconstruct_row
from klein_4b.portable.pipeline import Pipeline
from klein_4b.portable.pixels import fit_input, snap_crop
from klein_4b.portable.png import decode_png, encode_png
from klein_4b.portable.prompt import wrap
from klein_4b.portable.rope import dit_table, latent_ids, text_ids
from klein_4b.portable.schedule import empirical_mu, sigmas

_HORSE = Path(__file__).resolve().parent / "assets" / "favlo-horse.png"


def test_schedule_ends_at_zero_and_shrinks() -> None:
    values = sigmas(720, 4)
    assert values.shape == (5,)
    assert values[-1] == pytest.approx(0.0)
    assert values[0] == pytest.approx(1.0)
    assert np.all(np.diff(values.astype(np.float64)) < 0)
    m_200 = 0.00016927 * 720 + 0.45666666
    m_10 = 8.73809524e-05 * 720 + 1.89833333
    slope = (m_200 - m_10) / 190.0
    assert empirical_mu(720, 4) == pytest.approx(slope * 4 + (m_200 - 200.0 * slope))
    assert empirical_mu(5000, 4) == pytest.approx(0.00016927 * 5000 + 0.45666666)


def test_int8_roundtrip_stays_within_one_code() -> None:
    weight = np.random.default_rng(3).normal(0, 0.4, size=(5, 130)).astype(np.float32)
    weight[1] = 0
    codes, scales = pack_matrix(weight)
    assert codes.shape[0] % 128 == 0 and codes.shape[1] % 128 == 0
    assert scales[1] == 0
    restored = reconstruct_row(codes, scales)[:5, :130]
    peak = np.max(np.abs(weight), axis=1)
    error = np.max(np.abs(restored - weight), axis=1)
    assert np.all(error <= peak / 127 + 1e-5)


def test_prompt_wrap_closes_the_think_block() -> None:
    text = wrap("a red saddle")
    assert text.startswith("<|im_start|>user\n")
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in text


def test_rope_ids_follow_the_grid() -> None:
    ids = latent_ids(2, 3, 10)
    assert ids.shape == (6, 4)
    assert ids[0].tolist() == [10, 0, 0, 0]
    assert ids[4].tolist() == [10, 1, 1, 0]
    assert text_ids(4)[:, 3].tolist() == [0, 1, 2, 3]
    cos, sin = dit_table(ids)
    assert cos.shape == sin.shape == (6, 128)


def test_parse_draws_from_a_prompt_or_edits_a_still() -> None:
    pipeline = Pipeline()
    with pytest.raises(MediaRequestError) as missing:
        pipeline.parse({"image": "file:///still.png"})
    assert missing.value.param == "prompt"
    drawn = pipeline.parse({"prompt": " a tan horse "})
    assert drawn.prompt == "a tan horse"
    assert drawn.images == ()
    assert drawn.width == 1024 and drawn.height == 1024
    assert drawn.steps == 4
    with pytest.raises(MediaRequestError) as blank:
        pipeline.parse({"prompt": "add a saddle", "image": "  "})
    assert blank.value.param == "image"
    request = pipeline.parse(
        {"prompt": " add a saddle ", "image": "file:///still.png", "size": "576x320", "steps": 4, "seed": 0}
    )
    assert request.prompt == "add a saddle"
    assert request.width == 576 and request.height == 320
    with pytest.raises(MediaRequestError):
        pipeline.parse({"prompt": "x", "image": "file:///still.png", "size": "100x100"})
    four = tuple(f"file:///still-{index}.png" for index in range(4))
    listed = pipeline.parse({"prompt": "add a saddle", "image": list(four)})
    assert listed.images == four
    with pytest.raises(MediaRequestError) as extra:
        pipeline.parse({"prompt": "add a saddle", "image": [*four, "file:///still-4.png"]})
    assert extra.value.param == "image"


def test_tool_descriptions_fit_the_listener() -> None:
    tools = list(mcp_tools())
    assert [tool.name for tool in tools] == ["generate_image", "edit_image"]
    for tool in tools:
        assert tool.handler == "handle_image"
        assert tool.result == "image"
        assert len(tool.description) <= 256
        assert "prompt" in tool.parameters["required"]
        for prop in tool.parameters["properties"].values():
            assert len(prop["description"]) <= 48
    generate, edit = tools
    assert "image" not in generate.parameters["required"]
    assert "image" in edit.parameters["required"]
    image = edit.parameters["properties"]["image"]
    assert image["oneOf"][1]["maxItems"] == 4


def test_snap_crop_does_not_scale() -> None:
    image = np.zeros((1900, 1070, 3), dtype=np.uint8)
    image[10, 10] = (1, 2, 3)
    cropped = snap_crop(image)
    assert cropped.shape == (1888, 1056, 3)
    assert cropped[4, 3].tolist() == [1, 2, 3]
    aligned = np.zeros((320, 576, 3), dtype=np.uint8)
    assert snap_crop(aligned) is aligned


def test_fit_input_shrinks_long_side() -> None:
    tall = np.zeros((1920, 1072, 3), dtype=np.uint8)
    fitted = fit_input(tall)
    assert fitted.shape == (1024, 560, 3)
    small = np.zeros((320, 576, 3), dtype=np.uint8)
    assert fit_input(small) is small


def test_png_roundtrip_and_horse() -> None:
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[1, 2] = (10, 20, 30)
    assert np.array_equal(decode_png(encode_png(image)), image)
    if not _HORSE.is_file():
        return
    horse = decode_png(_HORSE.read_bytes())
    assert horse.shape == (320, 576, 3)
    assert horse.dtype == np.uint8
