"""Host checks for the FLUX.2 [klein] 4B CUDA package."""

from __future__ import annotations

import numpy as np

from klein_4b.cuda.formats import pack_int4, restore_int4
from klein_4b.cuda.pipeline import pipeline
from klein_4b.cuda.png import decode_png, encode_png
from klein_4b.cuda.prompt import wrap
from klein_4b.cuda.rope import apply_dit, dit_table, latent_ids
from klein_4b.cuda.schedule import empirical_mu, sigmas


def test_prompt_wraps_a_closed_think_turn() -> None:
    text = wrap("a horse")
    assert text.startswith("<|im_start|>user\n")
    assert "<think>\n\n</think>\n\n" in text
    assert "a horse" in text


def test_sigmas_end_at_zero() -> None:
    values = sigmas(4096, 4)
    assert values.shape == (5,)
    assert values[-1] == 0
    assert values[0] > values[1] > values[-2] > 0
    wide = empirical_mu(4301, 4)
    assert abs(wide - (0.00016927 * 4301 + 0.45666666)) < 1e-6


def test_int4_roundtrip_stays_inside_half_a_step() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((3, 5)).astype(np.float32)
    codes, scales, rows, inner = pack_int4(weight)
    restored = restore_int4(codes, scales)[:rows, :inner]
    step = scales[:rows, None] / 2
    assert np.max(np.abs(restored - weight)) <= float(np.max(step)) + 1e-5


def test_dit_rope_is_adjacent_pairs() -> None:
    ids = latent_ids(1, 1, 0.0)
    cos, sin = dit_table(ids)
    values = np.arange(128, dtype=np.float32)
    out = apply_dit(values.reshape(1, 128), cos, sin)
    assert out.shape == (1, 128)
    assert abs(float(out[0, 0]) - float(values[0] * cos[0, 0] - values[1] * sin[0, 0])) < 1e-5


def test_parse_defaults_and_rejects_a_bad_size() -> None:
    request = pipeline.parse({"prompt": "horse", "seed": 0})
    assert (request.width, request.height, request.steps, request.seed) == (1024, 1024, 4, 0)
    try:
        pipeline.parse({"prompt": "horse", "size": "1000x1000", "seed": 0})
    except Exception as exc:
        assert "16" in str(exc)
    else:
        raise AssertionError("size was accepted")


def test_png_roundtrip() -> None:
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(decode_png(encode_png(rgb)), rgb)


def test_tools_publish_generate_and_edit() -> None:
    tools = pipeline.mcp_tools()
    assert [tool.name for tool in tools] == ["generate_image", "edit_image"]
    for tool in tools:
        assert len(tool.description) <= 256
        assert tool.handler == "handle_image"
        assert tool.result == "image"
        for prop in tool.parameters["properties"].values():
            assert len(prop["description"]) <= 48
