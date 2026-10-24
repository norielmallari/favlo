"""Host checks for the int2 pack and the GGUF name map. No Metal dispatch."""

from __future__ import annotations

import numpy as np

from ternary_bonsai_2.convert import _language_name
from ternary_bonsai_2.formats import dequant_int2, quantize_int2


def test_int2_keeps_a_ternary_magnitude() -> None:
    weight = np.full((1, 512), np.float32(0.0173))
    weight[:, 1::2] *= -1
    weight[:, 2::3] = 0
    data, scales = quantize_int2(weight)
    restored = dequant_int2(data, scales, 1, 512)
    assert float(np.max(np.abs(restored - weight))) < 0.008


def test_int2_restores_a_power_of_two_peak() -> None:
    weight = np.zeros((4, 512), dtype=np.float32)
    weight[:, 0] = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    data, scales = quantize_int2(weight)
    restored = dequant_int2(data, scales, 4, 512)
    assert np.array_equal(restored[:, 0], weight[:, 0])
    assert np.all(restored[:, 1:] == 0.0)


def test_gguf_names_match_the_language_record() -> None:
    assert _language_name("token_embd.weight") == "model.language_model.embed_tokens.weight"
    assert _language_name("output.weight") == "lm_head.weight"
    assert _language_name("blk.3.attn_q.weight").endswith("self_attn.q_proj.weight")
    assert _language_name("blk.0.attn_qkv.weight").endswith("linear_attn.in_proj_qkv.weight")
    assert _language_name("blk.0.attn_gate.weight").endswith("linear_attn.in_proj_z.weight")
    assert _language_name("blk.0.ssm_alpha.weight").endswith("linear_attn.in_proj_a.weight")
    assert _language_name("blk.0.ssm_a").endswith("linear_attn.A_log")
