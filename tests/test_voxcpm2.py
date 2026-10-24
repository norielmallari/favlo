"""Host equations for the VoxCPM2 block. The RMSNorm check dispatches Metal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from voxcpm2.audio import encode_wav
from voxcpm2.block import embedding_scale, residual_add, rope_scale, swiglu
from voxcpm2.checkpoint import summarize, tensor_header
from voxcpm2.checks import (
    attention_matches,
    conv_matches,
    linear_matches,
    rmsnorm_matches,
    rope_matches,
)
from voxcpm2.config import load_config
from voxcpm2.dit import diffusion_schedule, zero_init_steps
from voxcpm2.fsq import quantize_bound
from voxcpm2.text import style_line
from voxcpm2.tokenize import AUDIO_START, LineTokenizer


def test_embedding_scale_is_one_when_mup_is_off() -> None:
    assert embedding_scale(False, 12.0) == 1.0
    assert embedding_scale(True, 12.0) == 12.0


def test_residual_add_skips_the_depth_scale_when_mup_is_off() -> None:
    residual = np.array([1.0, 2.0], dtype=np.float32)
    update = np.array([0.5, -0.5], dtype=np.float32)
    plain = residual_add(residual, update, use_mup=False, scale_depth=1.4, layers=28)
    assert np.allclose(plain, residual + update)
    scaled = residual_add(residual, update, use_mup=True, scale_depth=1.4, layers=28)
    assert np.allclose(scaled, residual + update * (1.4 / np.sqrt(28)))


def test_swiglu_is_silu_of_the_gate_times_up() -> None:
    gate = np.array([0.0, 1.0], dtype=np.float32)
    up = np.array([2.0, 3.0], dtype=np.float32)
    silu = gate * (1.0 / (1.0 + np.exp(-gate)))
    assert np.allclose(swiglu(gate, up), silu * up)


def test_rope_scale_is_one_when_the_caps_match() -> None:
    assert rope_scale(32768, 32768) == 1.0


def test_fsq_rounds_tanh_to_the_checkpoint_scale() -> None:
    projected = np.array([0.2, -1.5, 3.0], dtype=np.float64)
    expected = np.round(np.tanh(projected) * 9.0) / 9.0
    assert np.array_equal(quantize_bound(projected, 9), expected)


def test_style_line_wraps_instructions() -> None:
    assert style_line("Hello.", "") == "Hello."
    assert style_line("Hello.", "slightly faster, cheerful tone") == (
        "(slightly faster, cheerful tone)Hello."
    )
    assert style_line("Hello.", "(already)") == "(already)Hello."


def test_config_reads_the_published_widths(tmp_path: Path) -> None:
    body = {
        "patch_size": 4,
        "feat_dim": 64,
        "scalar_quantization_latent_dim": 512,
        "scalar_quantization_scale": 9,
        "residual_lm_num_layers": 8,
        "residual_lm_no_rope": True,
        "max_length": 8192,
        "lm_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "kv_channels": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "vocab_size": 73448,
            "max_position_embeddings": 32768,
            "use_mup": False,
            "scale_emb": 12,
            "scale_depth": 1.4,
            "rope_scaling": {
                "long_factor": [1.0, 1.0],
                "short_factor": [1.0, 1.0],
                "original_max_position_embeddings": 32768,
            },
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    config = load_config(path)
    assert config.lm.use_mup is False
    assert config.fsq_dim == 512
    assert config.fsq_scale == 9
    assert config.residual_no_rope is True
    assert config.cfg_rate == 2.0
    assert config.mean_mode is False
    assert config.decoder_rates == (8, 6, 5, 2, 2, 2)
    assert config.out_sample_rate == 48000


def test_tensor_header_counts_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, path)
    header = tensor_header(path)
    assert "weight" in header
    assert summarize(header) == "F32 1\ntensors 1"


def test_rmsnorm_row_matches_the_host_formula() -> None:
    rmsnorm_matches()


def test_half_linear_matches_the_host_formula() -> None:
    linear_matches()


def test_rope_and_attention_match_the_host_formulas() -> None:
    rope_matches()
    attention_matches()


def test_causal_conv_matches_the_host_formula() -> None:
    conv_matches()


def test_sway_schedule_keeps_the_endpoints() -> None:
    span = diffusion_schedule(10)
    assert len(span) == 11
    assert zero_init_steps(span) == 1
    assert abs(float(span[0]) - 1.0) < 1e-6
    assert abs(float(span[-1])) < 1e-6


def test_line_tokens_start_with_bos_and_end_with_audio() -> None:
    path = Path("models/openbmb/VoxCPM2/tokenizer.json")
    if not path.is_file():
        pytest.skip("checkout missing")
    tokens = LineTokenizer(path).encode("Hello.")
    assert int(tokens[0]) == 1
    assert int(tokens[-1]) == AUDIO_START
    assert list(tokens[1:-1]) == [21045, 72]


def test_open_accepts_an_empty_resident_percent(tmp_path: Path) -> None:
    from voxcpm2.runtime import open_runtime

    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        open_runtime(tmp_path, resident_percent=None)


def test_wav_is_mono_pcm(tmp_path: Path) -> None:
    import wave

    raw = encode_wav(np.zeros(16, dtype=np.float32), 48000)
    path = tmp_path / "line.wav"
    path.write_bytes(raw)
    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 16
