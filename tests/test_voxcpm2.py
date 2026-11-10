"""Host equations for the VoxCPM2 block. The RMSNorm check dispatches Metal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from favlo.media_pipeline import MediaPaused, MediaRequestError
from voxcpm2.metal41.audio import encode_wav
from voxcpm2.metal41.block import embedding_scale, residual_add, rope_scale, swiglu
from voxcpm2.metal41.checkpoint import summarize, tensor_header
from voxcpm2.metal41.checks import (
    activation_matches,
    attention_matches,
    conv_matches,
    linear_matches,
    rmsnorm_matches,
    rope_matches,
    stack_matches,
)
from voxcpm2.metal41.config import load_config
from voxcpm2.metal41.dit import diffusion_schedule, zero_init_steps
from voxcpm2.metal41.fsq import quantize_bound
from voxcpm2.metal41.progress import PATCH_CAP, VoiceProgress, patch_limit, sample_total
from voxcpm2.metal41.runtime import Pipeline, Runtime, VoiceRequest
from voxcpm2.metal41.text import style_line
from voxcpm2.metal41.tokenize import AUDIO_START, LineTokenizer


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


def test_speak_cli_requires_a_persona() -> None:
    from voxcpm2.metal41.commands import main

    with pytest.raises(SystemExit) as caught:
        main(["speak", "--text", "Hello.", "--out", "clip.wav"])
    assert caught.value.code == 2


def test_parse_requires_a_persona() -> None:
    pipeline = Pipeline()
    with pytest.raises(MediaRequestError, match="persona is required"):
        pipeline.parse({"input": "Hello."})
    request = pipeline.parse({"input": " Hello. ", "persona": " clear adult voice "})
    assert request == VoiceRequest("Hello.", "clear adult voice", "mp3")
    assert style_line(request.text, request.persona) == "(clear adult voice) Hello."


def test_style_line_joins_the_persona() -> None:
    assert style_line("Hello.", "clear adult voice") == "(clear adult voice) Hello."
    assert style_line("Hello.", "(already)") == "(already) Hello."
    assert style_line("Hello.", "Clear adult voice.") == "(Clear adult voice) Hello."
    assert style_line("Hello.", "Clear adult voice, even pace.") == (
        "(Clear adult voice, even pace) Hello."
    )
    assert style_line("你好", "（温柔）") == "(温柔)你好"
    assert style_line("Hello.", "（温柔）") == "(温柔) Hello."
    with pytest.raises(ValueError, match="persona is required"):
        style_line("Hello.", "  ")
    with pytest.raises(ValueError, match="persona is required"):
        style_line("Hello.", "...")


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


def test_float_activations_match_the_host_formulas() -> None:
    activation_matches()


def test_one_stack_step_is_one_command_buffer() -> None:
    stack_matches()


def test_sway_schedule_keeps_the_endpoints() -> None:
    span = diffusion_schedule(10)
    assert len(span) == 11
    assert zero_init_steps(span) == 1
    assert abs(float(span[0]) - 1.0) < 1e-6
    assert abs(float(span[-1])) < 1e-6


def test_line_tokens_follow_the_published_tokenize_path() -> None:
    path = Path("models/openbmb/VoxCPM2/tokenizer.json")
    if not path.is_file():
        pytest.skip("checkout missing")
    tokens = LineTokenizer(path).encode("Hello.")
    assert list(tokens[:-1]) == [21045, 72]
    assert int(tokens[-1]) == AUDIO_START
    spoken = "The small brown fox spoke one short line."
    line = style_line(spoken, "Clear adult voice, even pace.")
    styled = LineTokenizer(path).encode(line)
    opening = LineTokenizer(path).encode("(")
    bare = LineTokenizer(path).encode(spoken)
    assert int(styled[0]) == int(opening[0])
    assert list(styled[: -len(bare)]) + list(bare) == list(styled)
    assert int(styled[-1]) == AUDIO_START


def test_open_accepts_an_empty_resident_percent(tmp_path: Path) -> None:
    from voxcpm2.metal41.runtime import open_runtime

    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        open_runtime(tmp_path, resident_percent=None)


def test_patch_limit_follows_the_published_clamp() -> None:
    assert PATCH_CAP == 4096
    assert patch_limit(3, PATCH_CAP, 8192) == 3 * 6 + 10
    assert patch_limit(3, 3, 8192) == 3
    assert patch_limit(800, PATCH_CAP, 8192) == PATCH_CAP
    assert patch_limit(40, PATCH_CAP, 8192) == 250
    assert patch_limit(10, PATCH_CAP, 20) == 9
    with pytest.raises(ValueError, match="does not fit"):
        patch_limit(10, PATCH_CAP, 11)


def test_sample_total_counts_steps_and_the_decoder() -> None:
    assert sample_total(4, 10) == 41
    with pytest.raises(ValueError):
        sample_total(0, 10)


def test_sample_row_names_the_patch_and_counts_stages() -> None:
    store: dict[str, object] = {}
    job = {"status": "in_progress", "progress": 0, "stage": None, "eta": None}
    progress = VoiceProgress(_bars(store), job)
    progress.open(4)
    progress.stage("patch 1/2")
    progress.advance()
    progress.advance()
    handle = store["handle"]
    assert store["phase"] == "sample"
    assert store["unit"] == "stages"
    assert store["total"] == 4
    assert handle.detail == "patch 1/2"
    assert handle.done == 2
    assert job["stage"] == "patch 1/2"
    assert job["progress"] == 50
    assert isinstance(job["eta"], int)
    progress.note("patch 1/2 0.10s")
    progress.complete()
    progress.close(progress.summary())
    assert store["lines"] == ["patch 1/2 0.10s"]
    assert store["summary"] == "sample 2/2 stages"
    assert store["closed"] == 2
    assert job["progress"] == 99
    assert job["eta"] == 0


def test_speak_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def fake(*args, **kwargs):
        del args
        assert kwargs["max_len"] == PATCH_CAP
        progress = kwargs["progress"]
        progress.open(4)
        progress.stage("patch 1/2")
        progress.advance()
        progress.advance()
        progress.note("patch 1/2 0.10s")
        progress.patches = 1
        progress.decode_s = 0.25
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr("voxcpm2.metal41.runtime.generate", fake)
    runtime = _runtime(_bars(store))
    job = {"status": "in_progress", "progress": 0, "stage": None, "eta": None}
    samples = runtime.speak("Hello.", persona="Clear adult voice.", job=job)
    assert samples.shape == (8,)
    assert store["total"] == 4
    assert store["summary"] == "sample 2/2 stages"
    assert store["closed"] == 2
    assert job["progress"] == 99
    lines = store["lines"]
    assert isinstance(lines, list)
    assert lines[0] == "patch 1/2 0.10s"
    assert lines[1].startswith("speak ")
    assert "patches 1" in lines[1]
    assert "decode 0.25s" in lines[1]


def test_a_paused_speak_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def fake(*args, **kwargs):
        del args
        kwargs["progress"].open(4)
        raise MediaPaused("preempt", None)

    monkeypatch.setattr("voxcpm2.metal41.runtime.generate", fake)
    runtime = _runtime(_bars(store))
    with pytest.raises(MediaPaused):
        runtime.speak("Hello.", persona="Clear adult voice.")
    assert store["summary"] == "sample paused"


def test_a_cancelled_speak_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def fake(*args, **kwargs):
        del args
        kwargs["progress"].open(4)
        raise MediaPaused("cancel", None)

    monkeypatch.setattr("voxcpm2.metal41.runtime.generate", fake)
    runtime = _runtime(_bars(store))
    with pytest.raises(MediaPaused):
        runtime.speak("Hello.", persona="Clear adult voice.")
    assert store["summary"] == "sample cancelled"


def test_a_failed_speak_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def fake(*args, **kwargs):
        del args
        kwargs["progress"].open(4)
        raise RuntimeError("latent is not finite")

    monkeypatch.setattr("voxcpm2.metal41.runtime.generate", fake)
    runtime = _runtime(_bars(store))
    with pytest.raises(RuntimeError, match="latent is not finite"):
        runtime.speak("Hello.", persona="Clear adult voice.")
    assert store["summary"] == "sample failed"


def test_speak_without_a_console_prints_the_lines(monkeypatch, capsys) -> None:
    def fake(*args, **kwargs):
        del args
        progress = kwargs["progress"]
        progress.open(2)
        progress.note("patch 1/1 0.10s")
        progress.patches = 1
        progress.decode_s = 0.20
        progress.advance()
        return np.zeros(4, dtype=np.float32)

    monkeypatch.setattr("voxcpm2.metal41.runtime.generate", fake)
    runtime = _runtime(None)
    runtime.speak("Hello.", persona="Clear adult voice.")
    text = capsys.readouterr().out
    assert text.startswith("patch 1/1 0.10s\n")
    assert "patches 1" in text
    assert "decode 0.20s" in text


def test_generate_hands_the_job_to_speak() -> None:
    seen: dict[str, object] = {}

    class _Runtime:
        def speak(self, text: str, **kwargs: object) -> str:
            seen["text"] = text
            seen["job"] = kwargs["job"]
            return "wave"

    job = {"status": "in_progress"}
    result = Pipeline().generate(
        _Runtime(),
        VoiceRequest("Hello.", "Clear adult voice.", "wav"),
        None,
        job,
        lambda: None,
        None,
    )
    assert result == "wave"
    assert seen["text"] == "Hello."
    assert seen["job"] is job


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


def _runtime(console: object) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime.console = console
    runtime.config = None
    runtime._open = lambda: (None, None, None, None)
    return runtime


class _Row:
    def __init__(self) -> None:
        self.done = 0
        self.detail = ""
        self.total = 0


def _bars(store: dict[str, object]):
    class _Console:
        shows_bars = True

        def begin(self, phase: str, **kwargs: object) -> _Row:
            store["phase"] = phase
            store["unit"] = kwargs["unit"]
            store["total"] = kwargs["total"]
            opened = _Row()
            opened.total = int(kwargs["total"])
            store["handle"] = opened
            return opened

        def end(self, progress: _Row, summary: str) -> None:
            store["summary"] = summary
            store["closed"] = progress.done

        def log(self, line: str) -> None:
            lines = store.setdefault("lines", [])
            assert isinstance(lines, list)
            lines.append(line)

    return _Console()
