"""Host checks for the zonos2 package. These tests do not load a checkpoint."""

from __future__ import annotations

import io
import math
import pickle
import zipfile
from pathlib import Path

import pytest

import numpy as np

from zonos2.checkpoint import id_tables, list_tensors, read_tensor
from zonos2.delay import shear, unshear
from zonos2.formats import (
    dequant_fp4,
    dequant_fp8,
    promote_count,
    quantize_fp4,
    quantize_fp8,
    relative_error,
    scale_exponent,
)
from zonos2.audio import accurate_token, background_token, encode_wav, quality_token, text_ids
from zonos2.formats import choose_promoted
from zonos2.pack import matrix_plane, split_w13
from zonos2.layout import FP8_PERCENT, OUTPUT_ROWS, TEXT_ROWS, dense_layers, moe_layers, topk
from zonos2.forward import choose_experts
from zonos2.sampling import DEFAULT_VARIANT, VARIANTS
from zonos2.signals import voice_ok


def test_open_runtime_refuses_a_bf16_checkout(tmp_path: Path):
    from zonos2.runtime import open_runtime

    checkout = tmp_path / "ZONOS2-bf16"
    checkout.mkdir()
    (checkout / "model.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="packed record"):
        open_runtime(checkout, kv_cache="e4m3", console=None, resident_percent=100.0)
    with pytest.raises(ValueError, match="packed record"):
        open_runtime(tmp_path, kv_cache="bf16", console=None, resident_percent=100.0)


def test_bytes_are_the_text_rows_and_the_wav_is_mono():
    assert text_ids("A").tolist() == [2, 257, 3]
    payload = encode_wav(np.linspace(-1.5, 1.5, 441, dtype=np.float32))
    assert payload.startswith(b"RIFF")
    promoted = choose_promoted({3: 0.2, 4: 0.2, 5: 0.9, 6: 0.1}, 50)
    assert promoted == (3, 5)
    assert dense_layers() == (0, 1, 2, 27)
    assert moe_layers() == tuple(range(3, 27))
    assert topk(26) == 2
    assert topk(3) == 1
    assert topk(0) == 0
    assert promote_count(FP8_PERCENT, len(moe_layers())) == 4
    assert promote_count(FP8_PERCENT, len(dense_layers())) == 0
    assert matrix_plane("layers.0.attention.wq.weight") == "f16"
    assert matrix_plane("layers.0.attention.wkv.weight") == "f16"
    assert matrix_plane("layers.0.feed_forward.w_in.weight") == "f16"
    assert matrix_plane("layers.27.feed_forward.w_out.weight") == "f16"
    assert matrix_plane("multi_output.weight") == "f16"
    assert matrix_plane("layers.0.attention.gater.weight") == "fp4"
    assert TEXT_ROWS == 520
    assert OUTPUT_ROWS == 9234
    assert background_token(True) == 516
    assert background_token(False) == 517
    assert accurate_token() == 518
    assert quality_token(5, 3) == 511
    fused = np.arange(6144 * 2, dtype=np.float32).reshape(6144, 2)
    gate, up = split_w13(fused)
    assert np.array_equal(gate, fused[0::2])
    assert np.array_equal(up, fused[1::2])


def test_balancing_bias_ranks_after_the_softmax():
    picked, probs = choose_experts(np.array([0.3, 0.0]), np.array([0.0, 0.2]), 1)
    assert picked.tolist() == [1]
    assert probs[1] < probs[0]


def test_voice_signals_reject_silence_noise_and_a_tone():
    rate = 44100
    silence = np.zeros(rate, dtype=np.float32)
    noise = np.random.default_rng(0).normal(0, 0.2, rate).astype(np.float32)
    tone = np.sin(2 * math.pi * 200 * np.arange(rate) / rate).astype(np.float32) * 0.4
    assert voice_ok(silence)[0] is False
    assert voice_ok(noise)[0] is False
    assert voice_ok(tone)[0] is False
    time = np.arange(int(rate * 1.2)) / rate
    pitch = 110.0 + 25.0 * np.sin(2 * math.pi * 3.0 * time)
    phase = 2 * math.pi * np.cumsum(pitch) / rate
    voice = np.zeros_like(time)
    for harmonic, scale in enumerate((1.0, 0.6, 0.35, 0.2, 0.1), start=1):
        voice += scale * np.sin(harmonic * phase)
    voice *= 0.15
    voice += np.random.default_rng(1).normal(0, 0.01, voice.shape)
    ok, report = voice_ok(voice.astype(np.float32))
    assert ok, report


def test_sampler_card():
    row = VARIANTS[DEFAULT_VARIANT]
    assert set(row) == {"temperature", "top_p", "top_k", "presence_penalty"}
    assert row["temperature"] == 1.15
    assert row["top_p"] == 0.0
    assert row["top_k"] == 106
    assert row["presence_penalty"] == 0.0


def test_delay_round_trip():
    aligned = np.arange(18, dtype=np.int64).reshape(2, 9)
    delayed = shear(aligned)
    assert delayed.shape == (10, 9)
    assert delayed[0, 1] == 1025
    assert delayed[1, 1] == aligned[0, 1]
    assert np.array_equal(unshear(delayed), aligned)


def test_fp4_round_up_tie_lands_on_the_lower_code():
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 5.0
    assert scale_exponent(5.0, 6.0) == 127
    data, scales = quantize_fp4(weight)
    restored = dequant_fp4(data, scales, 1, 32)
    assert restored[0, 0] == 4.0
    assert math.isclose(relative_error(weight), 0.2, rel_tol=1e-6)


def test_fp8_round_up_tie_lands_on_the_lower_code():
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 272.0
    data, scales = quantize_fp8(weight)
    restored = dequant_fp8(data, scales, 1, 32)
    assert restored[0, 0] == 256.0
    assert math.isclose(relative_error_fp8(weight), 16.0 / 272.0, rel_tol=1e-5)


def test_zero_block_stores_exponent_zero():
    weight = np.zeros((2, 64), dtype=np.float32)
    _data, scales = quantize_fp4(weight)
    assert int(scales.max()) == 0


def relative_error_fp8(weight: np.ndarray) -> float:
    data, scales = quantize_fp8(weight)
    restored = dequant_fp8(data, scales, weight.shape[0], weight.shape[1])
    diff = weight.astype(np.float64) - restored.astype(np.float64)
    source = weight.astype(np.float64)
    return math.sqrt(float(np.dot(diff.ravel(), diff.ravel()) / np.dot(source.ravel(), source.ravel())))


def _rebuild(*_args):
    raise AssertionError("the checkpoint reader supplies rebuild")


class _Storage:
    def __init__(self, key: str, numel: int, dtype_name: str) -> None:
        self.key = key
        self.numel = numel
        self.dtype_name = dtype_name


class _Tensor:
    def __init__(self, key: str, numel: int, dtype_name: str, shape: tuple[int, ...]) -> None:
        self.key = key
        self.numel = numel
        self.dtype_name = dtype_name
        self.shape = shape

    def __reduce__(self):
        storage = _Storage(self.key, self.numel, self.dtype_name)
        return (_rebuild, (storage, 0, self.shape, (1,) * len(self.shape), False, None, None))


class BFloat16Storage:
    pass


class LongStorage:
    pass


class _Pickler(pickle._Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _Storage):
            kind = {"BFloat16Storage": BFloat16Storage, "LongStorage": LongStorage}[obj.dtype_name]
            return ("storage", kind, obj.key, "cpu", obj.numel)
        return None

    def save_global(self, obj, name=None):
        if obj is _rebuild:
            self.write(b"ctorch._utils\n_rebuild_tensor_v2\n")
            return
        super().save_global(obj, name)


_Pickler.dispatch = dict(pickle._Pickler.dispatch)
_Pickler.dispatch[type(_rebuild)] = _Pickler.save_global


def test_checkpoint_lists_views_and_ignores_float_embeddings(tmp_path: Path):
    payload = io.BytesIO()
    _Pickler(payload, protocol=4).dump(
        {
            "text": _Tensor("0", 4, "BFloat16Storage", (4,)),
            "token_ids": _Tensor("1", 3, "LongStorage", (3,)),
        }
    )
    bits = np.array([0x3F80, 0x4000, 0x4040, 0x4080], dtype="<u2").tobytes()
    archive_path = tmp_path / "model.pth"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data.pkl", payload.getvalue())
        archive.writestr("data/0", bits)
        archive.writestr("data/1", np.array([4, 1024, 1025], dtype="<i8").tobytes())
    tensors = list_tensors(archive_path)
    by_name = {item.name: item for item in tensors}
    assert set(by_name) == {"text", "token_ids"}
    assert by_name["text"].dtype == "BFloat16Storage"
    assert [item.name for item in id_tables(tensors)] == ["token_ids"]
    values = read_tensor(archive_path, by_name["text"])
    assert values.shape == (4,)
    assert np.allclose(values, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
