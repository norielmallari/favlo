"""Host math for BGE-M3. The packed program is not required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bge_m3.pack import pack_int8, restore_int8
from bge_m3.passages import passages
from bge_m3.quantize import _COSINE_MIN, _finished
from bge_m3.text import (
    BUCKETS,
    HIDDEN,
    PAD_ID,
    bucket_for,
    key_mask,
    l2_normalize,
    lexical_max,
    pad_ids,
    padded_length,
    position_ids,
)


def test_position_ids_start_at_two_and_pads_stay() -> None:
    assert position_ids([0, 5, 9, PAD_ID, PAD_ID]) == [2, 3, 4, PAD_ID, PAD_ID]


def test_the_sequence_pads_to_the_matmul_tile() -> None:
    assert padded_length(1) == 64
    assert padded_length(64) == 64
    assert padded_length(65) == 128
    assert padded_length(4096) == 4096
    try:
        padded_length(4097)
    except ValueError as exc:
        assert "4096" in str(exc)
    else:
        raise AssertionError("expected a length error")


def test_the_next_bucket_covers_the_token_count() -> None:
    assert bucket_for(1) == 128
    assert bucket_for(128) == 128
    assert bucket_for(129) == 256
    assert bucket_for(4096) == 4096
    try:
        bucket_for(4097)
    except ValueError as exc:
        assert "4096" in str(exc)
    else:
        raise AssertionError("expected a length error")
    packed = (128, 256, 512)
    assert bucket_for(512, packed) == 512
    try:
        bucket_for(513, packed)
    except ValueError as exc:
        assert "512" in str(exc)
    else:
        raise AssertionError("expected the packed window")


def test_pad_mask_is_the_finite_fill() -> None:
    mask = key_mask([0, 5, PAD_ID])
    assert mask.shape == (1, 1, 3)
    assert mask.dtype == np.float16
    assert float(mask[0, 0, 0]) == 0.0
    assert float(mask[0, 0, 2]) == -40000.0


def test_pad_ids_fill_the_bucket_with_the_pad_id() -> None:
    assert pad_ids([0, 2], 4) == [0, 2, PAD_ID, PAD_ID]


def test_dense_vector_is_unit_length() -> None:
    vector = l2_normalize(np.asarray([3.0, 4.0], dtype=np.float32))
    assert vector.shape == (2,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
    assert abs(float(vector[0]) - 0.6) < 1e-6


def test_lexical_weight_keeps_the_max_relu() -> None:
    rows = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [5.0, 5.0]],
        dtype=np.float32,
    )
    ids = [7, 7, 8, PAD_ID]
    weights = lexical_max(rows, ids, np.asarray([1.0, 2.0], dtype=np.float32), 0.0)
    assert weights[7] == 2.0
    assert weights[8] == 0.0
    assert PAD_ID not in weights


def test_int8_roundtrip_stays_within_half_a_step() -> None:
    weight = np.linspace(-1.0, 1.0, 128 * 256, dtype=np.float32).reshape(128, 256)
    codes, scale = pack_int8(weight)
    restored = restore_int8(codes, scale)
    step = np.maximum(scale, np.float32(1e-12))
    ratio = np.max(np.abs(restored - weight) / step[:, None])
    assert float(ratio) <= 1.0 + 1e-5


def test_calibration_is_128_short_passages() -> None:
    rows = passages()
    assert len(rows) == 128
    assert len(set(rows)) == 128
    assert all(len(row) < 80 for row in rows)
    assert HIDDEN == 1024
    assert BUCKETS[-1] == 4096


def test_only_a_complete_directory_counts_as_finished(tmp_path: Path) -> None:
    dest = _record(tmp_path)
    assert _finished(dest)
    (dest / "cosine.json").write_text(
        json.dumps({"passages": 128, "min": 0.5, "mean": 0.5}),
        encoding="utf-8",
    )
    assert not _finished(dest)
    (dest / "cosine.json").write_text(
        json.dumps({"passages": 128, "min": 0.995, "mean": 0.995}),
        encoding="utf-8",
    )
    body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    body["quant"] = "w8a8"
    (dest / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    assert not _finished(dest)


def test_the_cosine_gate_is_the_quantized_line() -> None:
    assert _COSINE_MIN == 0.96


def _record(root: Path) -> Path:
    dest = root / "bge-m3-int8"
    dest.mkdir(parents=True)
    np.savez(dest / "weights.npz", word=np.zeros(4, dtype=np.float16))
    (dest / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dest / "cosine.json").write_text(
        json.dumps({"passages": 128, "min": 0.995, "mean": 0.995}),
        encoding="utf-8",
    )
    (dest / "manifest.json").write_text(
        json.dumps({"complete": True, "quant": "int8", "hidden": HIDDEN, "context": 4096}),
        encoding="utf-8",
    )
    return dest
