"""Host math for BGE-M3. The packed program is not required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bge_m3.metal41.pack import pack_int8, restore_int8
from bge_m3.metal41.passages import passages
from bge_m3.metal41.quantize import (
    _COSINE_MIN,
    _finished,
    _load_weight_index,
    _weight_names,
    _write_blob,
)
from bge_m3.metal41.text import (
    BUCKETS,
    HIDDEN,
    LAYERS,
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


def test_weight_names_follow_the_upload_order() -> None:
    names = _weight_names()
    assert names[:5] == ("word", "position", "type", "embed_w", "embed_b")
    layer: list[str] = []
    for proj in ("q", "k", "v", "o", "up", "down"):
        layer.extend((f"l0_{proj}_codes", f"l0_{proj}_scale", f"l0_{proj}_bias"))
    layer.extend(("l0_attn_w", "l0_attn_b", "l0_ffn_w", "l0_ffn_b"))
    assert names[5 : 5 + len(layer)] == tuple(layer)
    assert names[-5] == "l23_down_bias"
    assert names[-4:] == ("l23_attn_w", "l23_attn_b", "l23_ffn_w", "l23_ffn_b")
    assert len(names) == 5 + LAYERS * len(layer)


def test_the_weight_blob_stores_real_sizes(tmp_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, name in enumerate(_weight_names()):
        if name.endswith("_codes"):
            arrays[name] = np.arange(3 + (index % 5), dtype=np.int8)
        elif name.endswith("_scale"):
            arrays[name] = np.ones(index % 3 + 1, dtype=np.float32)
        else:
            arrays[name] = np.ones(index % 4 + 1, dtype=np.float16)
    dest = tmp_path / "bge-m3-int8"
    dest.mkdir()
    _write_blob(dest, arrays)
    blob = (dest / "weights.bin").read_bytes()
    body = json.loads((dest / "weights.json").read_text(encoding="utf-8"))
    cursor = 0
    for row, name in zip(body["arrays"], _weight_names(), strict=True):
        raw = np.ascontiguousarray(arrays[name]).tobytes()
        assert row["name"] == name
        assert row["offset"] == cursor
        assert row["nbytes"] == len(raw)
        assert blob[cursor : cursor + len(raw)] == raw
        cursor += len(raw)
    assert len(blob) == cursor
    assert any(int(row["nbytes"]) < 16 for row in body["arrays"])
    rows = _load_weight_index(dest / "weights.json")
    assert rows[-1]["offset"] + rows[-1]["nbytes"] == len(blob)
    short = next(row for row in rows if int(row["nbytes"]) < 16)
    follow = rows[rows.index(short) + 1]
    assert follow["offset"] == short["offset"] + short["nbytes"]


def test_the_weight_index_rejects_a_pad(tmp_path: Path) -> None:
    rows = []
    for name in _weight_names():
        rows.append(
            {"name": name, "dtype": "int8", "shape": [4], "offset": 16, "nbytes": 4}
        )
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"arrays": rows}), encoding="utf-8")
    try:
        _load_weight_index(path)
    except RuntimeError as exc:
        assert "real size" in str(exc)
    else:
        raise AssertionError("expected a padded index to fail")


def test_the_weight_blob_refuses_a_missing_array(tmp_path: Path) -> None:
    dest = tmp_path / "partial"
    dest.mkdir()
    try:
        _write_blob(dest, {})
    except RuntimeError as exc:
        assert "word" in str(exc)
    else:
        raise AssertionError("expected a missing array")
    assert not (dest / "weights.bin").exists()


def test_a_blob_tree_counts_as_finished(tmp_path: Path) -> None:
    dest = tmp_path / "bge-m3-int8"
    dest.mkdir()
    (dest / "weights.bin").write_bytes(b"\x00")
    (dest / "weights.json").write_text("{}", encoding="utf-8")
    (dest / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dest / "cosine.json").write_text(
        json.dumps({"passages": 128, "min": 0.995, "mean": 0.995}),
        encoding="utf-8",
    )
    (dest / "manifest.json").write_text(
        json.dumps({"complete": True, "quant": "int8", "hidden": HIDDEN, "context": 4096}),
        encoding="utf-8",
    )
    assert not (dest / "weights.npz").exists()
    assert _finished(dest)
    (dest / "weights.bin").unlink()
    assert not _finished(dest)


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
