"""GPTQ and FOEM rounding on the fp4 grid. No device."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qwen38_27b.metal41.formats import quantize_fp4
from qwen38_27b.metal41.gptq import foem_fp4, pad_fp4, relative_error


def test_diagonal_gram_matches_round_to_nearest() -> None:
    matrix = np.random.default_rng(0).normal(size=(8, 64)).astype(np.float32) * np.float32(0.05)
    packed, scales = foem_fp4(matrix, np.eye(64), beta=0.0)
    data, plane = quantize_fp4(matrix)
    assert packed.tobytes() == data
    assert scales.tobytes() == plane


def test_foem_moves_a_correlated_matrix() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(16, 1)).astype(np.float32)
    spread = np.linspace(0.4, 1.4, 64, dtype=np.float32)
    matrix = base * spread
    tokens = rng.normal(size=(128, 64)).astype(np.float32)
    hessian = tokens.T @ tokens
    nearest, near_scale = foem_fp4(matrix, hessian, beta=0.0)
    packed, scales = foem_fp4(matrix, hessian, beta=0.1)
    assert not np.array_equal(packed, nearest)
    before = relative_error(matrix, nearest, near_scale, hessian)
    after = relative_error(matrix, packed, scales, hessian)
    assert np.isfinite(after)
    assert after <= before


def test_dead_column_stays_zero() -> None:
    matrix = np.ones((4, 64), dtype=np.float32)
    hessian = np.eye(64)
    hessian[3, 3] = 0.0
    packed, _scales = foem_fp4(matrix, hessian, beta=0.1)
    assert np.all((packed[:, 1] >> 4) == 0)


def test_weight_read_spans_pieces(tmp_path: Path) -> None:
    from qwen38_27b.metal41.calibration import _Shards, _disk_order, _read_exact

    payload = bytes(index % 251 for index in range(100))
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    with path.open("rb") as handle:
        handle.seek(3)
        got = _read_exact(handle, 40, piece=8)
    assert bytes(got) == payload[3:43]
    shards = _Shards(tmp_path)
    try:
        block = shards.read(path, 10, 50)
        assert bytes(block) == payload[10:60]
        follow = shards.read(path, 60, 20)
        assert bytes(follow) == payload[60:80]
    finally:
        shards.close()
    headers = {
        "b.safetensors": ({"m": {"data_offsets": [10, 20]}}, 100),
        "a.safetensors": (
            {"n": {"data_offsets": [0, 4]}, "p": {"data_offsets": [8, 12]}},
            50,
        ),
    }
    weight_map = {"m": "b.safetensors", "n": "a.safetensors", "p": "a.safetensors"}
    ordered = _disk_order(["m", "p", "n"], weight_map, headers, tmp_path)
    assert ordered == ["n", "p", "m"]


def test_pad_keeps_the_logical_prefix() -> None:
    matrix = np.random.default_rng(2).normal(size=(4, 32)).astype(np.float32) * np.float32(0.05)
    packed, scales = foem_fp4(matrix, np.eye(32), beta=0.0)
    data, plane = pad_fp4(packed, scales, 64, 256)
    assert len(data) == 64 * 128
    assert len(plane) == 64 * 8
    view = np.frombuffer(data, dtype=np.uint8).reshape(64, 128)
    assert np.array_equal(view[:4, :16], packed)
    assert int(view[4:].sum()) == 0
    assert int(view[:, 16:].sum()) == 0


def test_sequences_follow_the_shared_corpus(tmp_path: Path, monkeypatch) -> None:
    import favlo.calibration_corpus as corpus
    import qwen38_27b.metal41.calibration as calibration
    import qwen38_27b.portable.chat as chat

    def encode(_root, messages, _tools, _thinking):
        return list(range(int(messages[0]["content"])))

    def decode(_root, ids):
        return "<tool_call>" if ids and ids[-1] >= 500 else ""

    monkeypatch.setattr(chat, "encode_chat", encode)
    monkeypatch.setattr(chat, "decode_text", decode)
    monkeypatch.setattr(calibration, "_root", lambda: tmp_path)
    monkeypatch.setattr(calibration, "_QUOTAS", (("swe", 2), ("code", 1)))
    swe = {"messages": [{"role": "user", "content": "10000"}], "tools": []}
    bare = {"messages": [{"role": "user", "content": "100"}]}
    assert calibration._window(tmp_path, "swe", bare, 2048) == []
    swe_window = calibration._window(tmp_path, "swe", swe, 2048)
    assert swe_window[0] == 373
    assert len(swe_window) == 8192
    code_window = calibration._window(
        tmp_path, "code", {"messages": [{"role": "user", "content": "5000"}]}, 2048
    )
    assert code_window[0] == 0
    assert len(code_window) == 4096

    def load(kind: str):
        if kind == "swe":
            return [swe, bare]
        if kind == "code":
            return [{"messages": [{"role": "user", "content": "3000"}]}]
        return []

    monkeypatch.setattr(corpus, "load", load)
    sequences = calibration._sequences(tmp_path)
    assert [len(row) for row in sequences] == [2048, 2048, 2048]
    assert sequences[0][0] == 373
    assert sequences[2][0] == 0
    assert calibration._sequences(tmp_path) == sequences
