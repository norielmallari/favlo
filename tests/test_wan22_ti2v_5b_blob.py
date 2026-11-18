"""Kernel-order packs for TI2V. A temp directory, no GPU."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wan22_ti2v_5b.metal41.blob import (
    PIECE,
    Writer,
    conv_matrix,
    expect_conv,
    expect_f16,
    expect_i8,
    f16_matrix,
    fill,
    load_planes,
    resolve_named,
    write_vae_arrays,
)
from wan22_ti2v_5b.metal41.dit import BLOCK_MATRICES, STEMS, _stem
from wan22_ti2v_5b.metal41.quantize import _PAIRS, write_dit_record
from wan22_ti2v_5b.metal41.text import WEIGHTS
from wan22_ti2v_5b.metal41.weights import load_matrix, matrix_from_planes


def test_block_names_match_the_quantize_pairs() -> None:
    assert tuple(name for name, _stem_name in _PAIRS) == BLOCK_MATRICES
    assert STEMS[0] == "patch"
    assert WEIGHTS[0] == "attn.q.weight"


def test_float16_gemm_pads_n_to_32() -> None:
    weight = np.arange(15, dtype=np.float32).reshape(3, 5)
    stored = f16_matrix(weight)
    assert stored.shape == (32, 5)
    assert stored.dtype == np.float16
    assert np.array_equal(stored[:3], weight.astype(np.float16))
    assert np.all(stored[3:] == 0)


def test_wide_conv_is_the_padded_flatten_not_the_5d_order() -> None:
    weight = np.arange(40 * 4 * 1 * 5 * 4, dtype=np.float32).reshape(40, 4, 1, 5, 4)
    stored, meta = conv_matrix(weight)
    assert meta["layout"] == "conv64"
    assert stored.shape == (64, 128)
    assert stored.dtype == np.float16
    flat = np.ascontiguousarray(weight, dtype=np.float16).reshape(40, 80)
    assert np.array_equal(stored[:40, :80], flat)
    assert np.all(stored[40:] == 0)
    assert np.all(stored[:, 80:] == 0)
    assert stored.tobytes() != np.ascontiguousarray(weight).tobytes()


def test_four_d_conv_gains_a_time_tap_before_the_pad() -> None:
    weight = np.arange(40 * 4 * 5 * 4, dtype=np.float32).reshape(40, 4, 5, 4)
    stored, meta = conv_matrix(weight)
    assert meta["rank"] == 4
    assert meta["source_shape"] == [40, 4, 1, 5, 4]
    assert stored.shape == (64, 128)


def test_small_conv_stays_float32_in_conv_order() -> None:
    weight = np.arange(8 * 2 * 1 * 3 * 3, dtype=np.float32).reshape(8, 2, 1, 3, 3)
    stored, meta = conv_matrix(weight)
    assert meta["layout"] == "conv32"
    assert stored.dtype == np.float32
    assert np.array_equal(stored, weight)


def test_conv_boundary_is_32_outputs_and_64_taps() -> None:
    wide = np.zeros((32, 4, 1, 4, 4), dtype=np.float32)
    small = np.zeros((31, 4, 1, 4, 4), dtype=np.float32)
    assert conv_matrix(wide)[1]["layout"] == "conv64"
    assert conv_matrix(small)[1]["layout"] == "conv32"


def test_plane_nbytes_excludes_the_alignment_gap(tmp_path) -> None:
    writer = Writer()
    writer.add("codes", np.arange(3, dtype=np.int8))
    writer.add("scale", np.array([1.5], dtype=np.float32))
    blob = writer.finish(tmp_path / "weights.bin", tmp_path / "planes.json")
    planes = load_planes(tmp_path / "planes.json")
    assert planes["codes"]["offset"] == 0
    assert planes["codes"]["nbytes"] == 3
    assert planes["scale"]["offset"] == 128
    assert planes["scale"]["nbytes"] == 4
    raw = blob.read_bytes()
    assert len(raw) == 132
    assert raw[3:128] == bytes(125)
    assert raw[:3] == bytes([0, 1, 2])


def test_wrong_nbytes_and_wrong_pad_are_refused(tmp_path) -> None:
    (tmp_path / "planes.json").write_text(
        json.dumps(
            [{"name": "codes", "dtype": "int8", "offset": 0, "nbytes": 4, "shape": [3]}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="array size"):
        load_planes(tmp_path / "planes.json")
    bad = {
        "name": "patch",
        "dtype": "float16",
        "offset": 0,
        "nbytes": 30,
        "shape": [3, 5],
        "logical_n": 3,
        "logical_k": 5,
    }
    with pytest.raises(RuntimeError, match="kernel wants"):
        expect_f16(bad)
    bad_i8 = {
        "name": "self_q",
        "dtype": "int8",
        "offset": 0,
        "nbytes": 15,
        "shape": [3, 5],
        "logical_n": 3,
        "logical_k": 5,
    }
    with pytest.raises(RuntimeError, match="kernel wants"):
        expect_i8(bad_i8)


def test_fill_reads_across_an_8_mib_boundary(tmp_path) -> None:
    size = PIECE + 100
    payload = np.arange(size, dtype=np.uint8).tobytes()
    path = tmp_path / "dit.bin"
    path.write_bytes(payload)
    found = bytearray(size)
    fill(memoryview(found), path)
    assert bytes(found) == payload
    assert PIECE == 8 * 1024 * 1024


def test_missing_pair_keeps_npy_and_a_half_pair_fails(tmp_path) -> None:
    weight = np.arange(6, dtype=np.float16).reshape(2, 3)
    np.save(tmp_path / "patch.npy", weight)
    assert resolve_named(tmp_path, "dit.bin", "dit.json") is None
    matrix = _stem(tmp_path, "patch", None, None)
    assert matrix.span is None
    assert matrix.packed.shape == (2, 3)
    assert load_matrix(tmp_path / "patch.npy").kind == "f16"
    (tmp_path / "dit.bin").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="dit.json"):
        resolve_named(tmp_path, "dit.bin", "dit.json")
    (tmp_path / "weights.bin").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="planes.json"):
        resolve_named(tmp_path, "weights.bin", "planes.json")


def test_dit_writer_copies_int8_and_pads_float16(tmp_path) -> None:
    for name in STEMS:
        np.save(tmp_path / f"{name}.npy", np.arange(15, dtype=np.float16).reshape(3, 5))
    codes = np.arange(128 * 256, dtype=np.int8).reshape(128, 256)
    scales = np.linspace(0.1, 1.0, 128).astype(np.float32)
    bias = np.arange(96, dtype=np.float16)
    block = tmp_path / "blocks" / "00"
    block.mkdir(parents=True)
    for name in BLOCK_MATRICES:
        np.savez(
            block / f"{name}.npz",
            packed=codes,
            scales=scales,
            bias=bias,
            logical_n=np.int32(96),
            logical_k=np.int32(200),
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"layers": 1}), encoding="utf-8")
    write_dit_record(tmp_path)
    planes = load_planes(tmp_path / "dit.json")
    raw = (tmp_path / "dit.bin").read_bytes()
    patch = planes["patch"]
    assert tuple(patch["shape"]) == (32, 5)
    assert patch["dtype"] == "float16"
    assert raw[patch["offset"] : patch["offset"] + patch["nbytes"]] == f16_matrix(
        np.arange(15, dtype=np.float16).reshape(3, 5)
    ).tobytes()
    (tmp_path / "patch.npy").unlink()
    bound = _stem(tmp_path, "patch", object(), planes)
    assert bound.packed is None
    assert bound.stored_shape == (32, 5)
    assert bound.logical_n == 3
    plane = planes["blocks.00.self_q"]
    copied = raw[plane["offset"] : plane["offset"] + plane["nbytes"]]
    assert copied == codes.tobytes()
    assert plane["nbytes"] == codes.nbytes
    bias_plane = planes["blocks.00.self_q.bias"]
    assert tuple(bias_plane["shape"]) == (128,)
    stored_bias = np.frombuffer(
        raw[bias_plane["offset"] : bias_plane["offset"] + bias_plane["nbytes"]],
        dtype=np.float16,
    )
    assert np.array_equal(stored_bias[:96], bias)
    assert np.all(stored_bias[96:] == 0)
    matrix = matrix_from_planes(object(), planes, "blocks.00.self_q", kind="int8")
    umt5 = {
        "blocks.0.attn.o.weight.packed": {
            "name": "blocks.0.attn.o.weight.packed",
            "dtype": "int8",
            "shape": [128, 256],
            "offset": 0,
            "nbytes": 128 * 256,
        },
        "blocks.0.attn.o.weight.scales": {
            "name": "blocks.0.attn.o.weight.scales",
            "dtype": "float32",
            "shape": [128],
            "offset": 128 * 256,
            "nbytes": 512,
        },
        "blocks.0.attn.o.weight.bias": {
            "name": "blocks.0.attn.o.weight.bias",
            "dtype": "float16",
            "shape": [128],
            "offset": 128 * 256 + 512,
            "nbytes": 256,
        },
    }
    output = matrix_from_planes(object(), umt5, "blocks.0.attn.o.weight", kind="int8")
    assert output.packed_at == 0
    assert output.scales_at == umt5["blocks.0.attn.o.weight.scales"]["offset"]
    assert output.bias_at == umt5["blocks.0.attn.o.weight.bias"]["offset"]
    assert matrix.packed is None
    assert matrix.span is not None
    assert matrix.scales_at == planes["blocks.00.self_q.scales"]["offset"]
    assert matrix.bias_at == bias_plane["offset"]


def test_vae_writer_uses_the_upload_layouts(tmp_path) -> None:
    wide = np.arange(40 * 2 * 2 * 4 * 5, dtype=np.float32).reshape(40, 2, 2, 4, 5)
    small = np.arange(8 * 1 * 1 * 3 * 3, dtype=np.float32).reshape(8, 1, 1, 3, 3)
    flat = np.arange(40 * 4 * 5 * 4, dtype=np.float32).reshape(40, 4, 5, 4)
    linear = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    arrays = {
        "encoder.conv1.weight": wide,
        "encoder.conv1.bias": np.arange(40, dtype=np.float32),
        "encoder.head.0.gamma": np.arange(8, dtype=np.float32),
        "encoder.down.weight": small,
        "decoder.resample.weight": flat,
        "encoder.middle.1.to_qkv.weight": linear,
        "encoder.middle.1.to_qkv.bias": np.arange(9, dtype=np.float32),
    }
    dest = tmp_path / "vae.bin"
    write_vae_arrays(arrays, dest)
    planes = load_planes(dest.with_suffix(".json"))
    raw = dest.read_bytes()
    wide_plane = planes["encoder.conv1.weight"]
    expect_conv(wide_plane)
    stored, _meta = conv_matrix(wide)
    assert raw[wide_plane["offset"] : wide_plane["offset"] + wide_plane["nbytes"]] == stored.tobytes()
    assert wide_plane["layout"] == "conv64"
    assert planes["encoder.down.weight"]["layout"] == "conv32"
    assert planes["encoder.down.weight"]["dtype"] == "float32"
    assert planes["decoder.resample.weight"]["source_shape"] == [40, 4, 1, 5, 4]
    assert tuple(planes["encoder.middle.1.to_qkv.weight"]["shape"]) == (32, 5)
    assert resolve_named(tmp_path, "vae.bin", "vae.json") == dest
