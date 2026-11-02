"""Plane, promotion, catalog, and prefix checks for Gemma 4 26B-A4B."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from favlo.prefix import PrefixHeader, RegionRef
from gemma4_26b_a4b.checkpoint import load_tensor, tensor_names
from gemma4_26b_a4b.geometry import GLOBAL_LAYERS, expected, local_layers
from gemma4_26b_a4b.planes import (
    align_k,
    expert_bytes,
    expert_planes,
    pack_matrix,
    relative_rms,
    restore_matrix,
    scale_stride,
)
from gemma4_26b_a4b.prefix import (
    CacheAdapter,
    global_nbytes,
    local_nbytes,
    pack_global,
    pack_local,
    pack_token_major,
)
from gemma4_26b_a4b.promote import promotion_count, select_promoted


def test_fp8_roundtrip_and_zero_block() -> None:
    weight = np.zeros((4, 128), dtype=np.float32)
    weight[0, 0] = 1.0
    weight[1, 17] = -0.5
    restored = restore_matrix(pack_matrix(weight, "fp8"), 4, 128, "fp8")
    assert restored.shape == weight.shape
    assert restored[2, 0] == 0.0
    assert relative_rms(weight, restored) < 0.05
    zeros = np.zeros((2, 64), dtype=np.float32)
    assert np.all(restore_matrix(pack_matrix(zeros, "fp8"), 2, 64, "fp8") == 0.0)


def test_fp4_midpoint_keeps_the_even_magnitude() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 3.5
    restored = restore_matrix(pack_matrix(weight, "fp4"), 1, 32, "fp4")
    assert restored[0, 0] == 4.0


def test_fp4_scale_search_beats_the_peak_scale() -> None:
    weight = np.full((1, 32), 0.35, dtype=np.float32)
    weight[0, 0] = 6.2
    packed = pack_matrix(weight, "fp4")
    restored = restore_matrix(packed, 1, 32, "fp4")
    peak = float(np.max(np.abs(weight)))
    coarse = 2.0 ** np.ceil(np.log2(max(peak / 6.0, np.finfo(np.float64).tiny)))
    unit = weight / coarse
    edges = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    index = np.clip(np.digitize(np.abs(unit), edges, right=True), 0, 7)
    magnitude = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[index]
    peak_restore = np.sign(weight) * magnitude * coarse
    assert relative_rms(weight, restored) <= relative_rms(weight, peak_restore)


def test_fp4_low_nibble_is_element_zero() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 0.5
    weight[0, 1] = 1.0
    blob = pack_matrix(weight, "fp4")
    first = blob[0]
    assert first & 0x0F != 0
    restored = restore_matrix(blob, 1, 32, "fp4")
    assert abs(float(restored[0, 0]) - 0.5) < 0.26
    assert abs(float(restored[0, 1]) - 1.0) < 0.26


def test_q4_0_pairs_the_block_ends() -> None:
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 8.0
    weight[0, 16] = -4.0
    blob = pack_matrix(weight, "q4_0")
    assert len(blob) == 18
    pair = blob[2]
    assert pair & 0x0F == 0
    assert pair >> 4 == 12
    restored = restore_matrix(blob, 1, 32, "q4_0")
    assert abs(float(restored[0, 0]) - 8.0) < 1e-5
    assert abs(float(restored[0, 16]) + 4.0) < 1e-5
    zeros = np.zeros((2, 64), dtype=np.float32)
    assert np.all(restore_matrix(pack_matrix(zeros, "q4_0"), 2, 64, "q4_0") == 0.0)


def test_q4_0_recovers_a_qat_step() -> None:
    step = np.float32(2**-6)
    weight = np.zeros((1, 32), dtype=np.float32)
    weight[0, 0] = 3.0 * step
    weight[0, 1] = -2.0 * step
    weight[0, 16] = step
    restored = restore_matrix(pack_matrix(weight, "q4_0"), 1, 32, "q4_0")
    assert np.max(np.abs(restored - weight)) < 1e-6


def test_k_pad_and_tight_scale_row() -> None:
    assert align_k(704, "fp4") == 768
    assert align_k(704, "fp8") == 768
    assert align_k(2112, "fp4") == 2304
    assert align_k(2112, "fp8") == 2176
    assert align_k(1152, "fp8") == 1152
    assert align_k(1152, "fp4") == 1280
    assert align_k(704, "q4_0") == 704
    assert align_k(4304, "q4_0") == 4320
    assert scale_stride(2816) == 2816 // 32
    assert scale_stride(1024) == 1024 // 32


def test_expert_images_start_on_a_128_byte_boundary() -> None:
    for plane in ("fp4", "fp8"):
        parts = expert_planes(plane)
        assert all(part % 128 == 0 for part in parts)
    assert expert_bytes(4) == expert_planes("fp4")[-1]
    assert expert_bytes(8) == expert_planes("fp8")[-1]
    assert expert_bytes(8) > expert_bytes(4)


def test_promotion_count_and_tie_break() -> None:
    assert promotion_count(3840, 20) == 768
    assert promotion_count(3840, 0) == 0
    scores = [(1.0, 2, 0), (1.0, 1, 4), (9.0, 0, 1), (1.0, 1, 3)]
    chosen = select_promoted(scores, 75)
    assert chosen[0] == (0, 1)
    assert chosen[1] == (1, 3)
    assert chosen[2] == (1, 4)


def test_catalog_matches_the_checkpoint_map() -> None:
    catalog = expected()
    assert len(catalog) == 1013
    ups = [name for name, (role, _shape) in catalog.items() if role == "expert-up"]
    assert len(ups) == 30
    local = []
    missing_v = []
    for layer in range(30):
        name = f"model.language_model.layers.{layer}.self_attn.v_proj.weight"
        if layer in GLOBAL_LAYERS:
            assert name not in catalog
            missing_v.append(layer)
        else:
            assert name in catalog
            local.append(layer)
    assert len(local) == 25
    assert missing_v == list(GLOBAL_LAYERS)
    assert len(local_layers()) == 25


def test_bf16_reader_slices_one_expert(tmp_path: Path) -> None:
    values = np.array([[[1.0, -2.0], [0.5, 4.0]], [[3.0, 0.0], [-1.0, 8.0]]], dtype=np.float32)
    raw = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    header = {
        "experts": {
            "dtype": "BF16",
            "shape": [2, 2, 2],
            "data_offsets": [0, len(raw)],
        }
    }
    payload = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(payload)) + payload + raw)
    names = tensor_names(tmp_path)
    loaded = load_tensor(tmp_path, names["experts"], "experts", 1)
    assert loaded.shape == (2, 2)
    assert abs(float(loaded[0, 0]) - 3.0) < 0.05


def test_token_major_pack_matches_the_host_planes() -> None:
    count = 4
    rng = np.random.default_rng(0)
    local_k = rng.standard_normal((25, 8, count, 256)).astype(np.float16)
    local_v = rng.standard_normal((25, 8, count, 256)).astype(np.float16)
    global_k = rng.standard_normal((5, 2, count, 512)).astype(np.float16)
    global_v = rng.standard_normal((5, 2, count, 512)).astype(np.float16)
    assert _packed_layers(local_k, transposed=False) + _packed_layers(
        local_v, transposed=True
    ) == pack_local(
        local_k.astype(np.float32),
        np.transpose(local_v.astype(np.float32), (0, 1, 3, 2)),
        count,
    )
    assert _packed_layers(global_k, transposed=False) + _packed_layers(
        global_v, transposed=False
    ) == pack_global(
        global_k.astype(np.float32),
        global_v.astype(np.float32),
        count,
    )


def _packed_layers(plane: np.ndarray, *, transposed: bool) -> bytes:
    layers, heads, _count, dim = plane.shape
    parts: list[bytes] = []
    for layer in range(layers):
        token_major = np.ascontiguousarray(np.transpose(plane[layer], (1, 0, 2)))
        parts.append(
            pack_token_major(token_major.tobytes(), heads, dim, plane.shape[2], transposed=transposed)
        )
    return b"".join(parts)


def test_prefix_pack_length_and_exact_file_refusal() -> None:
    count = 4
    keys = np.zeros((25, 8, count, 256), dtype=np.float32)
    values = np.zeros((25, 8, 256, count), dtype=np.float32)
    global_keys = np.zeros((5, 2, count, 512), dtype=np.float32)
    global_values = np.zeros((5, 2, count, 512), dtype=np.float32)
    local = pack_local(keys, values, count)
    packed_global = pack_global(global_keys, global_values, count)
    assert len(local) == local_nbytes(count)
    assert len(packed_global) == global_nbytes(count)
    header = PrefixHeader(
        path=Path("prefix.bin"),
        model="gemma4-26b-a4b",
        digest=b"\0" * 32,
        tag="half",
        tokens=(1, 2, 3, 4),
        regions=(
            RegionRef("local", 0, len(local)),
            RegionRef("global", len(local), len(packed_global)),
        ),
        file_size=len(local) + len(packed_global),
    )
    adapter = CacheAdapter.__new__(CacheAdapter)
    assert adapter.accept(header, [1, 2, 3, 4]) is False
    assert adapter.accept(header, [1, 2, 3, 4, 5]) is True
    assert adapter.accept(header, [9, 2, 3, 4, 5]) is False
