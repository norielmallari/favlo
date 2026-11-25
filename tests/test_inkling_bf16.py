"""Host checks for the BF16 expert mix. No GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inkling_2_mini.metal41.bf16 import (
    POOL,
    _flat_view,
    _grouped_windows,
    _import_sidecars,
    fp4_relative,
    projection_image,
    promote_count,
    promoted_mask,
    source_variants,
)
from inkling_2_mini.metal41.layout import EXPERTS, HIDDEN, INTERMEDIATE
from inkling_2_mini.metal41.weights import safetensor_files


def test_promote_count_is_half_the_pool() -> None:
    assert promote_count(50, POOL) == 1536
    assert promote_count(0, POOL) == 0
    assert promote_count(100, POOL) == POOL


def test_ties_keep_the_lower_index() -> None:
    mask = promoted_mask(np.array([1.0, 1.0, 0.0]), 1)
    assert list(mask) == [True, False, False]


def test_zero_and_exact_fp4_have_no_relative_error() -> None:
    zeros = np.zeros((2, 32), dtype=np.float32)
    assert fp4_relative(zeros).tolist() == [0.0]
    exact = np.full((1, 64), 6.0, dtype=np.float32)
    assert fp4_relative(exact)[0] == 0.0


def test_width_plane_counts_every_expert() -> None:
    widths = np.full((24, 128), 4, dtype=np.uint8)
    assert len(_flat_view(widths)) == 24 * 128


def test_a_short_gap_stays_one_read() -> None:
    piece = 8 << 20
    near = [
        ("w13", 2, 2, 0, 0, 100),
        ("w2", 2, 2, 0, 100 + piece - 1, 50),
    ]
    assert len(_grouped_windows(near)) == 1
    far = [
        ("w13", 2, 2, 0, 0, 100),
        ("w2", 2, 2, 0, 100 + piece, 50),
    ]
    assert len(_grouped_windows(far)) == 2
    other = [
        ("w13", 2, 2, 0, 0, 100),
        ("w2", 2, 2, 1, 0, 50),
    ]
    assert len(_grouped_windows(other)) == 2


def test_source_card_matches_the_published_sampler() -> None:
    card = source_variants()
    assert card["thinking"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "presence_penalty": 0.0,
    }
    assert card["instruct"]["reasoning_effort"] == "minimal"
    assert card["instruct"]["top_k"] == 64


def test_sidecars_leave_the_weight_in_the_source(tmp_path: Path) -> None:
    source = tmp_path / "src"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (dest / "model-00001-of-00001.safetensors").write_bytes(b"partial")
    _import_sidecars(source, dest)
    assert (dest / "config.json").read_text() == "{}"
    assert not (dest / "model-00001-of-00001.safetensors").exists()
    assert (source / "model-00001-of-00001.safetensors").read_bytes() == b"weight"


def test_an_index_names_each_shard(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"y": "model-00002-of-00002.safetensors", '
        '"x": "model-00001-of-00002.safetensors"}}'
    )
    assert safetensor_files(tmp_path) == [first, second]
    assert safetensor_files(first) == [first]


def test_a_folder_without_an_index_uses_its_safetensors(tmp_path: Path) -> None:
    weight = tmp_path / "weights.safetensors"
    other = tmp_path / "extra.safetensors"
    weight.write_bytes(b"a")
    other.write_bytes(b"b")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("skip")
    assert safetensor_files(tmp_path) == [other, weight]


def test_image_bytes_are_aligned() -> None:
    assert projection_image(8, INTERMEDIATE * 2, HIDDEN) == (5_308_416, 165_888)
    assert projection_image(4, INTERMEDIATE * 2, HIDDEN) == (2_654_208, 165_888)
    assert projection_image(8, HIDDEN, INTERMEDIATE) == (2_654_208, 82_944)
    assert projection_image(4, HIDDEN, INTERMEDIATE) == (1_474_560, 92_160)
    assert EXPERTS == 128
