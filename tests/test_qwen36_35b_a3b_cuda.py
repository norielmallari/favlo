"""CUDA layout for qwen36_35b_a3b. No device."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "qwen36_35b_a3b" / "cuda"


def _load(filename: str, name: str) -> object:
    path = _ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_hybrid_and_expert_budget() -> None:
    layout = _load("layout.py", "qwen36_cuda_layout_test")
    assert layout.LAYERS == 40
    assert layout.FULL == 10
    assert layout.GDN_LAYERS == 30
    assert layout.EXPERTS == 256
    assert layout.LANGUAGE_EXPERTS == 10_240
    assert layout.promote_count(10) == 1_024
    assert layout.promote_count(20) == 2_048
    assert layout.promote_count(0) == 0
    assert layout.promote_count(100) == 10_240
    assert sum(1 for layer in range(layout.LAYERS) if layout.is_full(layer)) == 10


def test_weight_and_cache_sizes() -> None:
    layout = _load("layout.py", "qwen36_cuda_layout_test")
    assert layout.INT8_IMAGE == 3_158_016
    assert layout.INT4_IMAGE == 1_585_152
    assert layout.atlas_bytes() == 1_024 * layout.INT8_IMAGE + 9_216 * layout.INT4_IMAGE
    assert layout.atlas_bytes() == 17_842_569_216
    assert layout.pinned_expert_bytes() == 1_024 * layout.INT8_IMAGE
    assert layout.stage_bytes() == 8 * layout.INT4_IMAGE
    assert layout.cache_slots(layout.INT8_IMAGE * 3) == 3
    assert layout.cache_slots(layout.INT8_IMAGE - 1) == 0
    assert layout.KV_TOKEN_BYTES == 10 * 2 * 2 * (256 + 8)
    assert layout.GDN_STATE_BYTES == 30 * 32 * 128 * 128 * 4
    assert layout.CONV_BYTES == 30 * 8192 * 3 * 2
    scratch = dict(layout.scratch_items())
    assert scratch["state"] == layout.GDN_STATE_BYTES
    assert scratch["history"] == layout.CONV_BYTES
    assert layout.opening_bytes(2048) > layout.resident_bytes()
    assert layout.cache_tokens(layout.opening_bytes(2048), resident=layout.resident_bytes()) == 2048
    assert layout.record_name("Qwen3.6-35B-A3B-bf16", 20) == "Qwen3.6-35B-A3B-int8-e20"
    assert layout.record_name("Qwen3.6-35B-A3B-bf16", 0) == "Qwen3.6-35B-A3B-int4"


def test_stream_spans_follow_the_image_width() -> None:
    layout = _load("layout.py", "qwen36_cuda_layout_test")
    widths = [8, 4, 8]
    offsets = [0, layout.INT8_IMAGE, layout.INT8_IMAGE + layout.INT4_IMAGE]
    spans = layout.stream_spans(widths, offsets, [2, 0, 1])
    assert spans == [
        (offsets[2], layout.INT8_IMAGE, 0),
        (offsets[0], layout.INT8_IMAGE, 1),
        (offsets[1], layout.INT4_IMAGE, 2),
    ]


def test_int4_roundtrip_stays_inside_the_grid() -> None:
    formats = _load("formats.py", "qwen36_cuda_formats_test")
    matrix = np.linspace(-1.5, 1.5, 64, dtype=np.float32).reshape(2, 32)
    assert formats.relative_error(matrix) < 0.2
    blob = formats.pack_matrix(matrix, "int4")
    assert len(blob) == 2 * 16 + 2 * 4


def test_sampling_card() -> None:
    sampling = _load("sampling.py", "qwen36_cuda_sampling_test")
    assert sampling.DEFAULT_VARIANT == "thinking"
    assert sampling.VARIANTS["thinking"]["temperature"] == 1.0
    assert sampling.VARIANTS["instruct"]["reasoning_effort"] == "none"
    assert sampling.resolve_effort("low") == "high"
    assert sampling.resolve_effort("max") == "high"
    assert sampling.resolve_effort("none") == "none"
    assert sampling.CONTEXT_LENGTH == 131_072
