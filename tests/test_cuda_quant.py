"""Integer packing for the CUDA weight planes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "pipelines" / "qwen35_4b" / "cuda"


def _load(name: str, relative: str):
    sys.path.insert(0, str(CUDA))
    path = CUDA / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        message = f"cannot load {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_int8_row_uses_qmax_127() -> None:
    formats = _load("qwen35_cuda_formats", "formats.py")
    weight = np.array([[127.0, -63.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    codes, scale = formats.quantize_int8(weight)
    assert scale[0] == np.float32(1.0)
    assert scale[1] == np.float32(0.0)
    assert int(codes[0, 0]) == 127
    assert int(codes[0, 1]) == -64
    assert int(codes[1, 0]) == 0


def test_int4_packs_two_nibbles() -> None:
    formats = _load("qwen35_cuda_formats_int4", "formats.py")
    weight = np.array([[7.0, -7.0, 3.5, 0.0]], dtype=np.float32)
    packed, scale = formats.quantize_int4(weight)
    assert scale[0] == np.float32(1.0)
    assert packed.shape == (1, 2)
    assert int(packed[0, 0]) == 0x97
    assert int(packed[0, 1]) == 0x04


def test_four_and_six_gigabyte_openings() -> None:
    layout = _load("qwen35_cuda_layout_pack", "layout.py")
    small = layout.pack_for_budget(4 << 30)
    large = layout.pack_for_budget(6 << 30)
    assert small.plane == "int4" and small.vision is False and small.context == layout.CONTEXT_4GB
    assert large.plane == "int8" and large.vision is True and large.context >= 32_768
    whole = layout.pack_for_budget(8 << 30)
    assert whole.plane == "int8" and whole.context > large.context
