"""One dense vector from the packed program, when that directory is present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bge_m3.runtime import Runtime
from bge_m3.text import HIDDEN


def _record() -> Path | None:
    from favlo.cli import repo_root

    path = repo_root() / "models" / "bge-m3-w8a8"
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    return path


def test_packed_program_returns_a_unit_dense_vector() -> None:
    path = _record()
    if path is None:
        pytest.skip("models/bge-m3-w8a8 is not packed")
    runtime = Runtime(path)
    try:
        result = runtime.embed("A stone bridge under a clear sky.")
    finally:
        runtime.close()
    vector = np.asarray(result.vector, dtype=np.float32)
    assert vector.shape == (HIDDEN,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3
    assert result.tokens >= 2


def test_an_input_past_the_packed_window_is_an_error() -> None:
    path = _record()
    if path is None:
        pytest.skip("models/bge-m3-w8a8 is not packed")
    runtime = Runtime(path)
    try:
        text = "word " * 600
        count = len(runtime.tokenizer.encode(text).ids)
        assert count > runtime.buckets[-1]
        with pytest.raises(ValueError, match=str(runtime.buckets[-1])):
            runtime.embed(text)
    finally:
        runtime.close()
