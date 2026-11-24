"""One dense vector from the packed program, when that directory is present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bge_m3.metal41.runtime import Runtime
from bge_m3.metal41.text import HIDDEN


_PIPELINE = "bge_m3"


def _record() -> Path | None:
    from favlo.registry import choose_complete

    try:
        path, _selection = choose_complete(
            model=None, record=None, pipeline=_PIPELINE
        )
    except LookupError:
        return None
    if not (path / "manifest.json").is_file():
        return None
    return path


def test_packed_program_returns_a_unit_dense_vector() -> None:
    path = _record()
    if path is None:
        pytest.skip("bge_m3 registry default is not packed")
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
        pytest.skip("bge_m3 registry default is not packed")
    runtime = Runtime(path)
    try:
        text = "word " * 8000
        count = len(runtime.tokenizer.encode(text).ids)
        assert count > runtime.limit
        with pytest.raises(ValueError, match=str(runtime.limit)):
            runtime.embed(text)
    finally:
        runtime.close()
