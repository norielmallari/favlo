"""Host reads for the qwen35_4b CUDA weight file. No device."""

from __future__ import annotations

import ctypes
import os

import pytest

from qwen35_4b.cuda.disk import alloc_pages, free_pages, read_span

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the unbuffered reader is Windows")


def test_read_span_matches_the_file(tmp_path) -> None:
    payload = bytes((index * 17) % 251 for index in range((8 << 20) * 5 + 100))
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    window = alloc_pages(len(payload))
    try:
        read_span(path, 0, window, len(payload))
        assert ctypes.string_at(window, len(payload)) == payload
        read_span(path, 4096, window, 8192)
        assert ctypes.string_at(window, 8192) == payload[4096 : 4096 + 8192]
    finally:
        free_pages(window)
