"""Host quantize helpers. No GPU."""

from __future__ import annotations

import os

import numpy as np

from favlo.quantize import quantize_fp8, quantize_int8, read_at, worker_count, write_at
from favlo.qwen35_pack import choose_target


def test_positioned_io_round_trip(tmp_path) -> None:
    path = tmp_path / "plane.bin"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, 9)
        write_at(fd, b"ab\ncd", 4)
        write_at(fd, b"WXYZ", 0)
        assert read_at(fd, 9, 0) == b"WXYZab\ncd"
    finally:
        os.close(fd)


def test_fp8_and_int8_run_on_the_host() -> None:
    matrix = np.ones((2, 128), dtype=np.float32)
    codes, scales = quantize_fp8(matrix)
    assert len(codes) == 256
    assert len(scales) == 8
    packed, scale = quantize_int8(matrix)
    assert packed.shape == (2, 128)
    assert scale.shape == (2,)


def test_plane_flag_does_not_need_a_device() -> None:
    assert choose_target("fp8").plane == "fp8"
    assert choose_target("int4").vision is False
    assert choose_target("int4").context == 65_536
    assert choose_target("int8").plane == "int8"


def test_worker_count_is_every_logical_cpu() -> None:
    machine = os.cpu_count() or 0
    visible = os.process_cpu_count() or 0
    assert worker_count() == max(1, machine, visible)
