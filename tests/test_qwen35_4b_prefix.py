"""Prefix pack/unpack for qwen35_4b. Host only; no device."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "pipelines" / "metal41" / "qwen35_4b"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qwen35_4b.metal41.layout import BLOCK, HEAD_DIM, KV_HEADS  # noqa: E402
from qwen35_4b.metal41.prefix import pack_plane, unpack_plane  # noqa: E402


def test_half_plane_round_trip() -> None:
    max_pos = 16
    count = 5
    row = KV_HEADS * HEAD_DIM * 2
    live = bytearray(max_pos * row)
    for index in range(len(live)):
        live[index] = index % 251
    packed = pack_plane(bytes(live), max_pos, count, "half")
    assert len(packed) == count * row
    restored = bytearray(max_pos * row)
    unpack_plane(restored, max_pos, count, "half", packed)
    assert restored[: count * row] == live[: count * row]
    assert restored[count * row :] == b"\0" * (len(restored) - count * row)


def test_fp8_e4m3_plane_round_trip() -> None:
    max_pos = 16
    count = 7
    data_row = KV_HEADS * HEAD_DIM
    scale_row = KV_HEADS * (HEAD_DIM // BLOCK)
    live = bytearray(max_pos * data_row + max_pos * scale_row)
    for index in range(len(live)):
        live[index] = (index * 3) % 251
    packed = pack_plane(bytes(live), max_pos, count, "fp8_e4m3")
    assert len(packed) == count * data_row + count * scale_row
    restored = bytearray(len(live))
    unpack_plane(restored, max_pos, count, "fp8_e4m3", packed)
    assert restored[: count * data_row] == live[: count * data_row]
    scale_off = max_pos * data_row
    assert (
        restored[scale_off : scale_off + count * scale_row]
        == live[scale_off : scale_off + count * scale_row]
    )
