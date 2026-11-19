"""Host checks for the qwen35_4b CUDA vision tower. No device."""

from __future__ import annotations

import struct
import zlib

import numpy as np

from qwen35_4b.cuda.vision import (
    _decode_bitmap,
    expand_image_tokens,
    merged_grid,
    pack_image,
    rope_positions,
    smart_size,
)


def test_smart_size_and_pack() -> None:
    assert smart_size(256, 256) == (256, 256)
    assert smart_size(100, 100)[0] % 32 == 0
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    packed, grid_h, grid_w = pack_image(image, min_pixels=32 * 32, max_pixels=64 * 64)
    assert (grid_h, grid_w) == (4, 4)
    assert packed.shape == (16, 1536)
    assert merged_grid(image, min_pixels=32 * 32, max_pixels=64 * 64) == (2, 2)


def test_image_pads_move_the_rope_cut() -> None:
    expanded, spans = expand_image_tokens([1, 248_056, 2], [4], 248_056)
    assert spans == [1]
    assert expanded == [1, 248_056, 248_056, 248_056, 248_056, 2]
    positions, delta = rope_positions(len(expanded), [(1, 2, 2)])
    assert positions.shape == (3, 6)
    assert int(positions[0, 1]) == int(positions[0, 0]) + 1
    assert delta == int(positions.max()) + 1 - len(expanded)


def _png(width: int, height: int, rgb: np.ndarray) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_png_decodes_to_rgb() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    decoded = _decode_bitmap(_png(8, 8, rgb))
    assert decoded.shape == (8, 8, 3)
    assert int(decoded[:, :, 0].min()) > 150
