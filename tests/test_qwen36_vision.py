"""Vision packing and the image-pad expansion. They do not dispatch Metal."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from qwen36_35b_a3b.chat import encode_text
from qwen36_35b_a3b.vision import (
    decode_png,
    expand_image_tokens,
    interpolate_positions,
    pack_image,
    smart_size,
    vision_rotary,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "models" / "Qwen3.6-35B-A3B-fp4"


def _png(image: np.ndarray) -> bytes:
    height, width, _channels = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        check = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)

    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, 9)),
            chunk(b"IEND", b""),
        )
    )


def test_smart_size_keeps_a_factor_aligned_square() -> None:
    assert smart_size(512, 512) == (512, 512)
    assert smart_size(1024, 1024) == (512, 512)


def test_png_roundtrip_and_red_pack() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    assert np.array_equal(decode_png(_png(image)), image)
    packed, grid_h, grid_w = pack_image(image)
    assert (grid_h, grid_w) == (2, 2)
    assert packed.shape == (4, 1536)
    assert np.allclose(packed[:, :512], 1.0)
    assert np.allclose(packed[:, 512:], -1.0)


def test_position_sample_covers_the_table() -> None:
    side = 48
    table = np.arange(side * side, dtype=np.float32)[:, None] * np.ones((1, 1152), dtype=np.float32)
    sampled = interpolate_positions(table, side, side)
    assert sampled.shape == (side * side, 1152)
    assert sorted(sampled[:, 0].tolist()) == list(range(side * side))
    cos, sin = vision_rotary(4, 4)
    assert cos.shape == (16, 72)
    assert np.allclose(cos[:, :36], cos[:, 36:])
    assert sin.shape == cos.shape


def test_image_pad_expands_once() -> None:
    expanded, spans = expand_image_tokens([1, 9, 2, 9, 3], [2, 1], 9)
    assert expanded == [1, 9, 9, 2, 9, 3]
    assert spans == [1, 4]


def test_image_pad_is_one_token() -> None:
    if not (RECORD / "tokenizer.json").is_file():
        pytest.skip("quantized record is not installed")
    assert encode_text(RECORD, "<|image_pad|>") == [248056]
