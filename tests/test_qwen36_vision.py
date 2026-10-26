"""Vision packing and the image-pad expansion. They do not dispatch Metal."""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from AppKit import NSBitmapImageRep, NSColor, NSGIFFileType, NSJPEGFileType

from qwen36_35b_a3b.chat import encode_text
from qwen36_35b_a3b.vision import (
    MIN_PIXELS,
    _image_bytes,
    _pixel_bounds,
    decode_image,
    decode_png,
    expand_image_tokens,
    interpolate_positions,
    pack_image,
    rope_positions,
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


def test_smart_size_uses_the_processor_edges() -> None:
    assert smart_size(256, 256) == (256, 256)
    assert smart_size(512, 512) == (512, 512)
    assert smart_size(32, 32) == (256, 256)
    assert smart_size(8192, 8192) == (4096, 4096)
    assert smart_size(1024, 1024, max_pixels=MIN_PIXELS) == (256, 256)
    with pytest.raises(RuntimeError, match="aspect"):
        smart_size(1, 401)


def test_png_roundtrip_and_red_pack() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    assert np.array_equal(decode_png(_png(image)), image)
    packed, grid_h, grid_w = pack_image(image, min_pixels=32 * 32, max_pixels=32 * 32)
    assert (grid_h, grid_w) == (2, 2)
    assert packed.shape == (4, 1536)
    assert np.allclose(packed[:, :512], 1.0)
    assert np.allclose(packed[:, 512:], -1.0)


def _bitmap(kind: int) -> bytes:
    alloc = NSBitmapImageRep.alloc()
    name = (
        "initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_"
        "samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_"
    )
    rep = getattr(alloc, name)(
        None,
        8,
        8,
        8,
        3,
        False,
        False,
        "NSCalibratedRGBColorSpace",
        24,
        24,
    )
    color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 0, 0, 1)
    for y in range(8):
        for x in range(8):
            rep.setColor_atX_y_(color, x, y)
    data = rep.representationUsingType_properties_(kind, None)
    assert data is not None
    return bytes(data)


def test_chat_image_accepts_a_data_url_and_a_public_budget() -> None:
    jpeg = _bitmap(NSJPEGFileType)
    gif = _bitmap(NSGIFFileType)
    encoded = base64.b64encode(jpeg).decode()
    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}
    decoded = decode_image(_image_bytes(part))
    assert decoded.shape == (8, 8, 3)
    assert int(decoded[:, :, 0].min()) > 200
    assert int(decoded[:, :, 1].max()) < 30
    assert decode_image(gif).shape[2] == 3
    blank = base64.b64encode(_png(np.zeros((8, 8, 3), dtype=np.uint8))).decode()
    raw = {"type": "image_url", "image_url": {"url": blank}}
    assert decode_image(_image_bytes(raw)).shape == (8, 8, 3)
    low = {"type": "image_url", "image_url": {"url": "x", "detail": "low"}}
    assert _pixel_bounds(low) == (MIN_PIXELS, MIN_PIXELS)
    high = {"type": "image_url", "image_url": {"url": "x", "detail": "high"}}
    assert _pixel_bounds(high)[1] == 16777216
    sized = {"type": "image", "min_pixels": 65536, "max_pixels": 262144}
    assert _pixel_bounds(sized) == (65536, 262144)
    with pytest.raises(RuntimeError, match="public address"):
        _image_bytes({"image_url": {"url": "http://127.0.0.1/still.png"}})
    with pytest.raises(RuntimeError, match="http or https"):
        _image_bytes({"image_url": {"url": "file:///still.png"}})
    with pytest.raises(RuntimeError, match="outside"):
        _pixel_bounds({"image_url": {"url": "x"}, "max_pixels": 16777217})


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


def test_an_image_advances_mrope_by_its_longer_side() -> None:
    text, delta = rope_positions(4, [])
    assert delta == 0
    assert np.array_equal(text, np.tile(np.arange(4, dtype=np.uint32), (3, 1)))
    positions, delta = rope_positions(10, [(2, 2, 3)])
    assert positions[:, :2].tolist() == [[0, 1], [0, 1], [0, 1]]
    assert positions[0, 2:8].tolist() == [2, 2, 2, 2, 2, 2]
    assert positions[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]
    assert positions[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]
    assert positions[:, 8:].tolist() == [[5, 6], [5, 6], [5, 6]]
    assert delta == -3
    assert (10 - 1) + delta == int(positions[0, -1])
    assert 10 + delta == int(positions[0, -1]) + 1


def test_image_pad_expands_once() -> None:
    expanded, spans = expand_image_tokens([1, 9, 2, 9, 3], [2, 1], 9)
    assert expanded == [1, 9, 9, 2, 9, 3]
    assert spans == [1, 4]


def test_image_pad_is_one_token() -> None:
    if not (RECORD / "tokenizer.json").is_file():
        pytest.skip("quantized record is not installed")
    assert encode_text(RECORD, "<|image_pad|>") == [248056]
