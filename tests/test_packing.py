import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_pipeline():
    path = ROOT / "qwen38-flash-next.py"
    spec = importlib.util.spec_from_file_location("qwen38_flash_next_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline()


def test_int4_roundtrip_both_nibble_orders() -> None:
    codes = pipeline.INT4_ORDER_PATTERN
    for low_first in (True, False):
        raw = pipeline.pack_int4(codes, low_nibble_first=low_first)
        assert pipeline.unpack_int4(raw, low_nibble_first=low_first) == codes
        assert len(raw) == 16


def test_int4_orders_are_distinct() -> None:
    codes = pipeline.INT4_ORDER_PATTERN
    low = pipeline.pack_int4(codes, low_nibble_first=True)
    high = pipeline.pack_int4(codes, low_nibble_first=False)
    assert low != high
    assert pipeline.classify_subbyte(low, codes, bits=4) == "low_nibble_is_element_0"
    assert pipeline.classify_subbyte(high, codes, bits=4) == "high_nibble_is_element_0"


def test_int2_roundtrip_both_bit_orders() -> None:
    codes = pipeline.INT2_ORDER_PATTERN
    for low_first in (True, False):
        raw = pipeline.pack_int2(codes, low_bits_first=low_first)
        assert pipeline.unpack_int2(raw, low_bits_first=low_first) == codes
        assert len(raw) == 8


def test_e8m0_powers_of_two() -> None:
    assert pipeline.decode_e8m0(127) == 1.0
    assert pipeline.decode_e8m0(130) == 8.0
    assert pipeline.decode_e8m0(0xFF) != pipeline.decode_e8m0(0xFF)


def test_e4m3_known_values() -> None:
    assert pipeline.decode_e4m3(0x00) == 0.0
    assert pipeline.decode_e4m3(0x38) == 1.0
    assert pipeline.decode_e4m3(0x7E) == 448
    assert pipeline.decode_e4m3(0x7F) != pipeline.decode_e4m3(0x7F)


def test_e5m2_known_values() -> None:
    assert pipeline.decode_e5m2(0x00) == 0.0
    assert pipeline.decode_e5m2(0x3C) == 1.0
    assert pipeline.decode_e5m2(0x7C) == pytest.approx(float("inf"))


def test_e8m0_scale_rounds_up_and_skips_nan() -> None:
    assert pipeline.e8m0_scale_byte(0.0, 7) == 0
    assert pipeline.e8m0_scale_byte(7.0, 7) == 127
    assert pipeline.decode_e8m0(pipeline.e8m0_scale_byte(7.1, 7)) == 2.0
    assert pipeline.e8m0_scale_byte(1.0e40, 7) == 254
    assert pipeline.e8m0_scale_byte(1.0e40, 7) != 0xFF


def test_quantize_block_fits_the_signed_endpoints() -> None:
    values = [0.0] * 31 + [14.0]
    codes, scale_byte = pipeline.quantize_block(values, 4)
    assert scale_byte == 128
    assert max(codes) <= 7
    assert min(codes) >= -8
    assert codes[-1] == 7
    restored = [code * pipeline.decode_e8m0(scale_byte) for code in codes]
    assert restored[-1] == 14.0

    wide = [float(index - 16) for index in range(32)]
    int8_codes, int8_scale = pipeline.quantize_block(wide, 8)
    assert pipeline.decode_e8m0(int8_scale) == 0.25
    assert int8_codes[0] == -64
    assert int8_codes[-1] == 60

    narrow = [-1.5, 0.4, 0.0, 1.0] * 8
    int2_codes, _int2_scale = pipeline.quantize_block(narrow, 2)
    assert min(int2_codes) >= -2
    assert max(int2_codes) <= 1


def test_quantize_block_ties_to_even() -> None:
    values = [7.0, 0.5, -1.5] + [0.0] * 29
    codes, scale_byte = pipeline.quantize_block(values, 4)
    assert scale_byte == 127
    assert codes[1] == 0
    assert codes[2] == -2


def test_sign_extend_int4_endpoints() -> None:
    assert pipeline.sign_extend(0x8, 4) == -8
    assert pipeline.sign_extend(0x7, 4) == 7
    assert pipeline.sign_extend(0x0, 4) == 0
