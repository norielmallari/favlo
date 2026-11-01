"""Record sizes, fp4 packing, chat markup, and prefix slices. No GPU."""

from __future__ import annotations

import numpy as np

from spark_x26_1_7b.cache import take_channel_major, take_token_major
from spark_x26_1_7b.chat import parse_tool_calls, split_thinking
from spark_x26_1_7b.formats import dequant_fp4, quantize_fp4
from spark_x26_1_7b.layout import cache_bytes, full_token_bytes, record_bytes
from spark_x26_1_7b.prefix import scatter_channel_major, scatter_token_major


def test_record_and_cache_bytes() -> None:
    assert record_bytes() == 908_038_144
    assert full_token_bytes() == 7392
    assert cache_bytes(2048) == 15_138_816
    assert cache_bytes(131072) == 968_884_224
    assert cache_bytes(1_048_576) == 7_751_073_792


def test_fp4_roundtrip_uses_the_magnitude_table() -> None:
    table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    weight = np.resize(table, (64, 256)).copy()
    weight[::2] *= -1.0
    packed, scales = quantize_fp4(weight)
    restored = dequant_fp4(packed, scales, 64, 256)
    assert np.allclose(restored, weight)


def test_think_and_tool_markup() -> None:
    reasoning, visible = split_thinking("<think>trace</think>answer", True)
    assert reasoning == "trace"
    assert visible == "answer"
    reasoning, visible = split_thinking("trace<tool_call>lookup</tool_call>", True)
    assert reasoning == "trace"
    assert visible.startswith("<tool_call>")
    content, calls = parse_tool_calls(
        'note<tool_call>lookup<arg_key>q</arg_key><arg_value>carrots</arg_value></tool_call>'
    )
    assert content == "note"
    assert calls[0]["id"] == "call_0"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "lookup"
    assert calls[0]["function"]["arguments"] == '{"q":"carrots"}'


def test_prefix_slice_roundtrip() -> None:
    heads, channels, length, capacity = 2, 4, 3, 8
    source = bytearray(heads * capacity * channels)
    for head in range(heads):
        for token in range(length):
            start = (head * capacity + token) * channels
            source[start : start + channels] = bytes([head * 10 + token + offset for offset in range(channels)])
    tight = take_token_major(memoryview(source), heads, channels, length, capacity)
    dest = bytearray(len(source))
    scatter_token_major(dest, tight, heads, channels, length, capacity)
    assert dest[: length * channels] == source[: length * channels]
    channel_source = bytearray(heads * channels * capacity)
    for head in range(heads):
        for channel in range(channels):
            start = (head * channels + channel) * capacity
            channel_source[start : start + length] = bytes([token + 1 for token in range(length)])
    tight_values = take_channel_major(memoryview(channel_source), heads, channels, length, capacity)
    channel_dest = bytearray(len(channel_source))
    scatter_channel_major(channel_dest, tight_values, heads, channels, length, capacity)
    assert channel_dest[0:length] == channel_source[0:length]
