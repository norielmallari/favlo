"""Record sizes, fp8 packing, chat markup, and prefix slices. No GPU."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from favlo.reply import MemoryReply
from favlo.responses import chat_to_responses
from spark_x26_1_7b.cache import take_channel_major, take_token_major
from spark_x26_1_7b.chat import (
    ResponseWire,
    _fields,
    _mode,
    _response_payload,
    _sampling,
    handle_responses,
    parse_tool_calls,
    split_thinking,
)
from spark_x26_1_7b.formats import dequant_fp8, enc_e4m3, quantize_fp8
from spark_x26_1_7b.layout import cache_bytes, full_token_bytes, record_bytes
from spark_x26_1_7b.prefix import scatter_channel_major, scatter_token_major


def test_record_and_cache_bytes() -> None:
    assert record_bytes() == 1_761_579_008
    assert full_token_bytes() == 7392
    assert cache_bytes(2048) == 15_138_816
    assert cache_bytes(131072) == 968_884_224
    assert cache_bytes(1_048_576) == 7_751_073_792


def test_pack_offsets_cover_the_record() -> None:
    from spark_x26_1_7b.convert import _placements

    placed = _placements()
    cursor = 0
    for _raw, _logical, _kind, _rows, _inner, offset, data_bytes, scale_bytes in placed:
        assert offset == cursor
        assert data_bytes > 0
        cursor += data_bytes + scale_bytes
    assert cursor == record_bytes()


def test_fp8_roundtrip_keeps_exact_e4m3_values() -> None:
    codes = enc_e4m3(np.array([0.0, 1.0, -1.0, 384.0, 448.0], dtype=np.float32))
    assert [int(code) for code in codes] == [0x00, 0x38, 0xB8, 0x7C, 0x7E]
    table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    weight = np.resize(table, (64, 256)).copy()
    weight[::2] *= -1.0
    packed, scales = quantize_fp8(weight)
    restored = dequant_fp8(packed, scales, 64, 256)
    assert np.allclose(restored, weight)


def test_turn_markers_stop_the_completion() -> None:
    from spark_x26_1_7b.chat import STOP

    assert STOP == {0, 1, 2}


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


def test_effort_selects_the_template_level() -> None:
    selection = SimpleNamespace(
        variant="low",
        sampling={"temperature": 0.2, "top_p": 0.3, "top_k": 4, "presence_penalty": 1.0},
        variants={
            "low": {"temperature": 0.2, "top_p": 0.3, "top_k": 4, "presence_penalty": 1.0},
            "high": {"temperature": 0.9, "top_p": 0.8, "top_k": 7, "presence_penalty": 0.0},
            "none": {"temperature": 0.1, "top_p": 0.1, "top_k": 1, "presence_penalty": 0.0},
        },
        default_variant="high",
    )
    thinking, effort, card = _mode(selection, {})
    assert thinking is True and effort == "low" and card["top_k"] == 4
    thinking, effort, card = _mode(selection, {"reasoning": {"effort": "high"}})
    assert thinking is True and effort == "high" and card["top_k"] == 7
    thinking, effort, card = _mode(selection, {"reasoning": {"effort": "minimal"}})
    assert effort == "low"
    thinking, effort, card = _mode(selection, {"reasoning": {"effort": "xhigh"}})
    assert effort == "high"
    thinking, effort, card = _mode(selection, {"reasoning": {"effort": "none"}})
    assert thinking is False and effort is None and card["top_k"] == 1
    temperature, _top_p, top_k, presence, frequency = _sampling(
        {"temperature": 0.4, "top_k": 99, "presence_penalty": 3}, card
    )
    assert temperature == 0.4
    assert top_k == 1
    assert presence == 0.0
    assert frequency == 0.0
    low = chat_to_responses(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False,
            "reasoning_effort": "low",
        }
    )
    thinking, effort, _card = _mode(None, low)
    assert thinking is True and effort == "low"
    off = chat_to_responses(
        {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False}
    )
    assert _mode(None, off)[0] is False


def test_responses_object_keeps_the_trace_and_the_call() -> None:
    runtime = SimpleNamespace(_hit_limit=False, _cached_tokens=2, limit=2048)
    reply = SimpleNamespace(model_id="spark-x26-1-7b")
    text = (
        "<think>trace</think>see https://example.test/card"
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>carrots</arg_value></tool_call>"
    )
    payload = _response_payload(
        reply,
        runtime,
        {"store": False, "reasoning": {"effort": "high", "summary": "concise"}},
        None,
        "Card https://example.test/card\nA note",
        text,
        True,
        True,
        10,
        6,
    )
    kinds = [item["type"] for item in payload["output"]]
    assert kinds == ["reasoning", "message", "function_call"]
    reasoning = payload["output"][0]
    assert reasoning["content"][0]["text"] == "trace"
    assert reasoning["summary"][0]["text"] == "trace"
    assert reasoning["encrypted_content"] == "trace"
    message = payload["output"][1]["content"][0]
    assert message["text"] == "see https://example.test/card"
    assert message["annotations"][0]["type"] == "url_citation"
    assert message["annotations"][0]["url"] == "https://example.test/card"
    call = payload["output"][2]
    assert call["name"] == "lookup"
    assert call["arguments"] == '{"q":"carrots"}'
    assert payload["status"] == "completed"
    assert payload["usage"]["output_tokens_details"]["reasoning_tokens"] == 0
    hidden = _response_payload(
        reply,
        runtime,
        {"tool_choice": "none"},
        None,
        "",
        text,
        True,
        False,
        10,
        6,
    )
    assert [item["type"] for item in hidden["output"]] == ["reasoning", "message"]
    capped = SimpleNamespace(_hit_limit=True, _cached_tokens=0, limit=2048)
    incomplete = _response_payload(
        reply,
        capped,
        {"include": ["message.output_text.logprobs"]},
        None,
        "",
        "still thinking",
        True,
        True,
        4,
        16,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_details"]["reason"] == "max_output_tokens"
    assert incomplete["output"][-1]["content"][0]["logprobs"] == []
    reasoning, content, calls = _fields("<tool_call>lookup", True)
    assert reasoning == "" and content == "" and calls == []


def test_responses_route_rejects_what_the_core_does_not_run() -> None:
    import spark_x26_1_7b as pipeline

    assert callable(pipeline.handle_responses)
    reply = MemoryReply("spark-x26-1-7b")
    handle_responses.complete(
        reply,
        None,
        {"max_output_tokens": 8, "input": "hello"},
        None,
        None,
        {},
    )
    assert reply.status == 400
    reply = MemoryReply("spark-x26-1-7b")
    handle_responses.complete(
        reply,
        SimpleNamespace(),
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.test/a.png"}],
                }
            ]
        },
        None,
        None,
        {},
    )
    assert reply.status == 400
    assert reply.body["error"]["message"] == "input_image is an invalid request"
    inner = MemoryReply("spark-x26-1-7b")
    ResponseWire(inner).done()
    assert inner.sent == []


def test_prefix_slice_roundtrip() -> None:
    heads, channels, length, capacity = 2, 4, 3, 8
    source = bytearray(heads * capacity * channels)
    for head in range(heads):
        for token in range(length):
            start = (head * capacity + token) * channels
            source[start : start + channels] = bytes(
                [head * 10 + token + offset for offset in range(channels)]
            )
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
