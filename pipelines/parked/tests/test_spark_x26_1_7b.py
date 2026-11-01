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
    assert full_token_bytes("e4m3") == 7392
    assert full_token_bytes("f16") == 14336
    assert cache_bytes(2048, "e4m3") == 15_138_816
    assert cache_bytes(2048, "f16") == 29_360_128
    assert cache_bytes(131072, "e4m3") == 968_884_224
    assert cache_bytes(1_048_576, "e4m3") == 7_751_073_792
    assert cache_bytes(1_048_576, "f16") == 15_032_385_536


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


def test_phase_rows_use_the_console() -> None:
    from favlo.console import format_note, format_rate
    from spark_x26_1_7b.forward import SparkRuntime

    notes: list[str] = []
    ended: list[str] = []
    logged: list[str] = []

    class _Console:
        shows_bars = True
        live_decode = False

        def note(self, topic: str, **fields: object) -> None:
            notes.append(format_note(topic, **fields))

        def begin(self, phase: str, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(phase=phase, total=kwargs["total"], done=0, closed=False)

        def end(self, progress: SimpleNamespace, summary: str) -> None:
            progress.closed = True
            ended.append(summary)

        def log(self, line: str) -> None:
            logged.append(line)

    runtime = SparkRuntime.__new__(SparkRuntime)
    runtime.console = _Console()
    runtime._request_id = 0
    runtime._prefill_progress = None
    runtime._note("prompt", tokens=12)
    runtime._note("cache", hit=0)
    runtime._note("prefix_load", tokens=4)
    runtime._note("prefix_store", tokens=4)
    row = runtime._begin_phase("prefill", 8)
    runtime._prefill_progress = row
    runtime._note_prefill(3)
    runtime._note_prefill(8)
    runtime._end_phase(row, format_rate("prefill", 8, 2.0))
    runtime._end_phase(row, "prefill again")
    runtime._end_phase(None, format_rate("decode", 0, 0.0))
    assert notes == [
        "prompt tokens 12",
        "cache miss",
        "prefix load 4 tokens",
        "prefix store 4 tokens",
    ]
    assert row.done == 8
    assert ended == ["prefill 4.00 tok/s (8 tok in 2.00s)"]
    assert logged == ["decode 0 tok/s"]
    assert runtime._begin_phase("decode", 32, live=True) is None


def test_a_prefill_chunk_advances_the_row_when_it_finishes() -> None:
    from spark_x26_1_7b.forward import SparkRuntime
    from spark_x26_1_7b.layout import CHUNK

    class _Session:
        def __init__(self) -> None:
            self.active = False
            self.ended = 0

        def begin(self) -> None:
            self.active = True

        def end(self) -> None:
            self.active = False
            self.ended += 1

    runtime = SparkRuntime.__new__(SparkRuntime)
    runtime.limit = 2048
    runtime.session = _Session()
    runtime._ensure = lambda _pos: True
    runtime._encode = lambda _tokens, _pos, _logits: None
    runtime._check_stop = lambda: None
    runtime._prefill_progress = SimpleNamespace(total=CHUNK + 3, done=0)
    runtime._prefill(list(range(CHUNK + 3)), 0, True)
    assert runtime.session.ended == 1
    assert runtime._prefill_progress.done == CHUNK
    runtime._prefill_progress = SimpleNamespace(total=4, done=0)
    runtime._prefill(list(range(4)), 0, False)
    assert runtime._prefill_progress.done == 4


def test_generate_closes_rows_and_a_cancel_leaves_the_rate_off() -> None:
    from favlo.console import format_rate
    from spark_x26_1_7b.closed import RequestClosed
    from spark_x26_1_7b.forward import SparkRuntime

    begun: list[dict[str, object]] = []
    ended: list[tuple[str, int, str]] = []

    class _Console:
        shows_bars = True
        live_decode = True
        load_detail = ""

        def note(self, topic: str, **fields: object) -> None:
            del topic, fields

        def begin(self, phase: str, **kwargs: object) -> SimpleNamespace:
            begun.append({"phase": phase, **kwargs})
            return SimpleNamespace(phase=phase, total=kwargs["total"], done=0, closed=False)

        def end(self, progress: SimpleNamespace, summary: str) -> None:
            progress.closed = True
            ended.append((progress.phase, progress.done, summary))

        def log(self, line: str) -> None:
            del line

    class _Session:
        def __init__(self) -> None:
            self.active = False

        def begin(self) -> None:
            self.active = True

        def end(self) -> None:
            self.active = False

    class _Prefix:
        def __init__(self) -> None:
            self.loaded = True
            self.tokens = [4]
            self.dirty = False

        def align(self, prompt: list[int]) -> int:
            del prompt
            return 1

        def note(self, tokens: list[int]) -> None:
            self.tokens = list(tokens)

        def flush(self) -> None:
            return None

    runtime = SparkRuntime.__new__(SparkRuntime)
    runtime.console = _Console()
    runtime._request_id = 4
    runtime._prefill_progress = None
    runtime._hit_limit = False
    runtime._prefill_seconds = 0.0
    runtime._decode_seconds = 0.0
    runtime._output_ids = []
    runtime._cached_tokens = 0
    runtime._stop = None
    runtime.limit = 128
    runtime.kv = SimpleNamespace(length=8)
    runtime.session = _Session()
    runtime.prefix = _Prefix()
    runtime.sampler = SimpleNamespace(
        reseed=lambda _seed: None,
        clear=lambda: None,
        observe=lambda _token: None,
    )
    runtime._ensure = lambda _pos: True
    runtime._encode = lambda _tokens, _pos, _logits: None
    runtime._forward = lambda _tokens, _pos, _logits: None

    def _draw(*_args: object) -> int:
        runtime.session.end()
        return 9

    runtime._draw = _draw
    sent: list[int] = []
    out = runtime.generate([4, 5, 6], 1, set(), on_token=sent.append)
    assert out == [9]
    assert sent == [9]
    assert [item["phase"] for item in begun] == ["prefill", "decode"]
    assert begun[0]["total"] == 2
    assert begun[0]["detail"] == "req #4"
    assert begun[1]["total"] == 0
    assert ended[0] == ("prefill", 2, format_rate("prefill", 2, runtime._prefill_seconds))
    assert ended[1] == ("decode", 1, format_rate("decode", 1, runtime._decode_seconds))
    assert runtime._decode_seconds == 0.0

    def _stop_during_prefill(*_args: object) -> None:
        raise RequestClosed()

    runtime._prefill = _stop_during_prefill
    ended.clear()
    try:
        runtime.generate([4, 5, 6], 1, set())
    except RequestClosed:
        pass
    else:
        raise AssertionError("a closed request leaves generate")
    assert ended == [("prefill", 0, "")]


def test_weight_copy_reports_each_piece() -> None:
    from io import BytesIO

    from spark_x26_1_7b.forward import _copy_pieces

    payload = bytes(range(256)) * 4
    view = memoryview(bytearray(len(payload)))
    notes: list[int] = []
    _copy_pieces(BytesIO(payload), view, "weights.bin", notes.append)
    assert bytes(view) == payload
    assert notes == [len(payload)]


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
