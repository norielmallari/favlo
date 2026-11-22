"""Host checks for the Inkling codec and the effort card. No GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from favlo.prefix import PrefixHeader, RegionRef
from inkling_2_mini.metal41 import markup
from inkling_2_mini.metal41 import prefix as prefix_mod
from inkling_2_mini.metal41.atlas import partial_bytes, plan_opening
from inkling_2_mini.metal41.calibration import weighted_error
from inkling_2_mini.metal41.chat import forced_text
from inkling_2_mini.metal41.codec import e4m3_to_float, nvfp4_row
from inkling_2_mini.metal41.gptq import (
    PACK_FORMAT,
    PACK_NAME,
    Capture,
    config_digest,
    gptq_fp4,
    write_pack,
)
from inkling_2_mini.metal41.layout import WINDOW
from inkling_2_mini.metal41.prefix import (
    ROW_BYTES,
    VISIT_LAYERS,
    VISITS,
    InklingPrefix,
    visit_rows,
)
from inkling_2_mini.metal41.sampling import EFFORT_LEVEL, resolve_effort
from inkling_2_mini.metal41.weights import _quantize_fp4, open_pack


def test_e4m3_one_and_max() -> None:
    assert e4m3_to_float(0x38) == 1.0
    assert e4m3_to_float(0x7E) == 448.0


def test_nvfp4_row_uses_the_outer_scale() -> None:
    packed = np.full(16, 0x11, dtype=np.uint8)
    scales = np.full(2, 0x38, dtype=np.uint8)
    row = nvfp4_row(packed, scales, 2.0)
    assert row.shape == (32,)
    assert np.allclose(row, 1.0)


def test_plan_keeps_context_when_eight_experts_fit() -> None:
    def activation(ctx: int) -> int:
        return ctx * 1000

    act = 16384 * 1000
    context, complete = plan_opening(act + partial_bytes(), act + partial_bytes() + 1, activation)
    assert context == 16384
    assert complete is False
    context, complete = plan_opening(act + 50_000_000, 40_000_000, activation)
    assert context == 16384
    assert complete is True


def test_resolve_effort_keeps_the_template_names() -> None:
    assert resolve_effort(None) is None
    assert resolve_effort("high") == "high"
    assert resolve_effort("  Low ") == "low"
    assert set(EFFORT_LEVEL) == {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


class _Memory:
    def load(self, handle: bytearray, offset: int, nbytes: int) -> bytes:
        return bytes(handle[offset : offset + nbytes])

    def store(self, handle: bytearray, offset: int, data: bytes) -> None:
        handle[offset : offset + len(data)] = data


class _Runtime:
    def __init__(self, context: int, filled: int) -> None:
        self.context = context
        self._filled = filled
        self._ids: list[int] = []
        self.state_bytes = 64
        self.state_all = bytearray(self.state_bytes)
        self.k_cache = [bytearray(context * ROW_BYTES) for _ in range(VISITS)]
        self.v_cache = [bytearray(context * ROW_BYTES) for _ in range(VISITS)]


def test_prefix_keeps_the_window_tail_of_local_visits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prefix_mod, "memory", _Memory())
    assert VISITS == 48
    count = WINDOW + 5
    live = _Runtime(WINDOW + 16, count)
    live.state_all[:] = bytes(range(64))
    index = np.arange(count * ROW_BYTES, dtype=np.int64)
    for visit in range(VISITS):
        live.k_cache[visit][: index.size] = ((index + visit) % 251).astype(np.uint8).tobytes()
        live.v_cache[visit][: index.size] = ((index * 3 + visit) % 251).astype(np.uint8).tobytes()
    regions = InklingPrefix(live).capture()
    by_name = {region.name: bytes(region.data) for region in regions}
    full = VISIT_LAYERS.index(5)
    local = 0
    assert len(by_name[f"k{full:02d}"]) == count * ROW_BYTES
    assert len(by_name[f"k{local:02d}"]) == WINDOW * ROW_BYTES

    path = tmp_path / "record"
    blob = b"".join(by_name[name] for name in by_name)
    path.write_bytes(blob)
    refs = []
    offset = 0
    for name, data in by_name.items():
        refs.append(RegionRef(name, offset, len(data)))
        offset += len(data)
    tokens = tuple(range(count))
    header = PrefixHeader(path, "inkling_2_mini", b"", "half", tokens, tuple(refs), len(blob))
    fresh = _Runtime(WINDOW + 16, 0)
    assert InklingPrefix(fresh).accept(header, list(range(count + 1)))
    assert not InklingPrefix(fresh).accept(header, list(tokens))
    assert InklingPrefix(fresh).install(header)
    assert fresh._filled == count
    assert fresh.state_all == live.state_all
    end = count * ROW_BYTES
    assert fresh.k_cache[full][:end] == live.k_cache[full][:end]
    start, _ = visit_rows(local, count)
    assert start == 5
    tail = slice(start * ROW_BYTES, end)
    assert fresh.v_cache[local][tail] == live.v_cache[local][tail]


_CALL = (
    "<|message_model|><|content_thinking|>Call it.<|end_message|>"
    '<|message_model|>locker<|content_invoke_tool_json|>{"name":"locker","args":{"code":"A-7"}}'
    "<|end_message|><|content_model_end_sampling|>"
)


def test_markup_splits_reasoning_and_the_call() -> None:
    reasoning, visible, calls = markup.parse_completion(_CALL)
    assert reasoning == "Call it."
    assert visible is None
    assert calls == [("locker", {"code": "A-7"})]


def test_markup_keeps_a_body_cut_by_the_limit() -> None:
    reasoning, visible, calls = markup.parse_completion(
        "<|message_model|><|content_text|>A horse eats"
    )
    assert (reasoning, visible, calls) == ("", "A horse eats", [])
    assert markup.parse_completion("plain words") == ("", "plain words", [])


def test_markup_reads_a_raw_newline_inside_a_string() -> None:
    text = '<|message_model|>w<|content_invoke_tool_json|>{"name":"w","args":{"body":"a\nb"}}'
    assert markup.parse_completion(text)[2] == [("w", {"body": "a\nb"})]
    bad = '<|message_model|>w<|content_invoke_tool_json|>{"name":"w","args":{"body":"a\\sb"}}'
    assert markup.parse_completion(bad)[2] == [("w", {"body": "a\\sb"})]


def test_stream_gate_builds_the_call_in_pieces() -> None:
    gate = markup.StreamGate()
    events = []
    for char in _CALL:
        events.extend(gate.push(char))
    events.extend(gate.finish())
    reasoning = "".join(value for channel, value in events if channel == "reasoning")
    starts = [value for channel, value in events if channel == "tool_start"]
    pieces = [value["arguments"] for channel, value in events if channel == "tool_args"]
    assert reasoning == "Call it."
    assert starts == [{"index": 0, "name": "locker"}]
    assert len(pieces) > 1
    assert "".join(pieces) == '{"code":"A-7"}'


def test_forced_text_opens_the_required_call() -> None:
    opening = forced_text(False, "locker", plain=False)
    assert opening.endswith('{"name":"locker","args":')
    assert forced_text(False, None, plain=True) == "<|message_model|><|content_text|>"
    assert forced_text(True, None, plain=True) == ""
    assert forced_text(False, None, plain=False) == ""


def test_gptq_identity_hessian_is_round_to_nearest() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((64, 256)).astype(np.float32)
    packed, scales = gptq_fp4(matrix, np.eye(256))
    nearest = _quantize_fp4(matrix)
    assert np.array_equal(packed, nearest[0])
    assert np.array_equal(scales, nearest[1])


def test_gptq_lowers_the_output_error() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((128, 256)).astype(np.float32)
    inputs = rng.standard_normal((2048, 256)) @ (
        np.eye(256) + 0.2 * rng.standard_normal((256, 256))
    )
    hessian = inputs.T @ inputs
    found = weighted_error(matrix, *gptq_fp4(matrix, hessian), hessian)
    nearest = weighted_error(matrix, *_quantize_fp4(matrix), hessian)
    assert found < 0.7 * nearest


def test_capture_streams_and_round_trip(tmp_path: Path) -> None:
    capture = Capture()
    rows = np.ones((4, 8), dtype=np.float32)
    capture.add(0, "shared", rows)
    capture.add(0, "k", 2 * rows)
    capture.add(0, "wo", np.ones((4, 6), dtype=np.float32))
    capture.tokens = 4
    assert np.allclose(capture.hessian(0, "k"), 20.0)
    assert np.allclose(capture.hessian(0, "q"), 4.0)
    assert capture.hessian(0, "wo").shape == (6, 6)
    path = tmp_path / "inputs.npz"
    capture.save(path)
    again = Capture.load(path)
    assert again.tokens == 4
    assert np.allclose(again.hessian(0, "k"), 20.0)


def test_pack_round_trip(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    data = np.arange(32, dtype=np.uint8).reshape(4, 8)
    write_pack(
        tmp_path / PACK_NAME,
        {"plane.data": data},
        {"format": PACK_FORMAT, "config": config_digest(tmp_path)},
    )
    pack = open_pack(tmp_path)
    assert pack is not None
    try:
        assert np.array_equal(pack.bytes_of("plane.data", (4, 8)), data)
    finally:
        pack.close()
    (tmp_path / "config.json").write_text('{"changed": true}')
    assert open_pack(tmp_path) is None
