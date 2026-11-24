"""Host checks for the Inkling codec and the effort card. No GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from favlo.prefix import PrefixHeader, RegionRef
from inkling_2_mini.metal41 import prefix as prefix_mod
from inkling_2_mini.metal41.atlas import partial_bytes, plan_opening
from inkling_2_mini.metal41.calibration import (
    _load_round,
    _open_saved,
    _store_round,
    weighted_error,
)
from inkling_2_mini.metal41.codec import e4m3_to_float, nvfp4_row
from inkling_2_mini.metal41.commands import _repo_id, register_folder
from inkling_2_mini.metal41.fetch import File, keep, remaining_download, shortfall
from inkling_2_mini.metal41.forward import extend_past_template, piece_length
from inkling_2_mini.metal41.gptq import (
    PACK_FORMAT,
    PACK_NAME,
    Capture,
    config_digest,
    foem_fp4,
    gptq_fp4,
    write_pack,
)
from inkling_2_mini.metal41.layout import (
    EXPERTS,
    FULL_LAYERS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    WINDOW,
    stored_k,
)
from inkling_2_mini.metal41.prefix import (
    ROW_BYTES,
    VISIT_LAYERS,
    VISITS,
    InklingPrefix,
    cache_slots,
    template_boundary,
    visit_rows,
)
from inkling_2_mini.metal41.resident import (
    EXPERT_FORMAT,
    EXPERT_PACK,
    ExpertFile,
    Sink,
    catalog,
    open_expert_pack,
    pread_view,
)
from inkling_2_mini.metal41.weights import _quantize_fp4, open_pack
from inkling_2_mini.portable import markup
from inkling_2_mini.portable.chat import _required_name, forced_text
from inkling_2_mini.portable.sampling import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VARIANT,
    EFFORT_LEVEL,
    PRESERVE_THINKING,
    VARIANTS,
    resolve_effort,
    template_effort,
)


def test_e4m3_one_and_max() -> None:
    assert e4m3_to_float(0x38) == 1.0
    assert e4m3_to_float(0x7E) == 448.0


def test_nvfp4_row_uses_the_outer_scale() -> None:
    packed = np.full(16, 0x11, dtype=np.uint8)
    scales = np.full(2, 0x38, dtype=np.uint8)
    row = nvfp4_row(packed, scales, 2.0)
    assert row.shape == (32,)
    assert np.allclose(row, 1.0)


def test_plan_uses_the_default_window_when_it_fits() -> None:
    def activation(ctx: int) -> int:
        return ctx * 1000

    top = 131072
    context, complete = plan_opening(top * 1000 + 50_000_000, 40_000_000, activation)
    assert context == top
    assert complete is True


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


def test_template_effort_is_the_number_the_checkpoint_writes() -> None:
    assert template_effort("high") == 0.9
    assert template_effort("none") == 0.0
    assert template_effort("max") == 0.99
    assert template_effort("  Medium ") == 0.7
    assert template_effort(0.35) == 0.35
    assert template_effort("0.35") == 0.35
    assert template_effort(2) == 0.99
    assert template_effort(-0.2) == 0.0
    try:
        template_effort("sideways")
    except ValueError as exc:
        assert "template level" in str(exc)
    else:
        raise AssertionError("an unknown effort must be rejected")


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
        self.k_cache = []
        self.v_cache = []
        for layer in VISIT_LAYERS:
            slots = cache_slots(layer, context) * ROW_BYTES
            self.k_cache.append(bytearray(slots))
            self.v_cache.append(bytearray(slots))


def test_a_user_marker_ends_the_prefill_piece() -> None:
    marker = 7
    prompt = [1, 1, marker, 2, 2, marker, 3]
    assert piece_length(prompt, 0, 7, 8, marker) == 3
    assert piece_length(prompt, 3, 4, 8, marker) == 3
    assert piece_length(prompt, 0, 2, 8, marker) == 2
    assert piece_length(prompt, 6, 1, 8, marker) == 1


def test_a_template_boundary_is_only_the_first_marker() -> None:
    marker = 7
    assert template_boundary([1, 1, marker], marker)
    assert not template_boundary([1, marker, 2, marker], marker)
    assert not template_boundary([1, 2, 3], marker)


def test_the_first_user_marker_stays_in_the_piece() -> None:
    marker = 7
    prompt = [1, 1, marker, 2, 2, marker, 3]
    first = piece_length(prompt, 0, 7, 8, marker)
    assert extend_past_template(prompt, 0, first, 7, 8, marker) == 6
    second = piece_length(prompt, 3, 4, 8, marker)
    assert extend_past_template(prompt, 3, second, 4, 8, marker) == 3
    only = [1, marker, 4, 4]
    opened = piece_length(only, 0, 4, 8, marker)
    assert extend_past_template(only, 0, opened, 4, 8, marker) == 4
    long = [1, marker, *range(20)]
    started = piece_length(long, 0, 20, 8, marker)
    assert extend_past_template(long, 0, started, 20, 8, marker) == 8


def test_prefix_keeps_the_window_tail_of_local_visits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prefix_mod, "memory", _Memory())
    assert VISITS == 48
    count = WINDOW + 5
    live = _Runtime(WINDOW + 16, count)
    live.state_all[:] = bytes(range(64))
    index = np.arange(count * ROW_BYTES, dtype=np.int64)
    local_tail = {}
    for visit, layer in enumerate(VISIT_LAYERS):
        keys = ((index + visit) % 251).astype(np.uint8).tobytes()
        values = ((index * 3 + visit) % 251).astype(np.uint8).tobytes()
        if layer in FULL_LAYERS:
            live.k_cache[visit][: len(keys)] = keys
            live.v_cache[visit][: len(values)] = values
            continue
        first, rows = visit_rows(visit, count)
        row = ROW_BYTES
        for plane, blob in ((live.k_cache[visit], keys), (live.v_cache[visit], values)):
            for item in range(rows):
                slot = (first + item) % cache_slots(layer, live.context)
                start = item * row
                src = (first * row) + start
                plane[slot * row : (slot + 1) * row] = blob[src : src + row]
        local_tail[visit] = (keys[first * row : count * row], values[first * row : count * row])
    regions = InklingPrefix(live).capture()
    by_name = {region.name: bytes(region.data) for region in regions}
    full = VISIT_LAYERS.index(5)
    local = 0
    assert len(by_name[f"k{full:02d}"]) == count * ROW_BYTES
    assert len(by_name[f"k{local:02d}"]) == WINDOW * ROW_BYTES
    assert by_name[f"k{local:02d}"] == local_tail[local][0]
    assert by_name[f"v{local:02d}"] == local_tail[local][1]

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
    assert fresh.k_cache[local] == live.k_cache[local]
    assert fresh.v_cache[local] == live.v_cache[local]


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


def test_required_name_reads_the_responses_choice() -> None:
    assert _required_name({"tool_choice": {"type": "function", "name": "read_file"}}) == "read_file"
    assert (
        _required_name({"tool_choice": {"mode": "required", "required_function_name": "read_file"}})
        == "read_file"
    )
    assert _required_name({"tool_choice": "auto"}) is None
    assert _required_name({}) is None


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


def test_foem_beta_zero_matches_gptq_without_dead_columns() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((64, 256)).astype(np.float32)
    draw = rng.standard_normal((256, 48))
    hessian = draw @ draw.T + np.eye(256)
    assert np.array_equal(foem_fp4(matrix, hessian, beta=0.0)[0], gptq_fp4(matrix, hessian)[0])


def test_foem_beta_moves_a_correlated_matrix() -> None:
    rng = np.random.default_rng(6)
    matrix = rng.standard_normal((48, 256)).astype(np.float32)
    draw = rng.standard_normal((256, 32))
    hessian = draw @ draw.T + np.eye(256) * 0.05
    nearest = foem_fp4(matrix, hessian, beta=0.0)
    moved = foem_fp4(matrix, hessian, beta=0.1)
    assert not np.array_equal(nearest[0], moved[0])


def test_capture_streams_and_round_trip(tmp_path: Path) -> None:
    capture = Capture()
    rows = np.ones((4, 8), dtype=np.float32)
    capture.add(0, "shared", rows)
    capture.add(0, "k", 2 * rows)
    capture.add(0, "wo", np.ones((4, 6), dtype=np.float32))
    capture.tokens = 4
    capture.done = 3
    capture.source = "abc"
    assert np.allclose(capture.hessian(0, "k"), 20.0)
    assert np.allclose(capture.hessian(0, "q"), 4.0)
    assert capture.hessian(0, "wo").shape == (6, 6)
    path = tmp_path / "inputs.npz"
    capture.save(path)
    again = Capture.load(path)
    assert again.tokens == 4
    assert again.done == 3
    assert again.source == "abc"
    assert np.allclose(again.hessian(0, "k"), 20.0)
    scratch = Capture()
    scratch.add(0, "q", rows)
    scratch.tokens = 4
    again.absorb(scratch)
    assert again.tokens == 8
    assert np.allclose(again.hessian(0, "q"), 8.0)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, tokens=np.int64(9), **{"0:q": np.eye(2, dtype=np.float32)})
    old = Capture.load(legacy)
    assert old.done == 2**31
    assert old.source == ""


def test_capture_resume_keeps_the_matching_source(tmp_path: Path) -> None:
    capture = Capture()
    capture.add(1, "wo", np.ones((2, 4), dtype=np.float32))
    capture.done = 7
    capture.source = "same"
    capture.tokens = 14
    path = tmp_path / "dense-fp4-inputs.npz"
    capture.save(path)
    assert _open_saved(path, "same").done == 7
    assert _open_saved(path, "other") is None
    kept = {"layers.0.attn.data": np.zeros((2, 2), dtype=np.uint8)}
    partial = tmp_path / "dense-fp4-partial.npz"
    _store_round(partial, "same", kept)
    assert "layers.0.attn.data" in _load_round(partial, "same")
    assert _load_round(partial, "other") == {}


def test_nvfp4_matrix_matches_one_row() -> None:
    from inkling_2_mini.metal41.codec import dequant_nvfp4_matrix, nvfp4_row

    packed = np.arange(16, dtype=np.uint8).reshape(2, 8)
    scales = np.array([[3], [9]], dtype=np.uint8)
    found = dequant_nvfp4_matrix(packed, scales, 0.25)
    expect = np.stack([nvfp4_row(packed[row], scales[row], 0.25) for row in range(2)])
    assert found.shape == (2, 16)
    assert np.allclose(found, expect)


def test_expert_grams_keep_only_routed_rows() -> None:
    from inkling_2_mini.metal41.calibration import ExpertGrams
    from inkling_2_mini.metal41.layout import EXPERTS, HIDDEN, INTERMEDIATE, stored_k

    grams = ExpertGrams(0)
    hidden = np.zeros((3, HIDDEN), dtype=np.float32)
    hidden[1] = 1.0
    ids = np.full((3, 8), 127, dtype=np.uint32)
    ids[1, 0] = 4
    grams.add_up(hidden, ids)
    assert np.allclose(grams.up[4], np.ones((HIDDEN, HIDDEN), dtype=np.float32))
    assert float(np.sum(grams.up[3])) == 0.0
    width = stored_k(INTERMEDIATE)
    gated = np.zeros((4, width), dtype=np.float32)
    gated[1, 0] = 2.0
    grams.add_down(gated, np.array([4], dtype=np.uint32), np.array([1]), np.array([1]))
    assert grams.down[4, 0, 0] == 4.0
    assert float(np.sum(grams.down[4, 1:, 1:])) == 0.0
    assert grams.down.shape[0] == EXPERTS


def test_expert_catalog_pads_the_down_projection() -> None:
    specs = {(spec.layer, spec.proj, spec.kind): spec for spec in catalog()}
    assert len(specs) == LAYERS * 4
    assert specs[(0, "w2", "data")].shape[-1] == stored_k(INTERMEDIATE) // 2
    assert specs[(0, "w13", "data")].nbytes == EXPERTS * (INTERMEDIATE * 2) * (HIDDEN // 2)
    assert specs[(20, "w13", "data")].shape[-1] == HIDDEN
    assert specs[(20, "w13", "data")].nvfp4 is False


def test_pread_fills_eight_mebibyte_pieces(tmp_path: Path) -> None:
    blob = os.urandom((8 << 20) + 100)
    path = tmp_path / "blob"
    path.write_bytes(blob)
    fd = os.open(path, os.O_RDONLY)
    try:
        found = bytearray(len(blob))
        notes: list[int] = []
        pread_view(fd, memoryview(found), 0, notes.append)
    finally:
        os.close(fd)
    assert bytes(found) == blob
    assert notes == [8 << 20, 100]


def test_expert_pack_follows_the_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    header: dict[str, object] = {
        "__metadata__": {"format": EXPERT_FORMAT, "config": config_digest(tmp_path)}
    }
    offset = 0
    for spec in catalog():
        header[spec.name] = {
            "dtype": "U8",
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (tmp_path / EXPERT_PACK).write_bytes(len(raw).to_bytes(8, "little") + raw)
    record = open_expert_pack(tmp_path)
    assert record is not None
    record.close()
    (tmp_path / "config.json").write_text('{"changed": true}')
    assert open_expert_pack(tmp_path) is None


def test_a_small_pack_round_trips(tmp_path: Path) -> None:
    blob = bytes(range(32))
    sink = Sink(tmp_path / "plane.safetensors", [("row", (32,))], {"format": "test"})
    sink.write_view("row", memoryview(blob))
    sink.close()
    record = ExpertFile(tmp_path / "plane.safetensors")
    try:
        file_off, nbytes = record.span("row", (32,))
        found = bytearray(nbytes)
        pread_view(record.fileno(), memoryview(found), file_off)
    finally:
        record.close()
    assert bytes(found) == blob


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


def test_sampling_card_follows_the_published_defaults() -> None:
    assert DEFAULT_VARIANT == "thinking"
    assert DEFAULT_REASONING_EFFORT == "high"
    assert PRESERVE_THINKING is True
    assert set(VARIANTS) == {"thinking", "instruct"}
    assert VARIANTS["thinking"] == VARIANTS["instruct"]
    thinking = VARIANTS["thinking"]
    assert thinking["temperature"] == 1.0
    assert thinking["top_p"] == 1.0
    assert thinking["top_k"] == 0
    assert thinking["presence_penalty"] == 0.0
    assert "reasoning_effort" not in thinking


def test_progress_reports_through_the_console(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from favlo.console import format_rate
    from inkling_2_mini.metal41.forward import InklingRuntime

    notes: list[str] = []
    ended: list[tuple[str, str]] = []
    logged: list[str] = []
    begun: list[dict[str, object]] = []

    class _Console:
        shows_bars = True
        live_decode = True
        load_detail = "inkling_2_mini"

        def begin(self, phase: str, **kwargs: object) -> SimpleNamespace:
            begun.append({"phase": phase, **kwargs})
            return SimpleNamespace(
                phase=phase,
                total=kwargs["total"],
                done=0,
                closed=False,
                unit=kwargs["unit"],
            )

        def end(self, progress: SimpleNamespace, summary: str) -> None:
            progress.closed = True
            ended.append((progress.phase, summary))

        def log(self, line: str) -> None:
            logged.append(line)

        def note(self, topic: str, **fields: object) -> None:
            from favlo.console import format_note

            notes.append(format_note(topic, **fields))

    runtime = InklingRuntime.__new__(InklingRuntime)
    runtime.console = _Console()
    runtime.root = tmp_path
    runtime._request_id = 4
    runtime._disk_done = 0
    runtime._alloc_done = 0
    runtime._prefill_done = 0
    runtime._prefill_progress = None
    (tmp_path / "dense-fp4.safetensors").write_bytes(b"d" * 10)
    (tmp_path / "experts-fp4.safetensors").write_bytes(b"e" * 20)
    runtime._load_row = runtime._begin_load(runtime._opening_bytes())
    runtime.tick(8)
    runtime._note_alloc(4)
    assert runtime._load_row.done == 12
    assert begun[0]["phase"] == "load"
    assert begun[0]["unit"] == "bytes"
    assert begun[0]["detail"] == "inkling_2_mini"
    runtime._note("prompt", tokens=6)
    runtime._note("cache", hit=0)
    row = runtime._begin_phase("prefill", 6)
    runtime._prefill_progress = row
    runtime._note_prefill(3)
    runtime._note_prefill(3)
    runtime._end_phase(row, format_rate("prefill", 6, 2.0))
    runtime._end_phase(row, "prefill again")
    decode = runtime._begin_phase("decode", 128, live=True)
    assert decode is not None
    assert decode.total == 0
    runtime._end_phase(None, format_rate("decode", 0, 0.0))
    assert notes == ["prompt tokens 6", "cache miss"]
    assert ended == [("prefill", "prefill 3.00 tok/s (6 tok in 2.00s)")]
    assert logged == ["decode 0 tok/s"]
    assert row.done == 3
    plain = InklingRuntime.__new__(InklingRuntime)
    plain.console = _Console()
    plain.console.live_decode = False
    plain._request_id = 0
    assert plain._begin_phase("decode", 32, live=True) is None
    assert plain._begin_phase("prefill", 0) is None


def test_calibration_rows_skip_a_short_dialog_and_an_email() -> None:
    from inkling_2_mini.metal41.calibration import (
        _QUOTAS,
        _accept,
        _agent_dialog,
        _hermes_dialog,
    )

    short = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    assert _accept(short) is None
    mailed = [
        {"role": "user", "content": "Write to me"},
        {
            "role": "assistant",
            "content": "Send it to person@example.com and then " + ("word " * 200),
        },
    ]
    assert _accept(mailed) is None
    long = [
        {"role": "user", "content": "Explain sorting. " * 40},
        {"role": "assistant", "content": "A sort orders items. " * 40},
    ]
    assert _accept(long) is not None
    assert [name for name, _count in _QUOTAS] == ["swe", "code", "tool", "chat"]
    assert sum(count for _name, count in _QUOTAS) == 128
    missed = _agent_dialog(
        {
            "resolved": 0,
            "license": "MIT",
            "tools": ["{}"],
            "messages": [{"role": "user", "content": "Fix the bug."}],
        }
    )
    assert missed is None
    kept = _hermes_dialog(
        [
            {"from": "human", "value": "What is the weather in Lisbon?"},
            {
                "from": "gpt",
                "value": (
                    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lisbon"}}\n'
                    "</tool_call>"
                ),
            },
            {
                "from": "tool",
                "value": (
                    '<tool_response>\n{"name": "get_weather", "content": "18 C and clear."}\n'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "Lisbon is 18 C and clear."},
        ],
        '[{"type": "function", "function": {"name": "get_weather"}}]',
    )
    assert kept is not None
    assert kept["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Lisbon"}
    assert kept["messages"][2]["role"] == "tool"
    from inkling_2_mini.portable.chat import Pipeline

    assert Pipeline().has_vision() is True


def test_vision_grid_and_expand() -> None:
    import base64

    import pytest

    from inkling_2_mini.metal41.vision import (
        IMAGE_TOKEN,
        axis_taps,
        decode_png,
        dequant_fp8,
        expand_prompt,
        patch_hw,
        quantize_fp8,
        token_grid,
    )

    horse = Path(__file__).resolve().parent / "assets" / "favlo-horse.png"
    blob = horse.read_bytes()
    image = decode_png(blob)
    assert image.shape == (320, 576, 3)
    assert token_grid(320, 576) == (8, 15)
    url = "data:image/png;base64," + base64.b64encode(blob).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    ids, plan = expand_prompt(messages, [3, IMAGE_TOKEN, 4])
    assert ids == [3, *([IMAGE_TOKEN] * 120), 4]
    assert plan[0][0] == 1 and plan[0][1] == 120
    with pytest.raises(ValueError):
        expand_prompt([], [IMAGE_TOKEN])
    hw = patch_hw(16, 30)
    assert hw.shape == (480, 2)
    assert hw[0].tolist() == [0, 0]
    assert hw[1].tolist() == [0, 1]
    assert hw[2].tolist() == [1, 0]
    assert hw[4].tolist() == [0, 2]
    taps, weights = axis_taps(np.array([0.0]), 16)
    assert taps[0].tolist() == [0, 1]
    assert np.allclose(weights[0], [0.5, 0.5])
    matrix = np.random.default_rng(0).normal(size=(32, 1200)).astype(np.float32)
    codes, scales = quantize_fp8(matrix)
    assert codes.shape == (32, 1216)
    back = dequant_fp8(codes, scales)[:, :1200]
    peak = float(np.max(np.abs(matrix)))
    assert float(np.max(np.abs(back - matrix))) < 0.08 * peak + 1e-3


def test_repo_id_accepts_the_public_link() -> None:
    assert _repo_id("https://huggingface.co/thinkingmachines/Inkling-2-Mini") == (
        "thinkingmachines/Inkling-2-Mini"
    )
    assert _repo_id("thinkingmachines/Inkling-2-Mini") == "thinkingmachines/Inkling-2-Mini"


def test_download_keeps_the_drafter_file_and_counts_a_short_file(tmp_path: Path) -> None:
    assert keep("model-00001-of-00001.safetensors")
    assert keep("tiktoken/vocab.json")
    assert keep("mtp.safetensors")
    assert not keep("README.md")
    assert not keep(".gitattributes")
    (tmp_path / "config.json").write_text("{}")
    files = [File("config.json", 2, None), File("weights.safetensors", 8, "abc")]
    assert remaining_download(tmp_path, files) == 8
    message = shortfall(1 << 40, 0, tmp_path, tmp_path)
    assert message is not None
    assert "GiB" in message


def test_register_writes_the_recommended_card(tmp_path: Path) -> None:
    folder = tmp_path / "models" / "Inkling-2-Mini-fp4-gptq-foem"
    folder.mkdir(parents=True)
    register_folder(folder, alias="inkling_2_mini", root=tmp_path)
    from favlo.registry import load_registry

    row = load_registry(tmp_path)["models"]["Inkling-2-Mini-fp4-gptq-foem"]
    assert Path(row["source"]) == folder
    assert Path(row["directory"]) == folder
    assert row["alias"] == "inkling_2_mini"
    assert row["kv_cache"] == "half"
    assert row["default_variant"] == "thinking"
    assert row["enable_thinking"] is True
    assert row["reasoning_effort"] == "high"
    assert row["preserve_thinking"] is True
    assert set(row["variants"]) == {"thinking", "instruct"}
    assert "reasoning_effort" not in row["variants"]["thinking"]
    assert row["variants"]["thinking"]["temperature"] == 1.0
    assert row["variants"]["instruct"]["top_k"] == 0


def _shard(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    cursor = 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blobs.append(raw)
        cursor += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 8)
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"".join(blobs))


def test_repack_stores_language_as_float16_and_splits_vision(tmp_path: Path) -> None:
    from inkling_2_mini.metal41.repack import repack
    from inkling_2_mini.metal41.weights import Tensors

    values = np.array([[1.0, -2.0], [0.5, 0.25]], np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    visual = np.array([3.0, -4.0], np.float32)
    visual_bits = (visual.view(np.uint32) >> 16).astype(np.uint16)
    _shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.llm.embed.weight": ("BF16", bits),
            "model.visual.proj.weight": ("BF16", visual_bits),
        },
    )
    repack(tmp_path)
    language = Tensors(tmp_path / "model-00001-of-00001.safetensors")
    vision = Tensors(tmp_path / "vision.safetensors")
    try:
        assert "model.visual.proj.weight" not in language.header
        assert language.header["model.llm.embed.weight"]["dtype"] == "F16"
        assert vision.header["model.visual.proj.weight"]["dtype"] == "F16"
        got = language.vector("model.llm.embed.weight", (2, 2))
        assert np.allclose(got, values)
    finally:
        language.close()
        vision.close()
    marker = json.loads((tmp_path / "repack.json").read_text())
    assert marker["format"] == "favlo-inkling-f16-1"


def test_seal_writes_a_raw_record_and_drops_safetensors(tmp_path: Path) -> None:
    from inkling_2_mini.metal41.fetch import remaining_download
    from inkling_2_mini.metal41.record import (
        RawFile,
        open_language,
        open_vision,
        publish,
        ready,
        seal,
    )

    config = '{"hidden": 4}\n'
    (tmp_path / "config.json").write_text(config)
    embed = (np.array([[1.0, -2.0]], np.float32).view(np.uint32) >> 16).astype(np.uint16)
    _shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.llm.embed.weight": ("BF16", embed),
            "model.llm.layers.0.attn.wq_du.weight": ("F16", np.array([1, 2], np.float16)),
            "model.llm.layers.0.mlp.experts.w13_weight": ("U8", np.array([7, 7], np.uint8)),
            "model.llm.norm.weight": ("F32", np.array([0.5, 1.5], np.float32)),
        },
    )
    _shard(
        tmp_path / "experts-fp4.safetensors",
        {
            "layers.0.w13.data": ("U8", np.array([1, 2, 3, 4], np.uint8).reshape(1, 2, 2)),
            "layers.0.w13.scale": ("U8", np.array([9, 8], np.uint8)),
        },
    )
    _shard(
        tmp_path / "dense-fp8.safetensors",
        {"layers.0.attn.data": ("U8", np.array([4, 5, 6, 7], np.uint8))},
    )
    _shard(
        tmp_path / "vision-fp8.safetensors",
        {"pos.code": ("U8", np.array([3, 1], np.uint8))},
    )
    _shard(
        tmp_path / "mtp.safetensors",
        {"mtp.block.weight": ("BF16", np.array([0x3F80], np.uint16))},
    )
    seal(tmp_path)
    assert ready(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) == []
    resident, experts = open_language(tmp_path)
    vision = open_vision(tmp_path)
    try:
        assert "model.llm.layers.0.attn.wq_du.weight" not in resident.header
        assert "model.llm.layers.0.mlp.experts.w13_weight" not in resident.header
        assert np.allclose(resident.vector("model.llm.embed.weight", (1, 2)), [[1.0, -2.0]])
        assert resident.vector("model.llm.norm.weight", (2,)).tolist() == [0.5, 1.5]
        assert experts.span("layers.0.w13.data", (1, 2, 2)) == (0, 4)
        assert experts.blob(0, 4) == b"\x01\x02\x03\x04"
        assert experts.span("layers.0.w13.scale", (2,)) == (4, 2)
        assert vision is not None
        assert vision.blob(0, 2) == b"\x03\x01"
    finally:
        resident.close()
        experts.close()
        if vision is not None:
            vision.close()
    body = json.loads((tmp_path / "manifest.json").read_text())
    drafter = RawFile(tmp_path / "mtp.bin", body["mtp"])
    try:
        assert drafter.header["mtp.block.weight"]["dtype"] == "F16"
        assert np.allclose(drafter.vector("mtp.block.weight", (1,)), [1.0])
    finally:
        drafter.close()
    files = [
        File("model-00001-of-00001.safetensors", 99, "abc"),
        File("config.json", len(config), None),
    ]
    assert remaining_download(tmp_path, files) == 0
    (tmp_path / "model.safetensors.index.json").write_text("{}\n")
    publish(tmp_path)
    assert not (tmp_path / "model.safetensors.index.json").is_file()


def test_a_cached_still_does_not_load_the_tower(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from favlo.console import format_note
    from favlo.vision_cache import store_cache
    from inkling_2_mini.metal41 import vision

    blob = b"still"
    rows = np.zeros((2, vision.TEXT), np.float16)
    (tmp_path / "config.json").write_text("{}\n")
    store_cache(vision._cache_file(tmp_path, blob), {"rows": rows})
    notes: list[str] = []
    runtime = SimpleNamespace(
        root=tmp_path,
        _owner="test",
        _vision_plan=[(0, 2, blob)],
        _note=lambda topic, **fields: notes.append(format_note(topic, **fields)),
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tower")

    monkeypatch.setattr(vision, "Tower", boom)
    monkeypatch.setattr(vision.memory, "upload", lambda data, owner: data)
    vision.bind_vision(runtime, [1, 2], 0, 0)
    assert notes == ["vision cache 2"]
    notes.clear()
    vision.bind_vision(runtime, [1, 2], 2, 0)
    assert notes == ["vision cached 2"]
    assert runtime._vision_bank is None


def test_mxfp8_scale_covers_the_block_peak() -> None:
    from inkling_2_mini.metal41.weights import dequant_mxfp8, quantize_mxfp8

    matrix = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    matrix[0, 0] = 1000.0
    codes, scales = quantize_mxfp8(matrix)
    factor = np.ldexp(np.ones(scales.shape, np.float64), scales.astype(np.int32) - 127)
    peak = np.max(np.abs(matrix.reshape(64, 8, 32)), axis=-1)
    assert np.all(peak <= factor * 448.0 + 1e-6)
    rebuilt = dequant_mxfp8(codes, scales)
    assert float(np.max(np.abs(rebuilt - matrix)) / np.max(np.abs(matrix))) < 0.03


def test_portable_import_leaves_the_device_unloaded() -> None:
    import sys

    for name in list(sys.modules):
        if name == "inkling_2_mini.metal41" or name.startswith("inkling_2_mini.metal41."):
            del sys.modules[name]
        elif name == "inkling_2_mini.portable" or name.startswith("inkling_2_mini.portable."):
            del sys.modules[name]
    import inkling_2_mini.portable

    loaded = [
        name
        for name in sys.modules
        if name == "inkling_2_mini.metal41" or name.startswith("inkling_2_mini.metal41.")
    ]
    assert loaded == []
    assert inkling_2_mini.portable.PIPELINE == "inkling_2_mini"
