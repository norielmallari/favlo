"""Host checks for the Inkling codec and the effort card. No GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from favlo.prefix import PrefixHeader, RegionRef
from inkling_2_mini.metal41 import markup
from inkling_2_mini.metal41 import prefix as prefix_mod
from inkling_2_mini.metal41.atlas import partial_bytes, plan_opening
from inkling_2_mini.metal41.calibration import weighted_error
from inkling_2_mini.metal41.chat import forced_text
from inkling_2_mini.metal41.codec import e4m3_to_float, nvfp4_row
from inkling_2_mini.metal41.commands import _repo_id, register_folder
from inkling_2_mini.metal41.fetch import File, keep, remaining_download, shortfall
from inkling_2_mini.metal41.gptq import (
    PACK_FORMAT,
    PACK_NAME,
    Capture,
    config_digest,
    gptq_fp4,
    write_pack,
)
from inkling_2_mini.metal41.layout import EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, WINDOW, stored_k
from inkling_2_mini.metal41.prefix import (
    ROW_BYTES,
    VISIT_LAYERS,
    VISITS,
    InklingPrefix,
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
from inkling_2_mini.metal41.sampling import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VARIANT,
    EFFORT_LEVEL,
    EFFORT_ORDER,
    PRESERVE_THINKING,
    VARIANTS,
    resolve_effort,
    template_effort,
)
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
    thinking = VARIANTS["thinking"]
    assert thinking["temperature"] == 1.0
    assert thinking["top_p"] == 1.0
    assert thinking["top_k"] == 0
    assert thinking["presence_penalty"] == 0.0
    assert thinking["reasoning_effort"] == "high"
    assert VARIANTS["instruct"]["reasoning_effort"] == "none"
    for name in EFFORT_ORDER:
        assert VARIANTS[name]["reasoning_effort"] == name
        assert VARIANTS[name]["temperature"] == 1.0


def test_repo_id_accepts_the_public_link() -> None:
    assert _repo_id("https://huggingface.co/thinkingmachines/Inkling-2-Mini") == (
        "thinkingmachines/Inkling-2-Mini"
    )
    assert _repo_id("thinkingmachines/Inkling-2-Mini") == "thinkingmachines/Inkling-2-Mini"


def test_download_skips_the_drafter_and_counts_a_short_file(tmp_path: Path) -> None:
    assert keep("model-00001-of-00001.safetensors")
    assert keep("tiktoken/vocab.json")
    assert not keep("mtp.safetensors")
    assert not keep("README.md")
    assert not keep(".gitattributes")
    (tmp_path / "config.json").write_text("{}")
    files = [File("config.json", 2, None), File("weights.safetensors", 8, "abc")]
    assert remaining_download(tmp_path, files) == 8
    message = shortfall(1 << 40, 0, tmp_path, tmp_path)
    assert message is not None
    assert "GiB" in message


def test_register_writes_the_recommended_card(tmp_path: Path) -> None:
    folder = tmp_path / "models" / "Inkling-2-Mini"
    folder.mkdir(parents=True)
    register_folder(folder, alias="inkling_2_mini", root=tmp_path)
    from favlo.registry import load_registry

    row = load_registry(tmp_path)["models"]["Inkling-2-Mini"]
    assert row["alias"] == "inkling_2_mini"
    assert row["kv_cache"] == "half"
    assert row["default_variant"] == "thinking"
    assert row["enable_thinking"] is True
    assert row["reasoning_effort"] == "high"
    assert row["preserve_thinking"] is True
    assert row["variants"]["medium"]["reasoning_effort"] == "medium"
    assert row["variants"]["thinking"]["temperature"] == 1.0
