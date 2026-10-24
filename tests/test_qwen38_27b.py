"""Checkpoint shapes, fp4 packing, and shader checks for Qwen3.8-27B."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qwen38_27b.budget import (
    choose_wire,
    group_tensors,
    growth_budget,
    same_weight_image,
    startup_bytes,
    tensor_span,
)
from qwen38_27b.cache import cache_bytes, plane_bytes, sequence_cache_bytes
from qwen38_27b.chat import _reasoning_effort, _stream, decode_text, encode_chat, encode_text
from qwen38_27b.commands import installed_record
from qwen38_27b.convert import record_name
from qwen38_27b.checks import run_checks
from qwen38_27b.formats import (
    dequant_fp4,
    dequant_fp8,
    ffn_promote_count,
    promoted_layer_ids,
    quantize_fp4,
    quantize_fp8,
    relative_error,
    text_shapes,
)
from qwen38_27b.forward import Qwen38Runtime, repeated_tail
from qwen38_27b.layout import FULL_LAYERS
from qwen38_27b.prefill import prefill_ids
from qwen38_27b.tools import StreamText, parse_tool_calls


def _config() -> dict:
    layers = (["linear_attention"] * 3 + ["full_attention"]) * 16
    return {
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "output_gate_type": "swish",
            "layer_types": layers,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "rope_theta": 10_000_000,
            },
        }
    }


def test_open_runtime_forwards_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from qwen38_27b.runtime import open_runtime

    captured: dict[str, object] = {}

    def fake(path: Path, **kwargs: object) -> str:
        captured["path"] = path
        captured.update(kwargs)
        return "runtime"

    monkeypatch.setattr("qwen38_27b.runtime.Qwen38Runtime", fake)
    assert open_runtime(Path("record"), owner="qwen38-27b:7") == "runtime"
    assert captured["owner"] == "qwen38-27b:7"
    assert captured["max_memory"] is None


def test_a_missing_record_keeps_the_passed_owner(tmp_path: Path) -> None:
    from qwen38_27b.forward import Qwen38Runtime

    owner = "qwen38-27b:7"
    held: dict[str, str] = {}

    class _Probe(Qwen38Runtime):
        def __init__(self, path: Path, **kwargs: object) -> None:
            try:
                super().__init__(path, **kwargs)
            except FileNotFoundError:
                held["owner"] = self._owner
                raise

    with pytest.raises(FileNotFoundError):
        _Probe(tmp_path, owner=owner)
    assert held["owner"] == owner


def test_a_missing_record_mints_an_owner(tmp_path: Path) -> None:
    from qwen38_27b.forward import Qwen38Runtime

    held: dict[str, str] = {}

    class _Probe(Qwen38Runtime):
        def __init__(self, path: Path, **kwargs: object) -> None:
            try:
                super().__init__(path, **kwargs)
            except FileNotFoundError:
                held["owner"] = self._owner
                raise

    with pytest.raises(FileNotFoundError):
        _Probe(tmp_path)
    minted = held["owner"]
    assert minted.startswith("qwen38-27b:")
    assert minted != "qwen38-27b:7"


def test_text_shapes_follow_the_checkpoint_config() -> None:
    shapes = text_shapes(_config())
    assert shapes.hidden == 5120
    assert shapes.layers == 64
    assert shapes.vocab == 248320
    assert shapes.intermediate == 17408
    assert shapes.full_heads == 24
    assert shapes.kv_heads == 4
    assert shapes.head_dim == 256
    assert shapes.linear_q_heads == 16
    assert shapes.linear_v_heads == 48
    assert shapes.linear_head_dim == 128
    assert shapes.conv_kernel == 4
    assert shapes.rotary_pairs == 32
    assert shapes.mrope_section == (11, 11, 10)
    assert shapes.context == 262144
    assert shapes.output_gate == "swish"
    assert shapes.layer_types.count("full_attention") == 16
    assert shapes.layer_types.count("linear_attention") == 48
    assert shapes.layer_types[:4] == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )


def test_one_token_cache_is_33792_bytes() -> None:
    assert plane_bytes(1) == 4 * 264
    assert cache_bytes(1) == 33792
    assert cache_bytes(262144) == 33792 * 262144
    assert sequence_cache_bytes(1, 17) == 35904


def test_repeated_tail_catches_a_copied_span() -> None:
    block = list(range(16))
    assert repeated_tail(block + block[:-1]) is False
    assert repeated_tail(block + block) is True
    assert repeated_tail([7] * 8) is False


def test_fp4_round_ties_to_even() -> None:
    row = np.zeros(32, dtype=np.float32)
    row[0] = 6.0
    row[1] = 0.25
    row[2] = 0.75
    row[3] = -0.75
    packed, scales = quantize_fp4(row.reshape(1, 32))
    got = dequant_fp4(packed, scales, 1, 32)
    assert got[0, 0] == np.float32(6.0)
    assert got[0, 1] == np.float32(0.0)
    assert got[0, 2] == np.float32(1.0)
    assert got[0, 3] == np.float32(-1.0)


def test_tool_call_markup_becomes_an_openai_call() -> None:
    text = (
        "ok\n<tool_call><function=grove>\n"
        "<parameter=count>\n6</parameter>\n"
        "</function></tool_call>"
    )
    prefix, calls = parse_tool_calls(text)
    assert prefix == "ok"
    assert calls[0]["function"]["name"] == "grove"
    assert calls[0]["function"]["arguments"] == '{"count": "6"}'


def test_reasoning_effort_none_uses_the_template_default() -> None:
    assert _reasoning_effort({}) is None
    assert _reasoning_effort({"reasoning_effort": None}) is None
    assert _reasoning_effort({"reasoning_effort": "none"}) is None
    assert _reasoning_effort({"reasoning_effort": "None"}) is None
    assert _reasoning_effort({"chat_template_kwargs": {"reasoning_effort": "none"}}) is None
    assert _reasoning_effort({"reasoning_effort": "low"}) == "low"
    assert _reasoning_effort({"chat_template_kwargs": {"reasoning_effort": "medium"}}) == "medium"


def test_ffn_percentage_selects_whole_layers() -> None:
    assert ffn_promote_count(0, 64) == 0
    assert ffn_promote_count(10, 64) == 6
    assert ffn_promote_count(20, 64) == 12
    assert ffn_promote_count(100, 64) == 64
    assert record_name(0, 64) == "Qwen3.8-27B-fp4"
    assert record_name(12, 64) == "Qwen3.8-27B-fp8-fp4"
    assert record_name(64, 64) == "Qwen3.8-27B-fp8"
    scores = {0: 0.2, 1: 0.2, 2: 0.9}
    assert promoted_layer_ids(scores, 2) == [0, 2]
    row = np.zeros((1, 32), dtype=np.float32)
    row[0, 0] = 448.0
    packed, scales = quantize_fp8(row)
    restored = dequant_fp8(packed, scales, 1, 32)
    assert restored[0, 0] == np.float32(448.0)
    assert relative_error(np.zeros((1, 32), dtype=np.float32)) == 0.0


def test_wired_spans_are_read_in_file_order(tmp_path: Path) -> None:
    from qwen38_27b.resident import _fill

    raw = bytearray(64)
    raw[10:14] = b"ABCD"
    raw[40:44] = b"WXYZ"
    path = tmp_path / "resident.bin"
    path.write_bytes(raw)
    view = bytearray(8)
    copies = [("late", 40, 0, 4), ("early", 10, 4, 4)]
    noted: list[int] = []
    with path.open("rb") as handle:
        _fill(handle, memoryview(view), copies, noted.append)
    assert bytes(view) == b"WXYZABCD"
    assert noted == [4, 4]
    with path.open("rb") as handle:
        with pytest.raises(RuntimeError, match="overlaps"):
            _fill(handle, memoryview(bytearray(16)), [("a", 10, 0, 8), ("b", 12, 4, 4)], None)


def test_empty_think_turn_stays_a_prefix_of_the_next_prompt() -> None:
    root = installed_record()
    if root is None or not (root / "chat_template.jinja").is_file():
        pytest.skip("quantized record is not installed")
    user = [{"role": "user", "content": "Read notes.txt"}]
    first = encode_chat(root, user, None, True)
    assert decode_text(root, first).endswith("<think>\n")
    continuation = (
        "</think>\n\n<tool_call>\n<function=read_file>\n"
        "<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>"
    )
    applied = encode_text(root, decode_text(root, first) + continuation)
    assert applied[: len(first)] == first
    follow = user + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "notes.txt"}}}
            ],
        },
        {"role": "tool", "content": "ok"},
    ]
    second = encode_chat(root, follow, None, True)
    assert second[: len(applied)] == applied
    prose = encode_text(root, decode_text(root, first) + "</think>\n\nThe notes are short.")
    follow_prose = user + [
        {"role": "assistant", "content": "The notes are short.", "reasoning_content": ""}
    ]
    second_prose = encode_chat(root, follow_prose, None, True)
    assert second_prose[: len(prose)] == prose
    reasoned = encode_text(
        root, decode_text(root, first) + "Look at the file.\n</think>\n\nIt is empty."
    )
    follow_reasoned = user + [
        {
            "role": "assistant",
            "content": "It is empty.",
            "reasoning_content": "Look at the file.",
        }
    ]
    second_reasoned = encode_chat(root, follow_reasoned, None, True)
    assert second_reasoned[: len(reasoned)] == reasoned
    first_off = encode_chat(root, user, None, False)
    assert decode_text(root, first_off).endswith("<think>\n\n</think>\n\n")
    applied_off = encode_text(root, decode_text(root, first_off) + "done")
    follow_off = user + [{"role": "assistant", "content": "done"}]
    second_off = encode_chat(root, follow_off, None, False)
    assert second_off[: len(applied_off)] == applied_off


def test_prefill_progress_counts_finished_tokens() -> None:
    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    progress = SimpleNamespace(total=100, done=0)
    runtime._prefill_progress = progress
    runtime._note_prefill(40)
    assert progress.done == 40
    runtime._note_prefill(10)
    assert progress.done == 40
    runtime._note_prefill(500)
    assert progress.done == 100
    runtime._prefill_progress = None
    runtime._note_prefill(1)
    assert progress.done == 100


def test_prefill_ids_notes_each_finished_chunk(monkeypatch) -> None:
    noted: list[int] = []
    monkeypatch.setattr("qwen38_27b.prefill._chunk", lambda *_args: None)
    runtime = SimpleNamespace(
        max_pos=100_000,
        _prefill=object(),
        _check_stop=lambda: None,
        _note_prefill=noted.append,
    )
    prefill_ids(runtime, list(range(5000)), 10)
    assert noted == [2048, 4096, 5000]


def test_discard_returns_a_shared_buffer() -> None:
    from qwen38_27b.session import discard

    seen: list[int] = []

    class _Buffer:
        def setPurgeableState_(self, state: int) -> None:
            seen.append(state)

        def length(self) -> int:
            return 32

        def contents(self) -> None:
            message = "no mapping"
            raise BufferError(message)

    discard(_Buffer())
    discard(object())
    assert seen == [4]


def test_close_drops_the_device_before_another_pipeline_loads() -> None:
    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    weights = object()
    cache = object()
    scratch = object()
    seen: list[object] = []
    flushed: list[bool] = []
    purged: list[int] = []

    class _Buffer:
        def setPurgeableState_(self, state: int) -> None:
            purged.append(state)

        def length(self) -> int:
            return 32

        def contents(self) -> None:
            message = "no mapping"
            raise BufferError(message)

    class _Session:
        def close(self) -> None:
            seen.append(runtime.resident)

    class _Prefix:
        def __init__(self) -> None:
            self.adapter = SimpleNamespace(runtime=runtime)

        def flush(self) -> None:
            flushed.append(runtime.resident is not None and runtime.k_cache == [cache])

    runtime.session = _Session()
    runtime.prefix = _Prefix()
    runtime.resident = _Buffer()
    runtime.k_cache = [cache]
    runtime.v_cache = [cache]
    runtime._held = [weights]
    runtime._prefill = object()
    runtime.plane_probs = scratch
    runtime.console = None
    runtime.close()
    runtime.close()
    assert flushed == [True]
    assert seen == [None]
    assert runtime.session is None
    assert runtime.resident is None
    assert runtime.k_cache == []
    assert runtime.v_cache == []
    assert runtime._held == []
    assert runtime._prefill is None
    assert runtime.plane_probs is None
    assert runtime.prefix is None
    assert purged == [4]


def test_extend_cache_releases_the_planes_it_replaces() -> None:
    released: list[str] = []

    class _Buf:
        def __init__(self, name: str) -> None:
            self.name = name

        def release(self) -> None:
            released.append(self.name)

    class _Session:
        def zeros(self, _nbytes: int) -> _Buf:
            return _Buf("new")

    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    old_k = _Buf("k")
    old_v = _Buf("v")
    old_parts = _Buf("parts")
    runtime.session = _Session()
    runtime.k_cache = [old_k]
    runtime.v_cache = [old_v]
    runtime.plane_parts = old_parts
    runtime.resident = None
    runtime.history = None
    runtime.state = None
    plan = SimpleNamespace(positions=4096, slots=1)
    runtime._extend_cache(plan, 0, [])
    assert released == ["parts", "k", "v"]
    assert runtime.k_cache[0].name == "new"
    assert runtime.max_pos == 4096


def test_extend_cache_keeps_the_old_planes_when_the_device_refuses() -> None:
    released: list[str] = []

    class _Buf:
        def __init__(self, name: str) -> None:
            self.name = name

        def release(self) -> None:
            released.append(self.name)

    class _Session:
        def zeros(self, _nbytes: int) -> _Buf:
            made = _Buf("partial")
            released.append("allocated")
            if released.count("allocated") > 1:
                message = "device refused a longer cache"
                raise RuntimeError(message)
            return made

    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    old = _Buf("old")
    runtime.session = _Session()
    runtime.k_cache = [old]
    runtime.v_cache = [old]
    runtime.plane_parts = old
    plan = SimpleNamespace(positions=4096, slots=1)
    with pytest.raises(RuntimeError, match="device refused"):
        runtime._extend_cache(plan, 0, [])
    assert runtime.k_cache == [old]
    assert "old" not in released
    assert "partial" in released


def test_release_elastic_returns_the_weight_image() -> None:
    released: list[str] = []

    class _Buf:
        def __init__(self, name: str) -> None:
            self.name = name

        def release(self) -> None:
            released.append(self.name)

    class _Weights:
        def __init__(self) -> None:
            self.buffer = _Buf("image")

        def close(self) -> None:
            self.buffer = None

    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    runtime.weights = _Weights()
    runtime.resident = runtime.weights.buffer
    plane = _Buf("plane")
    runtime.k_cache = [plane]
    runtime.v_cache = []
    runtime.hold = None
    runtime.plane_q = _Buf("scratch")
    runtime._prefill = SimpleNamespace(mix=None)
    runtime._held = [plane]
    runtime._release_elastic()
    assert released == ["image", "plane", "scratch"]
    assert runtime.weights is None
    assert runtime.k_cache == []
    assert runtime.plane_q is None


def test_image_close_drops_the_weight_file_mapping(tmp_path: Path) -> None:
    import mmap

    from qwen38_27b.resident import DeviceImage

    path = tmp_path / "resident.bin"
    path.write_bytes(b"x" * 8192)
    handle = path.open("rb")
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    image = DeviceImage.__new__(DeviceImage)
    image._map = mapping
    image._view = None
    image.buffer = None
    image.offsets = {}
    image._staged_offsets = {}
    image.close()
    assert mapping.closed
    handle.close()


def test_session_close_finishes_an_open_pass() -> None:
    from qwen38_27b.session import Session

    session = Session.__new__(Session)
    ended: list[str] = []

    class _Encoder:
        def endEncoding(self) -> None:
            ended.append("end")

    class _Command:
        def commit(self) -> None:
            ended.append("commit")

        def waitUntilCompleted(self) -> None:
            ended.append("wait")

    session._pass_encoder = _Encoder()
    session._pass_command = _Command()
    session._pass_held = [object()]
    session._held = [object()]
    session.pipes = {"gemv_fp4": object()}
    session.queue = object()
    session.device = object()
    session.metal = object()
    session.close()
    assert ended == ["end", "commit", "wait"]
    assert session._pass_encoder is None
    assert session._pass_command is None
    assert session._pass_held == []
    assert session._held == []
    assert session.pipes == {}
    assert session.queue is None
    assert session.device is None
    assert session.metal is None


def test_startup_bytes_cover_the_record_and_the_cache() -> None:
    resident = 1000
    total = startup_bytes(resident, 4096)
    assert total > resident + cache_bytes(4096)
    assert total > resident + plane_bytes(4096) * FULL_LAYERS * 2


def test_load_row_counts_bytes_and_decode_has_no_bar() -> None:
    begun: list[dict] = []

    class _Console:
        shows_bars = True
        live_decode = True
        load_detail = "qwen38-27b"

        def begin(self, phase: str, **kwargs: object) -> SimpleNamespace:
            begun.append({"phase": phase, **kwargs})
            return SimpleNamespace(
                phase=phase,
                total=kwargs["total"],
                done=0,
                closed=False,
                unit=kwargs["unit"],
            )

    runtime = Qwen38Runtime.__new__(Qwen38Runtime)
    runtime.console = _Console()
    runtime.root = SimpleNamespace(name="record")
    runtime._request_id = 3
    runtime._load_progress = runtime._begin_load(100)
    runtime._note_load(40)
    runtime._note_load(80)
    assert runtime._load_progress.done == 100
    assert begun[0]["phase"] == "load"
    assert begun[0]["unit"] == "bytes"
    decode = runtime._begin_phase("decode", 2048, live=True)
    assert decode.total == 0
    prefill = runtime._begin_phase("prefill", 50)
    assert prefill.total == 50


def test_stream_text_emits_reasoning_then_reply() -> None:
    gate = StreamText(True)
    pieces = ("rea", "son</thi", "nk>\nans", "wer<tool_ca", "ll>")
    emitted: list[tuple[str, str]] = []
    for piece in pieces:
        emitted.extend(gate.push(piece))
    emitted.extend(gate.finish())
    assert emitted == [
        ("reasoning", "rea"),
        ("reasoning", "son"),
        ("content", "ans"),
        ("content", "wer"),
    ]


def test_openai_chunk_streams_one_delta_per_token() -> None:
    openai = pytest.importorskip("openai")
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    class _Runtime:
        _hit_limit = False
        _limit = 131072
        max_pos = 4096
        _cached_tokens = 1
        _prefill_seconds = 2.0
        _decode_seconds = 4.0
        _output_ids: list[int] = []

        def decode(self, ids: list[int]) -> str:
            words = {1: "Need", 2: " to", 3: "</think>", 4: "4"}
            return "".join(words[token] for token in ids)

        def generate(self, prompt, limit, stop, **kwargs):
            del prompt, limit, stop
            ids = [1, 2, 3, 4]
            on_token = kwargs["on_token"]
            for token in ids:
                on_token(token)
            self._output_ids = ids
            return ids

    from favlo.reply import MemoryReply

    reply = MemoryReply("qwen38-27b")
    _stream(reply, _Runtime(), [9, 9], [], False, 8, True)
    assert b"data: [DONE]\n\n" in reply.sent
    seen: list[tuple[str, object]] = []
    for payload in reply.events:
        chunk = ChatCompletionChunk.model_validate(payload)
        assert chunk.id == "chatcmpl-favlo"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == "qwen38-27b"
        assert isinstance(chunk.created, int)
        if not chunk.choices:
            seen.append(("usage", payload["usage"]["timings"]["predicted_n"]))
            assert payload["usage"]["context_length"] == 131072
            assert payload["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
            assert payload["usage"]["timings"]["prompt_n"] == 1
            assert payload["usage"]["timings"]["predicted_per_second"] == 1.0
            continue
        delta = chunk.choices[0].delta
        extra = delta.model_extra if isinstance(delta.model_extra, dict) else {}
        if extra.get("reasoning_content"):
            seen.append(("reasoning", extra["reasoning_content"]))
        elif delta.content:
            seen.append(("content", delta.content))
        elif chunk.choices[0].finish_reason:
            seen.append(("finish", chunk.choices[0].finish_reason))
    assert seen == [
        ("reasoning", "Need"),
        ("reasoning", " to"),
        ("content", "4"),
        ("finish", "stop"),
        ("usage", 4),
    ]
    del openai


def test_completion_usage_reports_the_served_window() -> None:
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    from qwen38_27b.chat import _completion

    runtime = SimpleNamespace(
        _limit=131072,
        max_pos=65536,
        _cached_tokens=12,
        _prefill_seconds=2.0,
        _decode_seconds=4.0,
    )
    body = _completion("ok", 100, 4, "stop", [], "", runtime, "example-model")
    usage = body["usage"]
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 104
    assert usage["context_length"] == 131072
    assert usage["prompt_tokens_details"] == {"cached_tokens": 12}
    assert usage["timings"]["prompt_n"] == 88
    assert usage["timings"]["prompt_per_second"] == 44.0
    assert usage["timings"]["predicted_n"] == 4
    assert usage["timings"]["predicted_per_second"] == 1.0
    parsed = ChatCompletion.model_validate(body)
    assert parsed.usage is not None
    assert parsed.usage.prompt_tokens == 100
    assert parsed.usage.prompt_tokens_details is not None
    assert parsed.usage.prompt_tokens_details.cached_tokens == 12
    extra = parsed.usage.model_extra if isinstance(parsed.usage.model_extra, dict) else {}
    assert extra.get("context_length") == 131072


def test_usage_rate_is_zero_when_the_clock_is_zero() -> None:
    from qwen38_27b.chat import _usage

    runtime = SimpleNamespace(
        _limit=131072,
        _cached_tokens=8,
        _prefill_seconds=0.0,
        _decode_seconds=0.0,
    )
    usage = _usage(runtime, 8, 1)
    assert usage["timings"]["prompt_n"] == 0
    assert usage["timings"]["prompt_per_second"] == 0.0
    assert usage["timings"]["predicted_n"] == 1
    assert usage["timings"]["predicted_per_second"] == 0.0


def test_generate_clock_ends_prefill_at_the_first_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(Qwen38Runtime)
    runtime.console = None
    runtime.max_pos = 3
    runtime._limit = 128
    runtime.filled = 0
    clock = {"t": 100.0}

    def now() -> float:
        return clock["t"]

    class Prefix:
        loaded = False
        tokens: list[int] = []
        dirty = False

        def align(self, prompt: list[int]) -> int:
            return 1

        def note(self, tokens: list[int]) -> None:
            self.tokens = list(tokens)
            clock["t"] += 3.0

        def flush(self) -> None:
            return None

    runtime.prefix = Prefix()

    def ensure(needed: int) -> bool:
        if needed == 4:
            clock["t"] += 7.0
            runtime.max_pos = 8
        return True

    runtime.ensure = ensure

    def step(token: int, pos: int, *, logits: bool, rope: tuple[int, int, int] | None = None) -> int | None:
        del rope
        runtime.filled = pos + 1
        clock["t"] += 2.0
        return 7 if logits else None

    runtime.step = step
    runtime._pick = lambda argmax_id, counts: int(argmax_id)
    runtime._check_stop = lambda: None
    monkeypatch.setattr("qwen38_27b.forward.time.perf_counter", now)

    def on_token(_token: int) -> None:
        clock["t"] += 0.5

    out = runtime.generate([9, 10], 2, set(), on_token=on_token)
    assert out == [7, 7]
    assert runtime._prefill_seconds == pytest.approx(2.0)
    assert runtime._decode_seconds == pytest.approx(12.5)

    clock["t"] = 100.0
    runtime.filled = 0
    runtime.max_pos = 3
    one = runtime.generate([9, 10], 1, set(), on_token=on_token)
    assert one == [7]
    assert runtime._prefill_seconds == pytest.approx(2.0)
    assert runtime._decode_seconds == 0.0


def test_openai_tool_call_delta_carries_an_index() -> None:
    pytest.importorskip("openai")
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

    class _Runtime:
        _hit_limit = False
        _cached_tokens = 0
        _prefill_seconds = 0.0
        _decode_seconds = 1.0
        _output_ids: list[int] = []

        def decode(self, ids: list[int]) -> str:
            text = "<tool_call>\n<function=grove>\n</function>\n</tool_call>"
            return text if ids else ""

        def generate(self, prompt, limit, stop, **kwargs):
            del prompt, limit, stop
            kwargs["on_token"](1)
            self._output_ids = [1]
            return [1]

    from favlo.reply import MemoryReply

    reply = MemoryReply("qwen38-27b")
    _stream(reply, _Runtime(), [9], [], False, 8, False)
    calls = [
        chunk.choices[0].delta.tool_calls
        for payload in reply.events
        if (chunk := ChatCompletionChunk.model_validate(payload)).choices
        and chunk.choices[0].delta.tool_calls
    ]
    assert calls and calls[0][0].index == 0
    assert calls[0][0].function is not None
    assert calls[0][0].function.name == "grove"
    arguments = "".join(
        call.function.arguments
        for group in calls
        for call in group
        if call.function is not None and call.function.arguments
    )
    assert arguments == "{}"


def test_tool_call_arguments_join_as_json() -> None:
    text = "<tool_call>\n<function=grove>\n<parameter=count>\n6</parameter>\n</tool_call>"
    gate = StreamText(False)
    events: list[tuple[str, object]] = []
    for index in range(0, len(text), 3):
        events.extend(gate.push(text[index : index + 3]))
    events.extend(gate.finish())
    arguments = "".join(
        payload["arguments"] for kind, payload in events if kind == "tool_args"
    )
    assert arguments == '{"count": "6"}'
    assert events[0] == ("tool_start", {"index": 0, "id": "call_0", "name": "grove"})
    plain = StreamText(False, tools=False)
    pieces: list[tuple[str, object]] = []
    pieces.extend(plain.push(text))
    pieces.extend(plain.finish())
    assert pieces == [("content", text)]


def test_openai_client_receives_a_token_before_generation_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    openai = pytest.importorskip("openai")
    import threading
    import time
    from types import SimpleNamespace

    from qwen38_27b.chat import _stream

    from favlo import listener

    def chat(handler, runtime, body, selection, args, bound_state):
        del body, selection, args, bound_state
        _stream(handler, runtime, [1], [], False, 8, False)

    marks: dict[str, float] = {}

    class _Runtime:
        root = SimpleNamespace()
        _hit_limit = False
        _halted = False
        _stop = None
        _cached_tokens = 0
        _prefill_seconds = 0.0
        _decode_seconds = 0.2
        _output_ids: list[int] = []

        def decode(self, ids: list[int]) -> str:
            return "".join("ab"[token - 1] for token in ids)

        def generate(self, prompt, limit, stop, **kwargs):
            del prompt, limit, stop
            on_token = kwargs["on_token"]
            on_token(1)
            marks["after_first"] = time.monotonic()
            time.sleep(0.4)
            on_token(2)
            self._output_ids = [1, 2]
            marks["done"] = time.monotonic()
            return [1, 2]

    state: dict = {
        "runtime": _Runtime(),
        "server": None,
        "record": None,
        "module": SimpleNamespace(handle_chat=chat),
        "pipeline": "qwen38-27b",
        "requests": 0,
    }

    def bind(body, bound_state, bound_args):
        del body, bound_state, bound_args
        return SimpleNamespace(requested_id="qwen38-27b", variant=None, sampling=None)

    monkeypatch.setattr(listener, "_bind_request", bind)
    args = SimpleNamespace(host="127.0.0.1", port=0, lazy=False, transcript=None, max_memory=None)
    thread = threading.Thread(target=listener._serve_loop, args=(args, state), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while state["server"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    server = state["server"]
    assert server is not None
    port = int(server.server_address[1])
    first = None
    try:
        client = openai.OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local",
            timeout=5.0,
        )
        stream = client.chat.completions.create(
            model="qwen38-27b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            stream=True,
            extra_body={"enable_thinking": False},
        )
        for chunk in stream:
            assert chunk.object == "chat.completion.chunk"
            if not chunk.choices:
                continue
            if first is None and chunk.choices[0].delta.content:
                first = time.monotonic()
        stream.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert first is not None
    assert "done" in marks
    assert first < marks["done"]


def _span_entry(span: int) -> dict:
    return {
        "dtype": "fp4",
        "data_nbytes": span,
        "scale_nbytes": 0,
        "scale_pad": 0,
        "offset": 0,
        "stored_shape": [1, 32],
    }


def _sample_group(*, layer_span: int = 3000, mtp_span: int = 5000, vision: int = 0):
    tensors = {
        "model.language_model.embed_tokens.weight": _span_entry(8000),
        "lm_head.weight": _span_entry(8000),
        "mtp.fc.weight": _span_entry(mtp_span),
    }
    if vision:
        tensors["model.visual.blocks.0.attn.proj.weight"] = _span_entry(vision)
    for layer in range(64):
        tensors[f"model.language_model.layers.{layer}.mlp.gate_proj.weight"] = _span_entry(layer_span)
        tensors[f"model.language_model.layers.{layer}.mlp.up_proj.weight"] = _span_entry(0)
        tensors[f"model.language_model.layers.{layer}.mlp.down_proj.weight"] = _span_entry(0)
    return group_tensors(tensors)


def test_float16_tensor_span_is_two_bytes_per_element() -> None:
    assert tensor_span({"dtype": "f16", "stored_shape": [5120], "offset": 0}) == 10240


def test_fit_drops_the_mtp_head_before_a_language_layer() -> None:
    group = _sample_group()
    full = choose_wire(10**18, 4096, group, 131072)
    assert full.mtp is True
    assert full.wired == tuple(range(64))
    assert full.stage == 0
    cost = startup_bytes(full.weights, 4096, slots=full.slots, drafts=True)
    wire = choose_wire(cost - 1, 4096, group, 131072)
    assert wire.positions == 4096
    assert wire.mtp is False
    assert wire.wired == tuple(range(64))
    assert wire.stage == 0


def test_fit_unwires_language_layers_from_the_end() -> None:
    group = _sample_group()
    off = choose_wire(10**18, 4096, group, 131072, extra_unwire=1)
    assert off.mtp is False
    assert off.wired == tuple(range(64))
    cost = startup_bytes(off.weights, 4096, slots=off.slots, drafts=False)
    wire = choose_wire(cost - 1, 4096, group, 131072)
    assert wire.mtp is False
    assert wire.wired == tuple(range(62))
    assert wire.stage == 3000
    assert wire.positions == 4096


def test_fit_covers_a_prompt_on_the_next_2048_boundary() -> None:
    group = _sample_group()
    full = choose_wire(10**18, 79872, group, 131072)
    budget = startup_bytes(full.weights, 79872, slots=full.slots, drafts=True)
    wire = choose_wire(budget, 79131, group, 131072)
    assert wire.positions == 79872
    assert wire.mtp is True


def test_fit_stops_at_the_longest_length_when_the_prompt_does_not_fit() -> None:
    group = _sample_group()
    bare = choose_wire(10**18, 2048, group, 131072, extra_unwire=65)
    assert bare.mtp is False
    assert bare.wired == ()
    cost = startup_bytes(bare.weights, 2048, slots=bare.slots, drafts=False)
    wire = choose_wire(cost, 90_000, group, 131072)
    assert wire.positions == 2048
    assert wire.mtp is False
    assert wire.wired == ()


def test_same_weight_image_keeps_the_resident_weights_when_the_cache_grows() -> None:
    group = _sample_group()
    current = choose_wire(10**18, 2048, group, 131072)
    nxt = choose_wire(10**18, 4096, group, 131072)
    assert nxt.positions == 4096
    assert same_weight_image(current, nxt)
    dropped = choose_wire(10**18, 4096, group, 131072, extra_unwire=1)
    assert same_weight_image(current, dropped) is False


def test_growth_budget_keeps_the_live_length_when_the_host_copy_is_reserved() -> None:
    group = _sample_group()
    current = choose_wire(10**18, 2048, group, 131072)
    held = startup_bytes(current.weights, 2048, slots=current.slots, drafts=True)
    nxt = choose_wire(10**18, 4096, group, 131072)
    nxt_cost = startup_bytes(nxt.weights, 4096, slots=nxt.slots, drafts=True)
    packed = sequence_cache_bytes(2048, current.slots)
    reclaimable = nxt_cost - held
    open_wire = choose_wire(growth_budget(reclaimable, held, 0), 4096, group, 131072)
    assert open_wire.positions == 4096
    assert open_wire.mtp is True
    wire = choose_wire(growth_budget(reclaimable, held, packed), 4096, group, 131072)
    assert wire.positions < 4096 or wire.mtp is False


def test_vision_tensors_stay_out_of_the_device_image() -> None:
    plain = choose_wire(10**18, 2048, _sample_group(), 131072)
    visual = choose_wire(10**18, 2048, _sample_group(vision=10**9), 131072)
    assert plain.weights == visual.weights


def test_install_reads_one_prefix_region_at_a_time(monkeypatch) -> None:
    import gc
    from pathlib import Path

    from favlo.prefix import PrefixHeader, RegionRef
    from qwen38_27b.prefix import (
        _HISTORY_BYTES,
        _STATE_BYTES,
        CacheAdapter,
        _packed_length,
    )

    count = 4
    slots = 2
    names = ["history", "state"]
    for slot in range(slots):
        names.extend((f"k.{slot}", f"v.{slot}"))
    lengths = {"history": _HISTORY_BYTES, "state": _STATE_BYTES}
    for slot in range(slots):
        lengths[f"k.{slot}"] = _packed_length(count, "key")
        lengths[f"v.{slot}"] = _packed_length(count, "value")
    header = PrefixHeader(
        Path("example.bin"),
        "qwen38-27b",
        b"\x00" * 32,
        "e4m3",
        tuple(range(count)),
        tuple(RegionRef(name, 16384, lengths[name]) for name in names),
        16384,
    )

    class _Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def __del__(self) -> None:
            released.append(self.name)

    released: list[str] = []
    calls: list[list[str]] = []

    def read_regions(_header: PrefixHeader, asked: list[str]) -> dict[str, _Blob]:
        gc.collect()
        assert len(asked) == 1
        if calls:
            assert released[-1] == calls[-1][0]
        name = asked[0]
        calls.append(list(asked))
        return {name: _Blob(name)}

    monkeypatch.setattr("qwen38_27b.prefix.read_regions", read_regions)
    monkeypatch.setattr("qwen38_27b.prefix._unpack", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("qwen38_27b.prefix._write", lambda *_args, **_kwargs: None)
    runtime = SimpleNamespace(
        kv_slots=slots,
        max_pos=32,
        filled=0,
        history=object(),
        state=object(),
        k_cache=[object(), object()],
        v_cache=[object(), object()],
        reset=lambda: None,
    )
    assert CacheAdapter(runtime).install(header)
    gc.collect()
    assert calls == [[name] for name in names]
    assert released == names
    assert runtime.filled == count


def test_image_tokens_use_the_merged_grid() -> None:
    from qwen38_27b.vision import IMAGE_PAD, expand_image_tokens, language_positions

    prompt = expand_image_tokens([1, IMAGE_PAD, 2], [4])
    positions, nxt = language_positions(prompt, [(2, 2)])
    assert prompt == [1, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 2]
    assert positions[0].tolist() == [0, 0, 0]
    assert positions[1].tolist() == [1, 1, 1]
    assert positions[2].tolist() == [1, 1, 2]
    assert positions[3].tolist() == [1, 2, 1]
    assert positions[4].tolist() == [1, 2, 2]
    assert positions[5].tolist() == [3, 3, 3]
    assert nxt == 4


def test_png_round_trip_and_square_grid() -> None:
    import struct
    import zlib

    from qwen38_27b.vision import decode_png, merged_grid, prepare_image

    def png_of(image: np.ndarray) -> bytes:
        height, width, _channels = image.shape
        raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

        def chunk(kind: bytes, payload: bytes) -> bytes:
            check = zlib.crc32(kind + payload) & 0xFFFFFFFF
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", check)

        return b"".join(
            (
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                chunk(b"IDAT", zlib.compress(raw)),
                chunk(b"IEND", b""),
            )
        )

    small = np.zeros((8, 8, 3), dtype=np.uint8)
    small[:, :, 0] = 255
    assert decode_png(png_of(small)).shape == (8, 8, 3)
    square = np.zeros((512, 512, 3), dtype=np.uint8)
    patches, grid = prepare_image(png_of(square))
    assert grid == (32, 32)
    assert merged_grid(grid) == (16, 16)
    assert patches.shape == (32 * 32, 3 * 2 * 16 * 16)


def test_shader_dispatches_match_the_host_formulas() -> None:
    report = run_checks()
    failed = [name for name, ok, _detail in report["rows"] if not ok]
    assert report["ok"], failed


def test_delete_removes_the_record_the_checkout_and_the_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from favlo.prefix import checkpoint_digest
    from favlo.registry import register_quantized
    from qwen38_27b.commands import main

    def models_dir(root: Path | None = None) -> Path:
        del root
        return tmp_path / "models"

    prefix_root = tmp_path / "prefix"
    monkeypatch.setattr("qwen38_27b.commands.models_dir", models_dir)
    monkeypatch.setattr("favlo.registry.models_dir", models_dir)
    monkeypatch.setattr("qwen38_27b.commands.default_prefix_dir", lambda: prefix_root)
    packed = tmp_path / "models" / "Qwen3.8-27B-fp4"
    packed.mkdir(parents=True)
    (packed / "manifest.json").write_text(json.dumps({"complete": True}) + "\n")
    mixed = tmp_path / "models" / "Qwen3.8-27B-fp8-fp4"
    mixed.mkdir()
    (mixed / "manifest.json").write_text(json.dumps({"complete": True, "mixed": True}) + "\n")
    checkout = tmp_path / "models" / "Qwen3.8-27B-bf16"
    checkout.mkdir()
    (checkout / "config.json").write_text("{}\n")
    kept = tmp_path / "models" / "kept-model"
    kept.mkdir()
    (kept / "manifest.json").write_text(json.dumps({"complete": True, "kept": True}) + "\n")
    outside = tmp_path / "outside-bf16"
    outside.mkdir()
    other = tmp_path / "models" / "other-model"
    other.mkdir()
    (other / "manifest.json").write_text("{}\n")
    register_quantized(
        "Qwen3.8-27B-fp4",
        source=checkout,
        alias="qwen38-27b",
        pipeline="qwen38-27b",
        runtime="metal41",
    )
    register_quantized(
        "Qwen3.8-27B-fp8-fp4",
        source=checkout,
        alias=None,
        pipeline="qwen38-27b",
        runtime="metal41",
    )
    register_quantized(
        "kept-model",
        source=outside,
        alias=None,
        pipeline="qwen38-27b",
        runtime="metal41",
    )
    register_quantized(
        "other-model",
        source=outside,
        alias=None,
        pipeline="qwen36-35b-a3b",
        runtime="metal41",
    )
    digest = checkpoint_digest(packed).hex()
    partition = prefix_root / "qwen38-27b" / digest
    partition.mkdir(parents=True)
    (partition / "cache.bin").write_bytes(b"x")
    assert main(["delete", "other-model"]) == 1
    assert other.is_dir()
    assert main(["delete", "qwen38-27b:instruct"]) == 0
    assert not packed.exists()
    assert checkout.is_dir()
    assert not partition.exists()
    assert mixed.is_dir()
    assert main(["delete", "Qwen3.8-27B-fp8-fp4"]) == 0
    assert not mixed.exists()
    assert not checkout.exists()
    assert outside.is_dir()
    assert kept.is_dir()
    registry = json.loads((tmp_path / "models" / "registry.json").read_text())
    assert "Qwen3.8-27B-fp4" not in registry["models"]
    assert "Qwen3.8-27B-fp8-fp4" not in registry["models"]
    assert "kept-model" in registry["models"]
    assert "other-model" in registry["models"]
    assert main(["delete", "kept-model"]) == 0
    assert not kept.exists()
    assert outside.is_dir()
    assert other.is_dir()
