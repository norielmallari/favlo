"""Shader checks for the Qwen3.6-35B-A3B pipeline. They do not load a checkpoint."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qwen36_35b_a3b.atlas import atlas_slots
from qwen36_35b_a3b.cache import (
    fit_positions,
    kv_buffer_bytes,
    kv_mode,
    max_position,
    parse_memory,
    repack_plane,
)
from qwen36_35b_a3b.checks import run_checks
from qwen36_35b_a3b.convert import expert_slot_map
from qwen36_35b_a3b.formats import EXPERT_IMAGE, dequant_fp4, fp4_image, quantize_fp4
from qwen36_35b_a3b.forward import (
    ThinkBudget,
    _sample,
    assert_think_repeat_cases,
    decode_steps,
    limit_keeps_piece,
)
from qwen36_35b_a3b.tools import (
    StreamText,
    function_has_parameters,
    parse_tool_calls,
    required_function,
)


def test_fp4_tie_picks_the_even_code() -> None:
    block = np.zeros((1, 32), dtype=np.float32)
    block[0, 0] = 6.0
    block[0, 1] = 0.75
    packed, scales = quantize_fp4(block)
    restored = dequant_fp4(packed, scales, 1, 32)
    assert restored[0, 0] == np.float32(6.0)
    assert restored[0, 1] == np.float32(1.0)


def test_expert_slots_follow_the_median_scale() -> None:
    gate = [bytes([10]), bytes([10]), bytes([10]), bytes([10])]
    down = [bytes([10]), bytes([1]), bytes([40]), bytes([10])]
    mapping = expert_slot_map(gate, down)
    assert mapping.tolist() == [1, 0, 3, 2]


def test_load_row_opens_when_the_console_shows_bars() -> None:
    from qwen36_35b_a3b.forward import Qwen36Runtime

    begun: list[dict[str, object]] = []

    class _Bars:
        shows_bars = True

        def begin(self, phase: str, **kwargs: object) -> SimpleNamespace:
            begun.append({"phase": phase, **kwargs})
            return SimpleNamespace(total=kwargs["total"], done=0, closed=False)

    runtime = Qwen36Runtime.__new__(Qwen36Runtime)
    runtime.console = _Bars()
    runtime.root = SimpleNamespace(name="record")
    runtime._load_progress = runtime._begin_load(100)
    runtime._note_load(40)
    assert begun[0]["phase"] == "load"
    assert begun[0]["unit"] == "bytes"
    assert runtime._load_progress.done == 40

    class _Plain:
        shows_bars = False

    runtime.console = _Plain()
    assert runtime._begin_load(100) is None


def test_an_omitted_completion_cap_is_the_context_room() -> None:
    from qwen36_35b_a3b.chat import completion_limit, requested_limit

    assert requested_limit({}) is None
    assert requested_limit({"max_tokens": None, "max_completion_tokens": 16}) == 16
    assert requested_limit({"max_tokens": 32}) == 32
    assert requested_limit({"max_tokens": 0}) == 0
    assert completion_limit(100, 40, None) == 60
    assert completion_limit(100, 40, 10) == 10
    assert completion_limit(100, 40, 80) == 60
    assert completion_limit(100, 140, 10) == 0
    # The room is the context ceiling. A 35k prompt is not cut at the cache
    # that was grown only to hold the prompt.
    assert decode_steps(262144, 262144, 35000) == 227144
    assert decode_steps(32, 262144, 35000) == 32
    assert decode_steps(262144, 36864, 35000) == 1864


def test_a_closed_think_tool_call_is_a_prefix_of_the_next_prompt() -> None:
    root = Path("models/Qwen3.6-35B-A3B-fp4")
    if not (root / "chat_template.jinja").is_file():
        pytest.skip("chat template is not installed")
    from qwen36_35b_a3b.chat import encode_chat, encode_text

    user = "Call read_file with path qwen38_flash_next/runtime.py."
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    call = (
        "<tool_call>\n<function=read_file>\n<parameter=path>\n"
        "qwen38_flash_next/runtime.py\n</parameter>\n</function>\n</tool_call>"
    )
    prompt = encode_chat(root, [{"role": "user", "content": user}], tools, False, None)
    joined = prompt + encode_text(root, call)
    whole = encode_chat(
        root,
        [
            {"role": "user", "content": user},
            {"role": "assistant", "reasoning_content": "", "content": call},
            {"role": "user", "content": "<tool_response>\nfile\n</tool_response>"},
        ],
        tools,
        True,
        None,
    )
    assert whole[: len(joined)] == joined


def test_an_omitted_assistant_turn_stays_a_prefix_of_the_next_prompt() -> None:
    root = Path("models/Qwen3.6-35B-A3B-fp4")
    if not (root / "chat_template.jinja").is_file():
        pytest.skip("chat template is not installed")
    from qwen36_35b_a3b.chat import encode_chat

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    asked = [
        {"role": "user", "content": "Call read_file with path a.py."},
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n<function=read_file>\n<parameter=path>\n"
                "a.py\n</parameter>\n</function>\n</tool_call>"
            ),
        },
        {"role": "tool", "content": "print(1)\n"},
        {"role": "user", "content": "Call read_file with path b.metal."},
    ]
    previous = encode_chat(root, asked, tools, False, None)
    follow = encode_chat(
        root,
        [*asked, {"role": "user", "content": "Call read_file with path c.py."}],
        tools,
        False,
        None,
    )
    assert follow[: len(previous)] == previous


def test_a_blank_piece_is_dropped_when_the_limit_cuts_an_open_think() -> None:
    assert not limit_keeps_piece(True, True)
    assert limit_keeps_piece(True, False)
    assert limit_keeps_piece(False, True)


def test_a_repeated_span_is_not_drawn_again() -> None:
    produced = [1, 2, 3, 4, 5, 6, 7, 9, 1, 2, 3, 4, 5, 6, 7]
    logits = np.full(16, -10.0, dtype=np.float32)
    logits[9] = np.float32(10.0)
    logits[8] = np.float32(3.0)
    drawn = _sample(logits, 0.0, 1.0, 0, 0.0, set(produced), None, produced)
    assert drawn == 8


def test_recorded_think_clauses_are_not_drawn_again() -> None:
    assert_think_repeat_cases()


def test_a_repeated_clause_ends_after_the_close() -> None:
    budget = ThinkBudget([2, 9, 3], 9, 8)
    assert budget._start(stop_after=True) == 2
    assert budget.stop_after
    assert not budget.closed
    assert budget.forced() == 9
    assert budget.forced() == 3
    assert budget.closed
    assert not budget.forcing


def test_an_unset_thinking_budget_does_not_close_on_a_count() -> None:
    budget = ThinkBudget([2, 9, 3], 9, None)
    for _ in range(32):
        assert budget.take(4) == 4
    assert not budget.forcing
    assert not budget.closed


def test_thinking_budget_forces_the_close_then_drops_that_token() -> None:
    budget = ThinkBudget([2, 9, 3], 9, 2)
    assert budget.take(4) == 4
    assert budget.take(5) == 5
    assert budget.take(8) == 2
    assert budget.forced() == 9
    assert budget.forced() == 3
    assert budget.closed
    logits = np.full(16, -10.0, dtype=np.float32)
    logits[9] = np.float32(10.0)
    logits[8] = np.float32(3.0)
    assert _sample(logits, 0.0, 1.0, 0, 0.0, set(), {9}) == 8


def test_a_stop_inside_an_open_think_starts_the_close() -> None:
    budget = ThinkBudget([2, 9, 3], 9, 8)
    assert budget.instead_of_stop(7, {7}) == 2
    assert budget.forcing
    assert budget.instead_of_stop(7, {7}) is None
    assert budget.forced() == 9
    assert budget.forced() == 3
    assert budget.instead_of_stop(7, {7}) is None


def test_a_natural_close_ends_the_budget() -> None:
    budget = ThinkBudget([2, 9, 3], 9, 8)
    assert budget.take(9) == 9
    assert budget.closed
    assert budget.take(4) == 4


def test_expert_image_is_two_fp4_planes() -> None:
    gate = np.zeros((1024, 2048), dtype=np.float32)
    down = np.zeros((2048, 512), dtype=np.float32)
    assert len(fp4_image(gate) + fp4_image(down)) == EXPERT_IMAGE


def test_stream_delta_holds_a_partial_character() -> None:
    from qwen36_35b_a3b.chat import _stream_delta

    piece, shown = _stream_delta("hello\ufffd", "", final=False)
    assert piece == "hello"
    assert shown == "hello"
    piece, shown = _stream_delta("hello世", shown, final=False)
    assert piece == "世"
    assert shown == "hello世"


def test_stream_chunks_carry_reasoning_then_the_reply() -> None:
    from types import SimpleNamespace

    from qwen36_35b_a3b.chat import _stream

    class _Handler:
        _model_id = "qwen36-35b-a3b"
        _started = False
        close_connection = False
        events: list[dict] = []
        sent: list[bytes] = []
        connection = SimpleNamespace(settimeout=lambda *_args, **_kwargs: None)

        def send_response(self, _code: int) -> None:
            return None

        def send_header(self, *_args: object) -> None:
            return None

        def end_headers(self) -> None:
            return None

        def _event(self, payload: dict) -> None:
            self.events.append(payload)

        def _send(self, data: bytes) -> None:
            self.sent.append(data)

    class _Runtime:
        _hit_limit = False
        _cached_tokens = 2
        _prefill_seconds = 2.0
        _decode_seconds = 4.0
        _output_ids: list[int] = []

        def decode(self, ids: list[int]) -> str:
            words = {1: "Need", 2: " to", 3: "</think>", 4: "4"}
            return "".join(words[token] for token in ids)

        def generate(self, prompt, limit, temperature, top_p, top_k, presence, stop, **kwargs):
            del prompt, limit, temperature, top_p, top_k, presence, stop
            ids = [1, 2, 3, 4]
            on_token = kwargs["on_token"]
            for token in ids:
                on_token(token)
            self._output_ids = ids
            return ids

    handler = _Handler()
    handler.events = []
    handler.sent = []
    _stream(handler, _Runtime(), [9, 9, 9], [], False, 8, 0.0, 1.0, 1, 0.0, True, [], None)
    assert handler.sent == [b"data: [DONE]\n\n"]
    seen: list[tuple[str, object]] = []
    for payload in handler.events:
        assert payload["object"] == "chat.completion.chunk"
        assert payload["id"] == "chatcmpl-favlo"
        assert payload["model"] == "qwen36-35b-a3b"
        assert isinstance(payload["created"], int)
        choices = payload["choices"]
        if not choices:
            seen.append(("usage", payload["usage"]["timings"]["predicted_n"]))
            assert payload["usage"]["timings"]["prompt_n"] == 1
            assert payload["usage"]["timings"]["predicted_per_second"] == 1.0
            continue
        delta = choices[0]["delta"]
        if "reasoning_content" in delta:
            seen.append(("reasoning", delta["reasoning_content"]))
        elif "content" in delta:
            seen.append(("content", delta["content"]))
        elif choices[0].get("finish_reason"):
            seen.append(("finish", choices[0]["finish_reason"]))
    assert seen == [
        ("reasoning", "Need"),
        ("reasoning", " to"),
        ("content", "4"),
        ("finish", "stop"),
        ("usage", 4),
    ]


def _qwen36_record() -> Path | None:
    from favlo.registry import load_registry, models_dir

    root = models_dir()
    for folder, entry in load_registry()["models"].items():
        if entry.get("pipeline") != "qwen36-35b-a3b":
            continue
        path = root / folder
        if (path / "tokenizer.json").is_file() and (path / "chat_template.jinja").is_file():
            return path
    return None


def test_openai_client_streams_the_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    openai = pytest.importorskip("openai")
    from qwen36_35b_a3b.chat import handle_chat
    from tokenizers import Tokenizer

    from favlo import listener

    root = _qwen36_record()
    if root is None:
        pytest.skip("qwen36 record is not registered")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    ids = tokenizer.encode("Need to").ids + [248069] + tokenizer.encode("4").ids
    rendered = tokenizer.decode(ids)
    reasoning_expect, _, content_expect = rendered.partition("</think>")
    content_expect = content_expect.lstrip("\n")

    class _Runtime:
        def __init__(self) -> None:
            self.root = root
            self.max_pos = 262144
            self._hit_limit = False
            self._halted = False
            self._stop = None
            self._cached_tokens = 0
            self._prefill_seconds = 0.5
            self._decode_seconds = 0.25
            self._output_ids: list[int] = []

        def generate(self, prompt, limit, temperature, top_p, top_k, presence, stop, **kwargs):
            del prompt, limit, temperature, top_p, top_k, presence, stop
            on_token = kwargs["on_token"]
            for token in ids:
                on_token(int(token))
            self._output_ids = [int(token) for token in ids]
            return list(self._output_ids)

    state: dict = {
        "runtime": _Runtime(),
        "server": None,
        "record": None,
        "module": SimpleNamespace(handle_chat=handle_chat),
        "pipeline": "qwen36-35b-a3b",
        "requests": 0,
    }

    def bind(body, bound_state, bound_args):
        del body, bound_state, bound_args
        return SimpleNamespace(requested_id="qwen36-35b-a3b", variant=None, sampling=None)

    monkeypatch.setattr(listener, "_bind_request", bind)
    args = SimpleNamespace(host="127.0.0.1", port=0, lazy=False, transcript=None)
    thread = threading.Thread(target=listener._serve_loop, args=(args, state), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while state["server"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    server = state["server"]
    assert server is not None
    port = int(server.server_address[1])
    try:
        client = openai.OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local",
        )
        stream = client.chat.completions.create(
            model="qwen36-35b-a3b",
            messages=[{"role": "user", "content": "2+2"}],
            max_tokens=8,
            stream=True,
            extra_body={"enable_thinking": True},
        )
        reasoning: list[str] = []
        content: list[str] = []
        finish = None
        for chunk in stream:
            assert chunk.id == "chatcmpl-favlo"
            assert chunk.object == "chat.completion.chunk"
            assert chunk.model == "qwen36-35b-a3b"
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            piece = getattr(delta, "reasoning_content", None)
            if piece is None and isinstance(getattr(delta, "model_extra", None), dict):
                piece = delta.model_extra.get("reasoning_content")
            if piece:
                reasoning.append(piece)
            if delta.content:
                content.append(delta.content)
            if choice.finish_reason:
                finish = choice.finish_reason
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert "".join(reasoning) == reasoning_expect
    assert "".join(content) == content_expect
    assert finish == "stop"


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
        ("content", "<tool_call>"),
    ]


def test_a_json_array_parameter_is_a_list() -> None:
    text = (
        "<tool_call>\n<function=save_intro>\n<parameter=lines>\n"
        '["Hello! I\'m Favlo, a small horse.", "I\'d love a carrot."]\n'
        "</parameter>\n</function>\n</tool_call>"
    )
    _content, calls = parse_tool_calls(text)
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments["lines"] == [
        "Hello! I'm Favlo, a small horse.",
        "I'd love a carrot.",
    ]


def test_tool_call_text_becomes_an_openai_call() -> None:
    text = (
        "ok\n<tool_call>\n<function=create_file>\n"
        "<parameter=path>\nnote.txt\n</parameter>\n"
        "<parameter=content>\nThe harbor log is open.\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, calls = parse_tool_calls(text)
    assert content == "ok"
    assert calls[0]["function"]["name"] == "create_file"
    assert "note.txt" in calls[0]["function"]["arguments"]
    choice = {"type": "function", "function": {"name": "create_file"}}
    assert required_function(choice) == "create_file"


def test_empty_function_has_no_parameters() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert function_has_parameters(tools, "read_file") is False
    assert function_has_parameters(tools, "missing") is True
    with_path = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    assert function_has_parameters(with_path, "read_file") is True


def test_record_name_is_the_source_model_plus_fp4() -> None:
    from qwen36_35b_a3b.convert import record_name

    plain = Path("models/Example-Model")
    assert record_name(plain) == "Example-Model-fp4"
    snapshot = Path("hub/models--org--Example-Model/snapshots/abc123")
    assert record_name(snapshot) == "Example-Model-fp4"


def test_s3_evicts_a_one_hit_before_a_reused_expert() -> None:
    from qwen36_35b_a3b.atlas import S3Fifo

    span = 100
    fifo = S3Fifo(span * 10, span)
    one = (0, 1)
    other = (0, 2)
    fifo.insert(one)
    fifo.insert(other)
    assert fifo.relieve(set()) == one
    fifo.hit(other)
    fifo.insert((0, 3))
    assert fifo.relieve(set()) is None
    assert fifo.place(other) == "m"
    fifo.insert((0, 4))
    assert fifo.relieve(set()) == (0, 3)
    assert fifo.place(other) == "m"


def test_a_longer_cache_keeps_the_tokens_already_stored() -> None:
    mode = "e4m3"
    old, new, count = 2048, 4096, 5
    src = np.zeros(kv_buffer_bytes(mode, old), dtype=np.uint8)
    src[: 2 * old * 256].reshape(2, old, 256)[1, 4, 9] = 77
    src[2 * old * 256 :].reshape(2, old, 8)[1, 4, 2] = 13
    packed = repack_plane(src, old, new, count, mode, "key")
    assert packed[: 2 * new * 256].reshape(2, new, 256)[1, 4, 9] == 77
    assert packed[2 * new * 256 :].reshape(2, new, 8)[1, 4, 2] == 13
    value = np.zeros(kv_buffer_bytes(mode, old), dtype=np.uint8)
    value[: 2 * old * 256].reshape(2, 256, old)[0, 3, 4] = 41
    moved = repack_plane(value, old, new, count, mode, "value")
    assert moved[: 2 * new * 256].reshape(2, 256, new)[0, 3, 4] == 41


def test_context_shrinks_to_the_bytes_the_cache_can_hold() -> None:
    mode = "e4m3"
    limit = 8192
    fixed = 1000
    def nothing(_positions: int) -> int:
        return 0
    full = fixed + 20 * kv_buffer_bytes(mode, 8192)
    assert fit_positions(full, fixed, mode, limit, nothing, keep=0) == 8192
    one = 20 * kv_buffer_bytes(mode, 2048)
    assert fit_positions(full, fixed, mode, limit, nothing, keep=one) == 6144
    short = fixed + 20 * kv_buffer_bytes(mode, 4096) - 1
    assert fit_positions(short, fixed, mode, limit, nothing, keep=0) == 2048
    assert fit_positions(fixed, fixed, mode, limit, nothing, keep=0) is None
    wider = fit_positions(full, fixed, "f32", limit, nothing, keep=0)
    assert wider is not None and wider < 8192


def test_a_tight_budget_keeps_context_by_shrinking_the_atlas() -> None:
    mode = "e4m3"
    limit = 8192
    fixed = 1000

    def nothing(_positions: int) -> int:
        return 0

    image = 1_671_168
    eight = 8 * image
    layer = 256 * image
    budget = fixed + 20 * kv_buffer_bytes(mode, limit) + eight
    assert fit_positions(budget, fixed, mode, limit, nothing, keep=eight) == limit
    assert fit_positions(budget, fixed, mode, limit, nothing, keep=layer) is None


def test_a_prefill_chunk_splits_experts_that_do_not_fit() -> None:
    from qwen36_35b_a3b.prefill import masked_expert_passes

    ids = np.arange(10, dtype=np.uint32)
    assert masked_expert_passes(ids, 10) is None
    passes = masked_expert_passes(ids, 8)
    assert passes is not None and len(passes) == 2
    assert passes[0].tolist()[:8] == list(range(8))
    assert passes[0][8] == np.uint32(0xFFFFFFFF)
    assert passes[1][0] == np.uint32(0xFFFFFFFF)
    assert passes[1][8:].tolist() == [8, 9]


def test_kv_widths_and_atlas_budget() -> None:
    assert kv_mode(None) == "e4m3"
    assert kv_mode("fp4") == "e2m1"
    assert kv_mode("f32") == "f32"
    positions = 262144
    assert kv_buffer_bytes("e4m3", positions) == 2 * positions * 264
    assert kv_buffer_bytes("e2m1", positions) == 2 * positions * 136
    assert kv_buffer_bytes("f32", positions) == 2 * positions * 1024
    file_bytes = 427_819_008
    reserved = 1 << 30
    assert atlas_slots(reserved + file_bytes * 40, file_bytes, reserved) == 40
    assert atlas_slots(reserved + file_bytes * 4, file_bytes, reserved) == 4
    image = 1_671_168
    images = 40 * 256
    assert atlas_slots((16 << 30), image, 6 << 30, layers=images) < images
    assert atlas_slots((32 << 30), image, 1 << 30, layers=images) == images
    assert parse_memory("32GiB") == 32 << 30
    assert parse_memory("512MiB") == 512 << 20
    try:
        atlas_slots(reserved, file_bytes, reserved)
    except ValueError:
        pass
    else:
        raise AssertionError("a cap below one layer file must fail")
    root = Path("models/Qwen3.6-35B-A3B-fp4")
    if (root / "config.json").is_file():
        assert max_position(root) == 262144


def test_omitted_sampling_uses_the_pipeline_card() -> None:
    from qwen36_35b_a3b.chat import _presence, _sampling
    from qwen36_35b_a3b.sampling import mode_card

    thinking, card = mode_card(None, None)
    assert thinking is True
    assert card is not None
    assert card["presence_penalty"] == 1.5
    closed, instruct = mode_card(None, False)
    assert closed is False
    assert instruct is not None
    assert instruct["temperature"] == 0.7

    assert _sampling({}, True, None) == (1.0, 0.95, 20)
    assert _presence({}, True, None) == 1.5
    assert _sampling({}, False, None) == (0.7, 0.8, 20)
    assert _presence({}, False, None) == 1.5
    assert _presence({"presence_penalty": 0}, True, None) == 0.0


def test_kernels_match_the_host_formulas() -> None:
    report = run_checks()
    failed = [name for name, ok, _detail in report["rows"] if not ok]
    assert report["ok"], failed
