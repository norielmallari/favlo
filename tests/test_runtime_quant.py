import io
import json
import mmap
import os

import numpy as np
import pytest
from qwen38_flash_next.formats import DEFAULT_QUANT_MAP, QUANT_LAYOUT_VERSION, quant_file
from qwen38_flash_next.record import QuantRecord
from qwen38_flash_next.repack import version5_place
from qwen38_flash_next.runtime import RequestClosed, mtp_holds, mtp_starts

from madmetal.console import Console
from test_packing import pipeline


def test_quantize_matrix_matches_block() -> None:
    row = np.zeros(32, dtype=np.float32)
    row[0] = 6.0
    row[1] = 1.0
    row[2] = -0.5
    row[3] = 1.5
    matrix = row[None, :]
    packed, scales = pipeline.quantize_matrix(matrix)
    restored = pipeline.dequant_int4(packed, scales, 1, 32)
    assert restored[0, 0] == np.float32(6.0)
    assert restored[0, 1] == np.float32(1.0)
    assert restored[0, 2] == np.float32(-0.5)
    assert restored[0, 3] == np.float32(1.5)
    assert np.all(restored[0, 4:] == 0)
    again, _again_scales = pipeline.quantize_matrix(restored)
    assert again == packed


def test_int4_image_matches_quantize_matrix() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.normal(scale=0.1, size=(3, 256)).astype(np.float32)
    matrix[1, :32] = 0
    image, stored = pipeline.quantize_int4_image(matrix)
    packed, scales = pipeline.quantize_matrix(matrix)
    assert stored == 256
    data_n, _scale_n = pipeline.int4_plane_sizes(3, stored)
    assert image[:data_n] == packed
    assert image[data_n:] == scales


def test_padded_k_leaves_zero_codes() -> None:
    matrix = np.ones((2, 640), dtype=np.float32)
    image, stored = pipeline.quantize_int4_image(matrix)
    assert stored == 768
    data_n, _scale_n = pipeline.int4_plane_sizes(2, stored)
    restored = pipeline.dequant_int4(image[:data_n], image[data_n:], 2, stored)
    assert np.all(restored[:, 640:] == 0)


def test_ngram_slot_is_128_bytes() -> None:
    row = np.linspace(-1.0, 1.0, pipeline.NGRAM_WIDTH, dtype=np.float32)[None, :]
    blob = pipeline.ngram_slots(row)
    assert len(blob) == pipeline.NGRAM_SLOT
    restored = pipeline.dequant_int4(blob[:80], blob[80:85], 1, pipeline.NGRAM_WIDTH)
    assert restored.shape == (1, pipeline.NGRAM_WIDTH)
    assert blob[85:] == bytes(pipeline.NGRAM_SLOT - 85)


def test_middle_expert_image_is_a_whole_number_of_pages() -> None:
    live = pipeline.expert_live_bytes()
    assert live == 2_785_280
    assert live % pipeline.PAGE == 0
    assert pipeline.expert_slot_bytes() == live
    assert (512 * live) % pipeline.PREFERRED_IO == 0


def test_ngram_resume_row_rewinds_when_the_file_is_short() -> None:
    assert pipeline.ngram_resume_row(100, 40 * pipeline.NGRAM_SLOT + 10) == 40


def test_ngram_resume_row_keeps_the_checkpoint_when_the_file_is_ahead() -> None:
    assert pipeline.ngram_resume_row(100, 180 * pipeline.NGRAM_SLOT) == 100


def _full_expert_file(path, experts: int) -> None:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, experts * pipeline.expert_slot_bytes())
    finally:
        os.close(fd)


def test_expert_start_skips_a_finished_layer(tmp_path) -> None:
    path = tmp_path / "00.bin"
    _full_expert_file(path, 2)
    progress = {"layers": [0]}
    start = pipeline._expert_start(
        progress,
        done=True,
        inflight_key=0,
        path=path,
        experts=2,
        slot=pipeline.expert_slot_bytes(),
    )
    assert start == 2


def test_expert_start_resumes_after_the_last_committed_expert(tmp_path) -> None:
    path = tmp_path / "01.bin"
    _full_expert_file(path, 4)
    progress = {"inflight_key": 1, "inflight_expert": 2}
    start = pipeline._expert_start(
        progress,
        done=False,
        inflight_key=1,
        path=path,
        experts=4,
        slot=pipeline.expert_slot_bytes(),
    )
    assert start == 3


def test_expert_start_rewrites_a_marked_layer_whose_file_is_missing(tmp_path) -> None:
    progress = {"layers": [0], "inflight_key": 0, "inflight_expert": 10}
    start = pipeline._expert_start(
        progress,
        done=True,
        inflight_key=0,
        path=tmp_path / "missing.bin",
        experts=4,
        slot=pipeline.expert_slot_bytes(),
    )
    assert start == 0


def test_progress_roundtrip_keeps_the_inflight_expert(tmp_path) -> None:
    progress = {
        "version": pipeline.RECORD_VERSION,
        "resident": True,
        "layers": [0],
        "mtp": False,
        "ngram_rows": 128,
        "inflight_key": 1,
        "inflight_expert": 63,
    }
    pipeline._save_progress(tmp_path, progress)
    loaded = pipeline._load_progress(tmp_path)
    assert loaded["ngram_rows"] == 128
    assert loaded["inflight_key"] == 1
    assert loaded["inflight_expert"] == 63


def test_pread_exact_uses_the_scratch_buffer(tmp_path) -> None:
    payload = bytes(index % 256 for index in range(pipeline.PAGE))
    path = tmp_path / "blob"
    path.write_bytes(payload)
    fd = os.open(path, os.O_RDONLY)
    scratch = mmap.mmap(-1, pipeline.PAGE)
    try:
        view = pipeline.pread_exact(fd, pipeline.PAGE, 0, scratch)
        got = bytes(view)
        del view
        assert got == payload
    finally:
        scratch.close()
        os.close(fd)


def test_gdn_and_attention_shaders_match() -> None:
    pipeline.check_state_kernels()


def test_prefill_always_uses_the_atlas() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime._bf16 = False
    runtime.tokens = [7]
    runtime.atlas = object()
    runtime.kv_len = {}
    runtime.resident_candidates = ()
    seen: list[str] = []
    runtime._ensure_atlas = lambda: seen.append("atlas")
    runtime._prepare_experts(decode=False, count=32)
    runtime._prepare_experts(decode=False, count=4096)
    assert seen == ["atlas", "atlas"]
    assert runtime.atlas is not None


def test_flush_keeps_the_atlas_across_the_prefix_write() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [1, 2]
    order: list[str] = []

    def flush() -> None:
        order.append("flush")

    def state() -> object:
        order.append("state")
        return object()

    def release() -> None:
        order.append("release")

    class Prefixes:
        def save(self, state: object) -> None:
            order.append("save")

    runtime.gpu = type("Gpu", (), {"flush": staticmethod(flush)})()
    runtime._prefix_state = state
    runtime._release_atlas = release
    runtime.prefixes = Prefixes()
    runtime._flush_prefix()
    assert order == ["flush", "state", "save"]


def test_completed_prefill_writes_without_releasing_the_atlas() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [1, 2, 3]
    runtime._prefix_dirty = True
    runtime._forward_open = False
    order: list[str] = []
    runtime.gpu = type("Gpu", (), {"flush": lambda self: order.append("flush")})()
    runtime._prefix_state = lambda: order.append("state") or object()
    runtime._release_atlas = lambda: order.append("release")

    class Prefixes:
        def save(self, state: object) -> object:
            order.append("save")
            return object()

    runtime.prefixes = Prefixes()
    runtime._persist_prefix(release_atlas=False)
    assert runtime._prefix_dirty is False
    assert order == ["flush", "state", "save"]


def test_uncommitted_draft_span_is_rewound_before_the_prefix_write() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = list(range(415))
    runtime.vocab = 100
    runtime._mtp_layer = 48
    runtime.kv_len = {0: 419, 48: 419}
    runtime._kv_snap = {
        0: (None, None, None, None, None, 415),
        48: (None, None, None, None, None, 419),
    }
    runtime._mtp_kv = (None, None, None, None, None, 415)
    runtime._prefix_dirty = True
    runtime._forward_open = False
    runtime.last_logits = type("Logits", (), {"n": 400})()
    runtime.gpu = type("Gpu", (), {"flush": lambda self: None})()
    seen: dict[str, object] = {}

    def restore_recurrent() -> None:
        for index, item in runtime._kv_snap.items():
            runtime.kv_len[index] = item[5]

    def restore_mtp() -> None:
        runtime.kv_len[runtime._mtp_layer] = runtime._mtp_kv[5]

    def state() -> object:
        seen["lengths"] = dict(runtime.kv_len)
        seen["logits"] = runtime.last_logits
        return object()

    runtime._restore_recurrent = restore_recurrent
    runtime._mtp_restore_kv = restore_mtp
    runtime._prefix_state = state
    runtime._release_atlas = lambda: None

    class Prefixes:
        def save(self, state: object) -> object:
            del state
            return object()

    runtime.prefixes = Prefixes()
    runtime._persist_prefix(release_atlas=False)
    assert seen["lengths"] == {0: 415, 48: 415}
    assert seen["logits"] is None
    assert runtime._prefix_dirty is False


def test_verify_forward_keeps_the_clean_draft_cache() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = list(range(10))
    runtime._mtp_layer = 48
    runtime.kv_len = {0: 10, 48: 12}
    runtime._kv_snap = {
        0: (None, None, None, None, None, 10),
        48: (None, None, None, None, None, 10),
    }
    runtime._mtp_kv = (None, None, None, None, None, 10)
    called: list[str] = []
    runtime._restore_recurrent = lambda skip=None: called.append("restore")
    runtime._mtp_restore_kv = lambda: called.append("mtp")
    runtime._rewind_uncommitted(keep_mtp=2)
    assert called == []
    assert runtime.kv_len == {0: 10, 48: 12}


def test_prefix_flush_logs_a_cache_mismatch_and_returns() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [1, 2]
    runtime.kv_len = {0: 6}
    runtime._kv_snap = None
    runtime._mtp_kv = None
    runtime.gpu = type("Gpu", (), {"flush": lambda self: None})()

    def state() -> object:
        message = "attention cache length 6 does not match 2 tokens"
        raise RuntimeError(message)

    runtime._prefix_state = state
    released: list[str] = []
    runtime._release_atlas = lambda: released.append("release")
    runtime._flush_prefix()
    assert released == []


def test_persist_writes_the_live_prefix_once() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [1, 2]
    runtime._prefix_dirty = True
    runtime.gpu = type("Gpu", (), {"flush": staticmethod(lambda: None)})()
    runtime._prefix_state = lambda: object()
    runtime._release_atlas = lambda: None
    writes: list[int] = []

    class Prefixes:
        def save(self, state: object) -> object:
            writes.append(len(runtime.tokens))
            return object()

    runtime.prefixes = Prefixes()
    runtime._persist_prefix()
    runtime._persist_prefix()
    runtime.tokens = [1, 2, 3]
    runtime._prefix_dirty = True
    runtime._persist_prefix()
    assert writes == [2, 3]


def test_close_writes_the_live_prefix_before_the_device_is_released() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime._closed = False
    order: list[str] = []
    runtime._persist_prefix = lambda: order.append("persist")
    runtime.gpu = type("Gpu", (), {"close": lambda self: order.append("close")})()
    runtime.files = None
    runtime.store = None
    runtime.close()
    runtime.close()
    assert order == ["persist", "close"]


def test_cancel_restores_the_snapshot_then_writes_the_prefix() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [7, 8]
    runtime._streaming_prefill = False
    runtime.atlas = None
    runtime._kv_snap = {0: object()}
    runtime._forward_open = True
    order: list[str] = []
    runtime.gpu = type("Gpu", (), {"flush": lambda self: order.append("flush")})()
    runtime._restore_recurrent = lambda: order.append("restore")
    runtime.reset = lambda: order.append("reset")
    runtime._persist_prefix = lambda: order.append("persist")
    runtime._cancel_forward()
    assert runtime.tokens == [7, 8]
    assert runtime._kv_snap is None
    assert runtime._forward_open is False
    assert order == ["flush", "restore", "persist"]


def test_shutdown_restores_an_open_forward_before_the_prefix_write() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [4, 5]
    runtime._forward_open = True
    runtime._kv_snap = {0: 1}
    runtime._prefix_dirty = True
    order: list[str] = []
    runtime.gpu = type("Gpu", (), {"flush": lambda self: order.append("flush")})()
    runtime._restore_recurrent = lambda: order.append("restore")
    runtime._prefix_state = lambda: object()
    runtime._release_atlas = lambda: order.append("release")

    class Prefixes:
        def save(self, state: object) -> object:
            order.append("save")
            return object()

    runtime.prefixes = Prefixes()
    runtime._persist_prefix()
    assert runtime._forward_open is False
    assert runtime._kv_snap is None
    assert order == ["flush", "restore", "flush", "save"]


def test_flush_continues_when_the_volume_refuses_the_prefix() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [1]
    runtime.gpu = type("Gpu", (), {"flush": staticmethod(lambda: None)})()
    runtime._prefix_state = lambda: object()
    runtime._release_atlas = lambda: None

    class Prefixes:
        def save(self, state: object) -> None:
            raise OSError(28, "preallocate failed for 4096 bytes")

    runtime.prefixes = Prefixes()
    runtime._flush_prefix()


def test_tok_rate_line() -> None:
    assert pipeline._tok_s("prefill", 0, 1.0) == "prefill 0 tok/s"
    assert pipeline._tok_s("decode", 4, 0.0) == "decode 0 tok/s"
    assert pipeline._tok_s("prefill", 58, 12.5) == "prefill 4.64 tok/s (58 tok in 12.50s)"


def test_transcript_appends_prompt_and_output(tmp_path) -> None:
    path = tmp_path / "turns.log"
    pipeline._append_transcript(path, "2+2=", "4")
    pipeline._append_transcript(path, "next", "more")
    text = path.read_text(encoding="utf-8")
    assert text == "prompt:\n2+2=\n\noutput:\n4\n\nprompt:\nnext\n\noutput:\nmore\n"
    pipeline._append_transcript(None, "skipped", "skipped")


def test_preallocate_covers_the_file_in_one_extent(tmp_path) -> None:
    import qwen38_flash_next.allocate as allocate

    path = tmp_path / "blob"
    nbytes = 8 * pipeline.PAGE
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        allocate.preallocate(fd, nbytes)
        os.pwrite(fd, b"\xab" * nbytes, 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    st = path.stat()
    assert st.st_size == nbytes
    assert st.st_blocks * 512 >= nbytes
    contig, device = allocate.contiguous_bytes(path, 0, nbytes)
    assert device >= 0
    assert contig >= nbytes


def test_preallocate_splits_a_refused_piece(tmp_path, monkeypatch) -> None:
    import qwen38_flash_next.allocate as allocate

    real = allocate._alloc_piece

    def refuse_large(fd: int, nbytes: int, *, contig: bool) -> int | None:
        if nbytes > 4 * pipeline.PAGE:
            return 28
        return real(fd, nbytes, contig=contig)

    monkeypatch.setattr(allocate, "_alloc_piece", refuse_large)
    monkeypatch.setattr(allocate, "_CHUNK", 8 * pipeline.PAGE)
    monkeypatch.setattr(allocate, "_FALLBACK", 4 * pipeline.PAGE)
    path = tmp_path / "slices"
    nbytes = 8 * pipeline.PAGE
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        allocate.preallocate(fd, nbytes)
    finally:
        os.close(fd)
    assert path.stat().st_size == nbytes
    for offset in (0, 4 * pipeline.PAGE):
        contig, device = allocate.contiguous_bytes(path, offset, 4 * pipeline.PAGE)
        assert device >= 0
        assert contig >= 4 * pipeline.PAGE


def test_preallocate_reports_the_errno(tmp_path, monkeypatch) -> None:
    import qwen38_flash_next.allocate as allocate

    def refuse(fd: int, nbytes: int, *, contig: bool) -> int | None:
        return 28

    monkeypatch.setattr(allocate, "_alloc_piece", refuse)
    path = tmp_path / "refused"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        with pytest.raises(OSError) as raised:
            allocate.preallocate(fd, 4 * pipeline.PAGE)
    finally:
        os.close(fd)
    assert raised.value.errno == 28


def test_preallocate_claims_each_chunk(tmp_path, monkeypatch) -> None:
    import qwen38_flash_next.allocate as allocate

    monkeypatch.setattr(allocate, "_CHUNK", 4 * pipeline.PAGE)
    path = tmp_path / "chunks"
    nbytes = 8 * pipeline.PAGE
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        allocate.preallocate(fd, nbytes)
    finally:
        os.close(fd)
    for offset in (0, 4 * pipeline.PAGE):
        contig, device = allocate.contiguous_bytes(path, offset, 4 * pipeline.PAGE)
        assert device >= 0
        assert contig >= 4 * pipeline.PAGE


def test_reasoning_effort_none_uses_the_template_default() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._reasoning_effort({}) is None
    assert commands._reasoning_effort({"reasoning_effort": None}) is None
    assert commands._reasoning_effort({"reasoning_effort": "none"}) is None
    assert commands._reasoning_effort({"reasoning_effort": "None"}) is None
    assert commands._reasoning_effort({"reasoning_effort": "low"}) == "low"
    assert commands._reasoning_effort({"reasoning_effort": "xhigh"}) == "xhigh"


def test_omitted_enable_thinking_leaves_the_block_open() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._enable_thinking({}) is True
    assert commands._enable_thinking({"enable_thinking": True}) is True
    assert commands._enable_thinking({"enable_thinking": False}) is False
    assert commands._enable_thinking({"enable_thinking": None}) is True
    assert commands._enable_thinking({"chat_template_kwargs": {"enable_thinking": False}}) is False
    assert (
        commands._enable_thinking(
            {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True}}
        )
        is False
    )


def test_penalty_window_defaults_to_the_last_twenty_tokens() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._penalty_window({}) == 20
    assert commands._penalty_window({"penalty_window": 64}) == 64
    assert commands._penalty_window({"penalty_window": 0}) == 0


def test_record_name_orders_wider_quants_first(tmp_path) -> None:
    from qwen38_flash_next.formats import record_name

    source = tmp_path / "bf16"
    assert record_name(source, [8, 4]) == f"{source.name}-madmetal-fp8-fp4"


def test_bare_id_uses_this_pipeline_card() -> None:
    from qwen38_flash_next.sampling import mode_card

    thinking, card = mode_card(None, None)
    assert thinking is True
    assert card is not None
    assert card["presence_penalty"] == 0.0
    closed, instruct = mode_card(None, False)
    assert closed is False
    assert instruct is not None
    assert instruct["presence_penalty"] == 1.5


def test_omitted_sampling_follows_the_model_card() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._sampling({}, thinking=True) == (1.0, 0.95, 20)
    assert commands._sampling({}, thinking=False) == (0.7, 0.8, 20)
    assert commands._sampling(
        {"temperature": None, "top_p": None, "top_k": None}, thinking=False
    ) == (0.7, 0.8, 20)
    assert commands._sampling({"temperature": 0}, thinking=True) == (0.0, 0.95, 20)
    assert commands._sampling({"temperature": 0, "top_p": 1, "top_k": 0}, thinking=False) == (
        0.0,
        1.0,
        0,
    )


def test_omitted_presence_follows_the_model_card() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._presence({}, thinking=True) == 0.0
    assert commands._presence({}, thinking=False) == 1.5
    assert commands._presence({"presence_penalty": None}, thinking=False) == 1.5
    assert commands._presence({"presence_penalty": 0}, thinking=False) == 0.0
    assert commands._presence({"presence_penalty": 1.5}, thinking=True) == 1.5


def test_grove_file_stores_the_greedy_completion(tmp_path) -> None:
    import qwen38_flash_next.grove as grove

    class Runtime:
        _hit_limit = False

        def encode_chat(
            self, messages: list[dict[str, str]], tools: object, thinking: bool
        ) -> list[int]:
            assert messages[0]["content"] == grove.GROVE_PROMPT
            assert tools is None
            assert thinking is False
            return [1]

        def generate(
            self,
            prompt: list[int],
            max_tokens: int | None,
            temperature: float,
            top_p: float,
            top_k: int,
        ) -> list[int]:
            assert prompt == [1]
            assert max_tokens == grove.GROVE_MAX_TOKENS
            assert temperature == 0.0
            assert top_p == 1.0
            assert top_k == 0
            return [2]

        def decode(self, token_ids: list[int]) -> str:
            assert token_ids == [2]
            return "six trees\n"

    grove.write_grove(tmp_path, replace=True, runtime=Runtime())
    stored = grove.load_grove(tmp_path)
    assert stored["output"] == "six trees"
    assert stored["thinking"] is False

    class Unused:
        def encode_chat(self, messages: object, tools: object, thinking: bool) -> list[int]:
            raise AssertionError("existing grove file")

    grove.write_grove(tmp_path, runtime=Unused())
    assert grove.load_grove(tmp_path)["output"] == "six trees"


def test_stream_detokenize_runs_off_the_step_thread() -> None:
    import threading

    import qwen38_flash_next.commands as commands

    step = threading.get_ident()

    class Runtime:
        def decode(self, token_ids: list[int]) -> str:
            assert threading.get_ident() != step
            return "".join(chr(97 + token) for token in token_ids)

    pieces: list[str] = []
    text = commands._drain_stream(Runtime(), iter((0, 1, 2)), pieces.append)
    assert pieces == ["a", "b", "c"]
    assert text == "abc"


def test_stream_holds_a_partial_character_until_it_resolves() -> None:
    import qwen38_flash_next.commands as commands

    class Runtime:
        def decode(self, token_ids: list[int]) -> str:
            if len(token_ids) < 3:
                return "\ufffd"
            return "你"

    pieces: list[str] = []
    text = commands._drain_stream(Runtime(), iter((0, 1, 2)), pieces.append)
    assert pieces == ["你"]
    assert text == "你"
    assert "\ufffd" not in text


def test_stream_flushes_a_finished_replacement_character() -> None:
    import qwen38_flash_next.commands as commands

    class Runtime:
        def decode(self, token_ids: list[int]) -> str:
            return "\ufffd"

    pieces: list[str] = []
    text = commands._drain_stream(Runtime(), iter((0,)), pieces.append)
    assert pieces == ["\ufffd"]
    assert text == "\ufffd"


def test_peer_closed_after_the_client_disconnects() -> None:
    import socket

    from madmetal.listener import _peer_closed

    server, client = socket.socketpair()
    try:
        assert _peer_closed(server) is False
        client.close()
        assert _peer_closed(server) is True
    finally:
        server.close()


def test_closed_client_stops_the_forward() -> None:
    from qwen38_flash_next.runtime import Qwen38Runtime, RequestClosed

    runtime = object.__new__(Qwen38Runtime)
    runtime._stop = None
    runtime._check_stop()
    runtime._stop = lambda: False
    runtime._check_stop()
    runtime._stop = lambda: True
    try:
        runtime._check_stop()
    except RequestClosed:
        return
    raise AssertionError("closed client kept the forward running")


def test_stream_write_failure_ends_the_request() -> None:
    import qwen38_flash_next.commands as commands
    from qwen38_flash_next.runtime import RequestClosed

    class Runtime:
        def decode(self, token_ids: list[int]) -> str:
            return "x" * len(token_ids)

    def emit(_piece: str) -> None:
        raise RequestClosed()

    try:
        commands._drain_stream(Runtime(), iter((0, 1, 2)), emit)
    except RequestClosed:
        return
    raise AssertionError("failed stream write kept the request open")


def test_record_close_releases_the_files(tmp_path) -> None:
    from qwen38_flash_next.record import QuantRecord

    record = object.__new__(QuantRecord)
    path = tmp_path / "tier.bin"
    path.write_bytes(b"abcd")
    record._closed = False
    record.hot_fds = [os.open(path, os.O_RDONLY)]
    record.cold_fds = []
    record.super_fds = []
    record.ngram_fd = os.open(path, os.O_RDONLY)
    record.expert_scratch = mmap.mmap(-1, 4096)
    record.page_scratch = mmap.mmap(-1, 4096)
    record.pages = {0: b"abcd"}
    record.hot_io = []
    record.cold_io = []
    record.resident_io = None
    record.ngram_io = None
    held = record.hot_fds[0]
    record.close()
    record.close()
    try:
        os.fstat(held)
    except OSError:
        return
    raise AssertionError("record close left the file open")


def test_signal_stops_the_runtime_once() -> None:
    import signal

    import qwen38_flash_next.commands as commands
    from qwen38_flash_next.runtime import Qwen38Runtime, RequestClosed

    runtime = object.__new__(Qwen38Runtime)
    runtime._stop = None
    runtime.halt()
    try:
        runtime._check_stop()
    except RequestClosed:
        pass
    else:
        raise AssertionError("halt did not stop the forward")

    shutdown = commands._Shutdown()
    shutdown.runtime = runtime
    try:
        shutdown.handle(signal.SIGINT, None)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("signal did not interrupt the process")
    assert shutdown.signaled
    shutdown.closing = True
    shutdown.handle(signal.SIGTERM, None)


def test_partial_runtime_close_is_idempotent() -> None:
    from qwen38_flash_next.runtime import Qwen38Runtime

    runtime = object.__new__(Qwen38Runtime)
    runtime._closed = False
    runtime.gpu = None
    runtime.files = None
    runtime.store = None
    runtime.close()
    runtime.close()


def test_session_close_releases_the_device() -> None:
    from qwen38_flash_next.session import MetalSession

    session = MetalSession()
    session.close()
    session.close()
    assert session.device is None


def test_release_closes_the_listening_socket() -> None:
    import socket
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import qwen38_flash_next.commands as commands

    class Quiet(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Quiet)
    port = int(server.server_address[1])
    shutdown = commands._Shutdown()
    shutdown.signaled = True
    commands._release(None, shutdown, server)
    probe = socket.socket()
    try:
        probe.connect(("127.0.0.1", port))
    except OSError:
        return
    finally:
        probe.close()
    raise AssertionError("listening socket stayed open")


def test_a_closed_client_does_not_print_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    from madmetal.listener import _SingleRequestServer

    server = _SingleRequestServer(("127.0.0.1", 0), type("Quiet", (), {}))
    try:
        try:
            raise BrokenPipeError(32, "Broken pipe")
        except BrokenPipeError:
            server.handle_error(None, ("127.0.0.1", 1))
        quiet = capsys.readouterr().err
        assert "Traceback" not in quiet
        assert "Exception occurred" not in quiet
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            server.handle_error(None, ("127.0.0.1", 1))
        logged = capsys.readouterr().err
        assert "Traceback" in logged
        assert "boom" in logged
    finally:
        server.server_close()


def test_omitted_output_limit_is_uncapped() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._request_limit({}) is None
    assert commands._request_limit({"max_tokens": None}) is None
    assert commands._request_limit({"max_tokens": 0}) == 0
    assert commands._request_limit({"max_tokens": 32}) == 32
    assert commands._request_limit({"max_completion_tokens": 16}) == 16


def test_complete_thinking_flag_enables_thinking(monkeypatch, tmp_path) -> None:
    import qwen38_flash_next.commands as commands

    seen: list[bool] = []

    class Tokenizer:
        def encode(self, text: str) -> object:
            return type("Encoded", (), {"ids": [4]})()

    class Runtime:
        def __init__(self, path: object, **_kwargs: object) -> None:
            self.tokenizer = Tokenizer()

        def encode_chat(self, messages: object, tools: object, enable_thinking: bool) -> list[int]:
            seen.append(enable_thinking)
            return [1]

        def align(self, prompt: list[int]) -> int:
            return 0

        def close(self) -> None:
            return None

        def forward(self, prompt: list[int]) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

        def _prompt_logits(self, prompt: list[int]) -> np.ndarray:
            return self.forward(prompt)

        def decode(self, ids: list[int]) -> str:
            return ""

        def _new_tokens(self, *args: object) -> list[int]:
            return []

        def _log_decode(self) -> None:
            return None

    monkeypatch.setattr(commands, "Qwen38Runtime", Runtime)
    record = tmp_path / "record"
    thinking = ["complete", "--prompt", "2+2", "--thinking", "--record", str(record)]
    plain = ["complete", "--prompt", "2+2", "--record", str(record)]
    assert pipeline.main(thinking) == 0
    assert pipeline.main(plain) == 0
    assert seen == [True, False]


def test_super_image_is_page_aligned() -> None:
    assert pipeline.super_live_bytes() == 5_068_800
    assert pipeline.super_slot_bytes() == 5_079_040
    assert pipeline.super_slot_bytes() % pipeline.PAGE == 0
    planes = pipeline.expert_planes(8)
    assert planes.gate_inner == 2560
    assert planes.down_inner == 640
    gate = np.zeros((1280, 2560), np.float32)
    down = np.zeros((2560, 640), np.float32)
    image = pipeline.quantize_int8_image(gate)[0] + pipeline.quantize_int8_image(down)[0]
    assert len(image) == pipeline.super_live_bytes()


def test_quant_map_names_the_super_pin(tmp_path) -> None:
    path = tmp_path / "quant_map.json"
    path.write_text(
        json.dumps(
            {
                "experts": [[4] * 512 for _ in range(48)],
                "super": [
                    {"layer": 2, "expert": 322},
                    {"layer": 4, "expert": 108},
                    {"layer": 4, "expert": 373},
                    {"layer": 6, "expert": 223},
                    {"layer": 10, "expert": 107},
                ],
            }
        )
    )
    _grid, rows = pipeline.load_quant_map(path, 48, 512)
    assert rows[2] == [322]
    assert rows[4] == [108, 373]
    assert rows[6] == [223]
    assert rows[10] == [107]
    assert rows[0] == []


def test_int8_kernels_match_dequant() -> None:
    from qwen38_flash_next.session import MetalSession

    rng = np.random.default_rng(1)
    cols, inner = 32, 32
    weight = rng.normal(size=(cols, inner)).astype(np.float32)
    image, stored = pipeline.quantize_int8_image(weight, align_k=False)
    assert stored == inner
    data_n, _scale_n = pipeline.int8_plane_sizes(cols, inner)
    restored = pipeline.dequant_int8(image[:data_n], image[data_n:], cols, inner)
    session = MetalSession()
    session.begin_scratch(4096)
    vector = rng.normal(size=(inner,)).astype(np.float32)
    resident = session.upload_resident(image)
    result = session.gemv_int8(
        session.dev_from(vector), resident, cols, inner, 0, data_n, inner, half=False
    )
    assert np.allclose(session.read_f32(result), restored @ vector, rtol=1e-3, atol=1e-3)


class _FixedUniform:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def test_parallel_sample_draws_inside_the_top_k() -> None:
    from qwen38_flash_next.session import MetalSession

    logits = np.full(64, -20.0, np.float32)
    logits[50] = 5.0
    logits[1] = 4.0
    logits[15] = 1.0
    logits[40] = 0.0
    session = MetalSession()
    values = session.dev_from(logits)
    assert session.sample(values, 1.0, 1.0, 4, _FixedUniform(0.0)) == 1
    assert session.sample(values, 1.0, 1.0, 4, _FixedUniform(0.27)) == 15
    assert session.sample(values, 1.0, 1.0, 4, _FixedUniform(0.99)) == 50
    assert session.sample(values, 0.0, 1.0, 4, _FixedUniform(0.99)) == 50


def test_cold_image_is_page_aligned() -> None:
    assert pipeline.cold_live_bytes() == 1_658_880
    assert pipeline.cold_slot_bytes() == 1_671_168
    assert pipeline.cold_slot_bytes() % pipeline.PAGE == 0
    assert (128 * pipeline.cold_slot_bytes()) % pipeline.PREFERRED_IO == 0
    planes = pipeline.expert_planes(2)
    assert planes.down_inner == 1024
    assert planes.gate_inner == 2560
    gate = np.zeros((1280, 2560), np.float32)
    down = np.zeros((2560, 640), np.float32)
    image = pipeline.quantize_int2_image(gate)[0] + pipeline.quantize_int2_image(down)[0]
    assert len(image) == pipeline.cold_live_bytes()


def test_int2_block_matches_quantize_block() -> None:
    values = [0.2, -0.3, 1.5, -1.5, 0.0, 0.05, -0.05, 0.4] + [0.0] * 24
    codes, scale = pipeline.quantize_block(values, 2)
    image, stored = pipeline.quantize_int2_image(
        np.asarray(values, dtype=np.float32).reshape(1, 32), align_k=False
    )
    assert stored == 32
    data_n, _scale_n = pipeline.int2_plane_sizes(1, 32)
    got = [
        pipeline.sign_extend(code, 2)
        for code in pipeline.unpack_int2(image[:data_n], low_bits_first=True, count=32)
    ]
    assert got == codes
    assert image[data_n] == scale


def test_group_tier_skips_the_other_bank() -> None:
    ids = np.array([[0, 1, 2], [2, 0, 3]], dtype=np.uint32)
    experts, _offsets, tokens, inverse = pipeline.group_tier(ids, 2, 3, {0, 2})
    assert experts.tolist() == [0, 2]
    assert int(inverse[1]) == 0xFFFFFFFF
    assert int(inverse[5]) == 0xFFFFFFFF
    assert int(tokens.size) == 4


def test_quant_map_lists_every_expert(tmp_path) -> None:
    path = tmp_path / "quant_map.json"
    path.write_text(
        json.dumps({"experts": [[4, 4, 4, 4], [4, 4, 8, 4]], "super": [{"layer": 1, "expert": 2}]})
    )
    grid, pins = pipeline.load_quant_map(path, 2, 4)
    assert grid == [[4, 4, 4, 4], [4, 4, 8, 4]]
    assert pins == [[], [2]]
    path.write_text(json.dumps({"experts": [[8, 2, 4]], "super": []}))
    grid, pins = pipeline.load_quant_map(path, 1, 3)
    assert grid == [[8, 2, 4]]
    assert pins == [[]]


def test_tier_file_and_resume_set(tmp_path) -> None:
    slot = pipeline.PAGE
    hot = tmp_path / "00-hot.bin"
    cold = tmp_path / "00-cold.bin"
    pipeline.write_tier_file(hot, [b"\x11" * 32, b"\x22" * 32], slot)
    pipeline.write_tier_file(cold, [b"\x33" * 16], slot)
    assert hot.stat().st_size == 2 * slot
    blob = hot.read_bytes()
    assert blob[:32] == b"\x11" * 32
    assert blob[slot : slot + 32] == b"\x22" * 32
    assert cold.read_bytes()[:16] == b"\x33" * 16
    assert pipeline.pending_indices([0], 2) == [1]


def test_ngram_tensors_do_not_overlap(tmp_path) -> None:
    slot = pipeline.NGRAM_SLOT
    path = tmp_path / "ngram.bin"
    first = pipeline.ngram_byte_offset(0, 0)
    second = pipeline.ngram_byte_offset(0, 10)
    assert second - first == 10 * slot
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.pwrite(fd, b"\xab" * slot, first)
        os.pwrite(fd, b"\xcd" * slot, second)
        os.fsync(fd)
    finally:
        os.close(fd)
    blob = path.read_bytes()
    assert blob[first : first + slot] == b"\xab" * slot
    assert blob[second : second + slot] == b"\xcd" * slot
    assert pipeline.pending_indices([0, 2], 4) == [1, 3]


def test_ngram_groups_share_a_file() -> None:
    def key(shard: int) -> str:
        return (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{shard}.weight"
        )

    groups = pipeline.ngram_file_groups(
        {key(0): "a.bin", key(1): "a.bin", key(2): "b.bin"}, [2, 0, 1]
    )
    assert groups == [[2], [0, 1]]


def test_after_atlas_load_starts_when_resident_stops_growing() -> None:
    from qwen38_flash_next.runtime import after_atlas_load

    assert after_atlas_load([]) == 0
    assert after_atlas_load([10]) == 1
    assert after_atlas_load([10, 40, 80, 80, 80]) == 3
    assert after_atlas_load([10, 40, 30, 90]) == 2


def test_prefill_table_places_cold_after_the_hot_file() -> None:
    from qwen38_flash_next.record import prefill_image_table

    hot = [5, 1]
    supers = [9]
    slot = 16384
    hot_bytes, cold_bytes, table = prefill_image_table(hot, supers, 12, slot, slot)
    assert hot_bytes == 2 * slot
    assert cold_bytes == 9 * slot
    assert int(table[5]) == 0
    assert int(table[1]) == slot
    assert int(table[9]) == -1
    assert int(table[0]) == hot_bytes
    assert int(table[2]) == hot_bytes + slot
    try:
        prefill_image_table([9], [9], 12, slot, slot)
    except RuntimeError as exc:
        assert "super" in str(exc)
    else:
        raise AssertionError("a super id in the hot file was accepted")


def _decode_runtime() -> tuple[object, Console, io.StringIO]:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    stream = io.StringIO()
    console = Console(
        "pretty",
        pipeline="qwen38-flash-next",
        stream=stream,
        width=160,
        color=0,
        start_thread=False,
    )
    runtime.console = console
    runtime._decode_progress = None
    runtime._decode_reported = True
    runtime._request_id = 4
    runtime._decode_tokens = 129
    runtime._decode_seconds = 40.0
    runtime._output_ids = [1] * 129
    return runtime, console, stream


def test_a_closed_stream_ends_the_decode_row() -> None:
    runtime, console, stream = _decode_runtime()
    runtime.tokens = [1]
    runtime._mtp_on = False
    runtime._stop = lambda: True
    with pytest.raises(RequestClosed):
        list(runtime._new_tokens(None, None, 0.0, 1.0, 0))
    assert console._handles == []
    assert "decode 0 tok/s" in stream.getvalue()


def test_the_next_request_does_not_keep_the_previous_decode_row() -> None:
    runtime, console, stream = _decode_runtime()
    runtime._begin_decode(None)
    runtime._request_id = 6
    runtime._begin_decode(65536)
    assert [item.request for item in console._handles] == [6]
    assert console._handles[0].total == 0
    assert "decode 3.23 tok/s (129 tok in 40.00s)" in stream.getvalue()
    runtime._log_decode()
    runtime._close_decode_row()
    runtime.console.begin("prefill", total=8, unit="tok", detail="req #6", request=6)
    assert [item.phase for item in console._handles] == ["prefill"]


def test_decode_always_uses_the_mtp_cycle() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = list(range(3000))
    runtime._mtp_on = True
    runtime.stop_ids = {9}
    runtime.required_function = None
    runtime._decode_reported = True
    runtime._begin_drafts = lambda: setattr(runtime, "_mtp_drafts", False)
    runtime._close_decode_row = lambda: None
    runtime._begin_decode = lambda _max_tokens: None
    runtime._check_stop = lambda: None
    runtime._log_decode = lambda: None
    picks = iter((7, 9))
    runtime._pick = lambda *_args: next(picks)
    seen: list[tuple[int, bool]] = []

    def cycle(
        first: int,
        produced: list[int],
        room: int,
        temperature: float,
        top_p: float,
        top_k: int,
        *,
        echo: bool,
    ) -> list[int]:
        del produced, room, temperature, top_p, top_k
        seen.append((first, echo))
        runtime._mtp_stopped = False
        runtime.last_logits = object()
        return [first] if echo else []

    runtime._mtp_cycle = cycle
    produced = list(runtime._new_tokens(object(), None, 0.0, 1.0, 0))
    assert produced == [7]
    assert seen == [(7, True)]
    assert runtime._mtp_drafts is False


def test_prefetch_stops_when_the_atlas_cannot_hold_the_layer() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime._predicted = {0: [1], 1: [2]}
    runtime._prefetch_events = {}

    class Atlas:
        def stage(self, layer: int, ids: list[int], protect: set[tuple[int, int]]) -> int:
            del ids, protect
            if layer == 1:
                message = "resident expert budget cannot hold the experts for this layer"
                raise RuntimeError(message)
            return 3

    runtime.atlas = Atlas()
    runtime._prefetch_decode()
    assert runtime._prefetch_protect == set()
    assert runtime._prefetch_events[0] == 3


def test_drafts_start_whenever_the_mtp_layer_is_loaded() -> None:
    assert mtp_starts(58, loaded=True) is True
    assert mtp_starts(2048, loaded=True) is True
    assert mtp_starts(2049, loaded=True) is True
    assert mtp_starts(28776, loaded=True) is True
    assert mtp_starts(16, loaded=False) is False


def test_drafts_hold_until_the_recent_window_falls_behind() -> None:
    assert mtp_holds([12.0, 12.0, 12.0, 12.0]) is True
    assert mtp_holds([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) is True
    assert mtp_holds([12.0, 12.0, 12.0, 12.0, 4.0, 4.0, 4.0, 4.0]) is False


def test_a_slow_window_stops_drafts_for_the_rest_of_the_request() -> None:
    runtime = pipeline.Qwen38Runtime.__new__(pipeline.Qwen38Runtime)
    runtime.tokens = [1, 2, 3]
    runtime._mtp_on = True
    runtime._decode_progress = None
    runtime._begin_drafts()
    assert runtime._mtp_drafts is True
    for rate in (12.0, 12.0, 12.0, 12.0, 4.0, 4.0, 4.0, 4.0):
        runtime._note_draft_rate(rate, 1.0)
    assert runtime._mtp_drafts is False
    runtime.tokens = list(range(4096))
    runtime._begin_drafts()
    assert runtime._mtp_drafts is True


def test_version5_place_keeps_the_pin_out_of_the_hot_file() -> None:
    placed = version5_place([0], [2], 4)
    assert placed == [("hot", 0), ("cold", 0), ("super", 0), ("cold", 1)]


def test_quant_layout_reads_an_expert_at_its_id(tmp_path) -> None:
    slot = pipeline.PAGE
    live = 16
    experts = tmp_path / "experts"
    experts.mkdir()
    blob = bytearray(slot * 2)
    blob[:4] = b"aaaa"
    blob[slot : slot + 4] = b"bbbb"
    (experts / quant_file(0, 4)).write_bytes(blob)
    (tmp_path / "ngram.bin").write_bytes(b"\x00")
    body = {
        "version": QUANT_LAYOUT_VERSION,
        "complete": True,
        "layers": 1,
        "experts": 2,
        "expert_quant": 4,
        "expert_slot": slot,
        "expert_live": live,
        "super": [[1]],
        "super_bits": 4,
        "ngram_rows": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(body))
    record = QuantRecord(tmp_path)
    try:
        assert record.tier_row(0, 0) == (1, 0)
        assert record.tier_row(0, 1) == (1, 1)
        assert record.expert_image(0, 0)[:4] == b"aaaa"
        assert record.expert_image(0, 1)[:4] == b"bbbb"
        assert record.hot_set(0) == set()
        assert record.cold_set(0) == {0, 1}
        assert 1 in record.super_set(0)
    finally:
        record.close()


def test_shipped_quant_map_is_fp4() -> None:
    grid, pins = pipeline.load_quant_map(DEFAULT_QUANT_MAP, 48, 512)
    assert len(grid) == 48
    assert len(grid[0]) == 512
    assert {bits for row in grid for bits in row} == {4}
    assert pins[2] == [322]
    assert pins[4] == [108, 373]
    assert pins[6] == [223]
    assert pins[10] == [107]
    assert sum(len(row) for row in pins) == 5


def test_two_widths_read_each_quant_file(tmp_path) -> None:
    root = tmp_path / "record"
    experts = root / "experts"
    experts.mkdir(parents=True)
    slot = pipeline.PAGE
    live = 16
    (experts / quant_file(0, 8)).write_bytes(b"hhhh" + b"\x00" * (slot - 4))
    (experts / quant_file(0, 4)).write_bytes(b"cccc" + b"\x00" * (slot - 4))
    manifest = {
        "version": QUANT_LAYOUT_VERSION,
        "complete": True,
        "layers": 1,
        "experts": 2,
        "expert_quant": [[8, 4]],
        "quants": {
            "8": {"format": "fp8_e4m3", "live": live, "slot": slot},
            "4": {"format": "fp4_e2m1", "live": live, "slot": slot},
        },
        "super": [[]],
        "super_bits": 4,
        "super_live": live,
        "super_slot": slot,
        "mtp_experts": 0,
        "mtp_slot": slot,
        "ngram_rows": 1,
        "ngram_format": "int4",
        "ngram_live": 16,
        "ngram_slot": slot,
        "embed_key": "model.language_model.embed_tokens.weight",
        "head_key": "lm_head.weight",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "ngram.bin").write_bytes(b"\x00" * slot)
    record = QuantRecord(root)
    try:
        assert record.tier_row(0, 0) == (0, 0)
        assert record.tier_row(0, 1) == (1, 0)
        assert record.expert_image(0, 0)[:4] == b"hhhh"
        assert record.expert_image(0, 1)[:4] == b"cccc"
        assert record.narrow_set(0) == set()
    finally:
        record.close()


def test_three_widths_put_int2_on_tier_3(tmp_path) -> None:
    root = tmp_path / "record"
    experts = root / "experts"
    experts.mkdir(parents=True)
    slot = pipeline.PAGE
    live = 16
    (experts / quant_file(0, 8)).write_bytes(b"hhhh" + b"\x00" * (slot - 4))
    (experts / quant_file(0, 4)).write_bytes(b"cccc" + b"\x00" * (slot - 4))
    (experts / quant_file(0, 2)).write_bytes(b"nnnn" + b"\x00" * (slot - 4))
    manifest = {
        "version": QUANT_LAYOUT_VERSION,
        "complete": True,
        "layers": 1,
        "experts": 3,
        "expert_quant": [[8, 4, 2]],
        "quants": {
            "8": {"format": "fp8_e4m3", "live": live, "slot": slot},
            "4": {"format": "fp4_e2m1", "live": live, "slot": slot},
            "2": {"format": "int2", "live": live, "slot": slot},
        },
        "super": [[1]],
        "super_bits": 4,
        "super_live": live,
        "super_slot": slot,
        "mtp_experts": 0,
        "mtp_slot": slot,
        "ngram_rows": 1,
        "ngram_format": "int4",
        "ngram_live": 16,
        "ngram_slot": slot,
        "embed_key": "model.language_model.embed_tokens.weight",
        "head_key": "lm_head.weight",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "ngram.bin").write_bytes(b"\x00" * slot)
    record = QuantRecord(root)
    try:
        assert record.tier_row(0, 0) == (0, 0)
        assert record.tier_row(0, 1) == (2, 0)
        assert record.tier_row(0, 2) == (3, 0)
        assert record.hot_set(0) == {0}
        assert record.cold_set(0) == set()
        assert record.narrow_set(0) == {2}
        assert record.hot_bits == 8
        assert record.cold_bits == 4
        assert record.narrow_bits == 2
        assert record.stride(3) == slot
        assert record.expert_image(0, 0)[:4] == b"hhhh"
        assert record.expert_image(0, 1)[:4] == b"cccc"
        assert record.expert_image(0, 2)[:4] == b"nnnn"
    finally:
        record.close()
