import mmap
import os

import numpy as np

from test_packing import pipeline


def test_quantize_matrix_matches_block() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(3, 32)).astype(np.float32)
    matrix[1] = 0
    matrix[2, 0] = 7
    matrix[2, 1] = 0.5
    matrix[2, 2] = -1.5
    packed, scales = pipeline.quantize_matrix(matrix)
    for row in range(matrix.shape[0]):
        codes, scale = pipeline.quantize_block(matrix[row].tolist(), 4)
        assert scales[row] == scale
        start = row * 16
        got = pipeline.unpack_int4(packed[start : start + 16], low_nibble_first=True)
        signed = [pipeline.sign_extend(code, 4) for code in got]
        assert signed == codes


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
        progress, done=True, inflight_key=0, path=path, experts=2
    )
    assert start == 2


def test_expert_start_resumes_after_the_last_committed_expert(tmp_path) -> None:
    path = tmp_path / "01.bin"
    _full_expert_file(path, 4)
    progress = {"inflight_key": 1, "inflight_expert": 2}
    start = pipeline._expert_start(
        progress, done=False, inflight_key=1, path=path, experts=4
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


def test_tok_rate_line() -> None:
    assert pipeline._tok_s("prefill", 0, 1.0) == "prefill 0 tok"
    assert pipeline._tok_s("decode", 4, 0.0) == "decode 0 tok"
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


def test_omitted_enable_thinking_leaves_the_block_open() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._enable_thinking({}) is True
    assert commands._enable_thinking({"enable_thinking": True}) is True
    assert commands._enable_thinking({"enable_thinking": False}) is False
    assert commands._enable_thinking({"enable_thinking": None}) is True
    assert commands._enable_thinking({"chat_template_kwargs": {"enable_thinking": False}}) is False
    assert commands._enable_thinking({"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": True}}) is False


def test_penalty_window_defaults_to_the_last_twenty_tokens() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._penalty_window({}) == 20
    assert commands._penalty_window({"penalty_window": 64}) == 64
    assert commands._penalty_window({"penalty_window": 0}) == 0


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


def test_omitted_max_tokens_runs_until_stop() -> None:
    import qwen38_flash_next.commands as commands

    assert commands._request_limit({}) is None
    assert commands._request_limit({"max_tokens": None}) is None
    assert commands._request_limit({"max_tokens": 0}) == 0
    assert commands._request_limit({"max_tokens": 32}) == 32
    assert commands._request_limit({"max_completion_tokens": 16}) == 16


def test_complete_thinking_flag_enables_thinking(monkeypatch) -> None:
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

        def forward(self, prompt: list[int]) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

        def decode(self, ids: list[int]) -> str:
            return ""

        def _new_tokens(self, *args: object) -> list[int]:
            return []

        def _log_decode(self) -> None:
            return None

    monkeypatch.setattr(commands, "Qwen38Runtime", Runtime)
    assert pipeline.main(["complete", "--prompt", "2+2", "--thinking"]) == 0
    assert pipeline.main(["complete", "--prompt", "2+2"]) == 0
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


def test_super_ids_leave_the_keep_set(tmp_path) -> None:
    path = tmp_path / "super.json"
    path.write_text(
        '{"super_experts": ['
        '{"layer": 2, "expert": 322},'
        '{"layer": 4, "expert": 108},'
        '{"layer": 4, "expert": 373},'
        '{"layer": 6, "expert": 223},'
        '{"layer": 10, "expert": 107}'
        "]}"
    )
    rows = pipeline.load_super_ids(path, 48, 512)
    assert rows[2] == [322]
    assert rows[4] == [108, 373]
    assert rows[0] == []
    keep = list(range(384))
    hot, cold, pin = pipeline.split_keep(keep, [108, 373], 512)
    assert pin == [108, 373]
    assert len(hot) == 382
    assert 108 not in hot and 373 not in hot
    assert len(cold) == 128
    assert set(cold).isdisjoint(keep)


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


def test_hot_ids_require_the_same_count(tmp_path) -> None:
    path = tmp_path / "hot.json"
    path.write_text('{"0": [0, 1], "1": [2]}')
    try:
        pipeline.load_hot_ids(path, 2, 4)
    except RuntimeError as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("expected hot counts to differ")
    path.write_text('{"0": [0, 1], "1": [1, 2]}')
    assert pipeline.load_hot_ids(path, 2, 4) == [[0, 1], [1, 2]]


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
