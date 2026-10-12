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
    assert live < pipeline.EXPERT_SLOT


def test_ngram_resume_row_rewinds_when_the_file_is_short() -> None:
    assert pipeline.ngram_resume_row(100, 40 * pipeline.NGRAM_SLOT + 10) == 40


def test_ngram_resume_row_keeps_the_checkpoint_when_the_file_is_ahead() -> None:
    assert pipeline.ngram_resume_row(100, 180 * pipeline.NGRAM_SLOT) == 100


def _full_expert_file(path, experts: int) -> None:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, experts * pipeline.EXPERT_SLOT)
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
