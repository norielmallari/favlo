"""CPU checks for the music schedule, the request, and the packed weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ace_step_15_turbo.graph import _boundary_device, generate
from ace_step_15_turbo.hold import ClipHold, ClipPaused
from ace_step_15_turbo.pack import dequant_fp4, dequant_fp8, pack_fp4, pack_fp8, promote_count, promoted_ids
from ace_step_15_turbo.pcm import wav_bytes, wav_info
from ace_step_15_turbo.request import parse_music
from ace_step_15_turbo.runtime import (
    Runtime,
    decoder_bytes,
    handle_music,
    open_runtime,
    step_scratch_bytes,
)
from ace_step_15_turbo.schedule import (
    HOP,
    MAX_SECONDS,
    condition_text,
    latent_frames,
    timestep_schedule,
)
from ace_step_15_turbo.tiles import dependency, plan, tile_frames
from favlo.music import music_dir, store_wav


def test_ten_seconds_is_250_frames() -> None:
    assert latent_frames(10) == 250
    with pytest.raises(ValueError):
        latent_frames(9)


def test_open_runtime_forwards_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(path, console=None, *, owner=None, max_memory=None):
        captured["path"] = path
        captured["owner"] = owner
        captured["max_memory"] = max_memory
        return "runtime"

    monkeypatch.setattr("ace_step_15_turbo.runtime.Runtime", fake)
    assert open_runtime(
        Path("record"),
        kv_cache="e4m3",
        console=None,
        resident_percent=None,
        owner="ace-step-15-turbo:7",
        max_memory="16GiB",
    ) == "runtime"
    assert captured["owner"] == "ace-step-15-turbo:7"
    assert captured["max_memory"] == "16GiB"


def test_a_missing_record_keeps_the_passed_owner(tmp_path: Path) -> None:
    owner = "ace-step-15-turbo:7"
    held: dict[str, str] = {}

    class _Probe(Runtime):
        def __init__(self, path, console=None, *, owner=None, max_memory=None) -> None:
            try:
                super().__init__(path, console, owner=owner, max_memory=max_memory)
            except FileNotFoundError:
                held["owner"] = self._owner
                raise

    with pytest.raises(FileNotFoundError):
        _Probe(tmp_path, owner=owner)
    assert held["owner"] == owner


def test_a_missing_record_mints_an_owner(tmp_path: Path) -> None:
    held: dict[str, str] = {}

    class _Probe(Runtime):
        def __init__(self, path, console=None, *, owner=None, max_memory=None) -> None:
            try:
                super().__init__(path, console, owner=owner, max_memory=max_memory)
            except FileNotFoundError:
                held["owner"] = self._owner
                raise

    with pytest.raises(FileNotFoundError):
        _Probe(tmp_path)
    minted = held["owner"]
    assert minted.startswith("ace-step-15-turbo:")
    assert minted != "ace-step-15-turbo:7"


def test_step_scratch_grows_with_the_duration() -> None:
    assert step_scratch_bytes(10) < step_scratch_bytes(MAX_SECONDS)


def test_omitted_duration_is_30_seconds() -> None:
    fields = parse_music({"prompt": "solo piano"})
    assert fields.seconds == 30


def test_decoder_peak_is_one_tile() -> None:
    frames = latent_frames(30)
    if frames % 2:
        frames += 1
    one = 128 * tile_frames(frames) * HOP * 4
    assert decoder_bytes(30) == one * 3
    assert decoder_bytes(10) == decoder_bytes(180)
    full = 3 * 128 * frames * HOP * 4
    assert decoder_bytes(30) < full


def test_tiles_cover_the_clip_and_its_neighborhood() -> None:
    frames = 250
    covered = 0
    for latent_lo, latent_hi, audio_lo, audio_hi in plan(frames):
        assert 0 <= latent_lo < latent_hi <= frames
        assert audio_lo == covered
        assert audio_hi > audio_lo
        positions = range(audio_lo, audio_hi, 64)
        for pos in (*positions, audio_hi - 1):
            lo, hi = dependency(pos, pos)
            assert latent_lo <= max(0, lo)
            assert latent_hi >= min(frames, hi + 1)
        covered = audio_hi
    assert covered == frames * HOP


def test_dependency_matches_the_convolution_taps() -> None:
    for pos in (0, 1, 100, 1919, 1920, 50_000):
        assert dependency(pos, pos) == _brute_dependency(pos)


def test_a_short_ledger_keeps_the_requested_duration() -> None:
    runtime = Runtime.__new__(Runtime)
    runtime._memory_cap = None
    runtime._room = lambda: step_scratch_bytes(30) - 1
    seconds = 30.0
    with pytest.raises(RuntimeError, match="30 seconds"):
        runtime._reserve_step(seconds)
    assert seconds == 30.0


def test_shift_three_nodes() -> None:
    nodes = timestep_schedule()
    expect = np.array([1, 21 / 22, 9 / 10, 5 / 6, 3 / 4, 9 / 14, 1 / 2, 3 / 10])
    assert nodes.shape == (8,)
    assert np.allclose(nodes, expect)


def test_condition_text_keeps_the_labels() -> None:
    caption, lyric = condition_text("solo piano", "", 10, "en")
    assert caption.startswith("# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\n")
    assert "- duration: 10 seconds\n" in caption
    assert caption.endswith("<|endoftext|>\n")
    assert lyric == "# Languages\nen\n\n# Lyric\n<|endoftext|>"


def test_duration_wins_and_turbo_locks_steps() -> None:
    fields = parse_music({"prompt": "solo piano", "duration": 10, "seconds": 30, "output_format": "wav"})
    assert fields.seconds == 10
    assert fields.output_format == "wav"
    with pytest.raises(ValueError):
        parse_music({"prompt": "solo piano", "steps": 4})
    with pytest.raises(ValueError):
        parse_music({"prompt": "solo piano", "output_format": "wav", "response_format": "mp3"})


def test_fp4_roundtrip_on_the_grid() -> None:
    weight = np.zeros((32, 256), dtype=np.float32)
    weight[0, 0] = 6
    weight[0, 1] = -3
    weight[1, 32] = 0.5
    nibbles, scales, rows, cols = pack_fp4(weight)
    assert (rows, cols) == (32, 256)
    back = dequant_fp4(nibbles, scales, 32, 256)
    assert abs(float(back[0, 0]) - 6) < 1e-5
    assert abs(float(back[0, 1]) + 3) < 1e-5
    assert abs(float(back[1, 32]) - 0.5) < 1e-5


def test_fp8_roundtrip_on_the_grid() -> None:
    weight = np.zeros((32, 256), dtype=np.float32)
    weight[0, 0] = 448
    weight[0, 1] = -256
    weight[1, 32] = 0.5
    codes, scales, rows, cols = pack_fp8(weight)
    assert (rows, cols) == (32, 256)
    back = dequant_fp8(codes, scales, 32, 256)
    assert abs(float(back[0, 0]) - 448) < 1e-3
    assert abs(float(back[0, 1]) + 256) < 1e-3
    assert abs(float(back[1, 32]) - 0.5) < 1e-5


def test_mlp_budget_is_six_layers_and_keeps_the_lower_tie() -> None:
    assert promote_count(20, 32) == 6
    assert promote_count(20, 64) == 12
    assert promoted_ids({0: 0.5, 1: 0.5, 2: 0.1, 3: 0.9}, 2) == [0, 3]


def test_wav_is_ten_seconds_of_stereo() -> None:
    audio = np.zeros((2, 480_000), dtype=np.float32)
    audio[0, 0] = 0.5
    payload = wav_bytes(audio)
    rate, channels, frames = wav_info(payload)
    assert (rate, channels, frames) == (48_000, 2, 480_000)


def test_music_id_rejects_a_dot(tmp_path) -> None:
    with pytest.raises(ValueError):
        store_wav("music_../x", b"RIFF", home=tmp_path, output_format="wav")


def test_hold_keeps_finished_music_stages() -> None:
    text = np.zeros((2, 4), dtype=np.float32)
    hold = ClipHold(stage="step", step=2, text=text, lyric=text, timbre=np.zeros(4), noise=np.zeros((64, 8)))
    assert hold.reached("text")
    assert hold.reached("step")
    assert not hold.reached("decode")
    assert ClipHold(stage="text").reached("text")
    assert not ClipHold(stage="text").reached("lyric")


def test_a_kept_waveform_is_packed_without_the_device() -> None:
    audio = np.zeros((2, 8), dtype=np.float32)
    audio[0, 0] = 0.25
    payload = generate(
        None,
        {},
        None,
        "solo piano",
        "",
        10,
        0,
        "en",
        None,
        None,
        None,
        None,
        hold=ClipHold(stage="decode", audio=audio),
    )
    _rate, channels, frames = wav_info(payload)
    assert channels == 2
    assert frames == 8


def test_a_finished_latent_skips_the_dit() -> None:
    class _Gpu:
        def f32(self, array: np.ndarray) -> np.ndarray:
            return array

        def clip(self, buf: np.ndarray) -> np.ndarray:
            return buf

        def zeros(self, count: int) -> np.ndarray:
            return np.zeros(count, dtype=np.float32)

        def span(
            self,
            src: np.ndarray,
            dst: np.ndarray,
            count: int,
            src_offset: int,
            dst_offset: int,
        ) -> None:
            flat = np.ravel(src)
            dst[dst_offset : dst_offset + count] = flat[src_offset : src_offset + count]

    text = np.zeros((1, 2048), dtype=np.float32)
    hold = ClipHold(
        stage="step",
        step=8,
        text=text,
        lyric=text,
        timbre=np.zeros(2048, dtype=np.float32),
        noise=np.zeros((64, 250), dtype=np.float32),
    )
    with pytest.raises(KeyError, match="decoder.conv1"):
        generate(_Gpu(), {}, None, "solo piano", "", 10, 0, "en", None, None, None, None, hold=hold)


def test_a_lyric_hold_does_not_rerun_text() -> None:
    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> object:
            del text, add_special_tokens
            raise AssertionError("the text tower was already checkpointed")

    class _Gpu:
        def f32(self, array: np.ndarray) -> np.ndarray:
            return array

        def clip(self, buf: np.ndarray) -> np.ndarray:
            return buf

    text = np.zeros((1, 2048), dtype=np.float32)
    hold = ClipHold(stage="lyric", text=text, lyric=text)
    with pytest.raises(KeyError, match="silence_latent"):
        generate(_Gpu(), {}, _Tokenizer(), "solo piano", "", 10, 0, "en", None, None, None, None, hold=hold)


def test_a_reached_text_stage_requires_its_states() -> None:
    with pytest.raises(RuntimeError, match="the paused text is missing the text states"):
        generate(None, {}, None, "solo piano", "", 10, 0, "en", None, None, None, None, hold=ClipHold(stage="text"))


def test_a_preempt_reads_the_finished_text() -> None:
    class _Gpu:
        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            self.waited = True

        def read(self, buf: object, count: int, dtype: np.dtype) -> np.ndarray:
            del buf, dtype
            return np.ones(count, dtype=np.float32)

    class _Ask:
        done = 4

        def interrupt(self) -> str:
            return "preempt"

    gpu = _Gpu()
    with pytest.raises(ClipPaused) as caught:
        _boundary_device(gpu, _Ask(), "text", text=(object(), (2, 4)))
    assert gpu.waited
    assert caught.value.hold.stage == "text"
    assert caught.value.hold.text is not None
    assert caught.value.hold.text.shape == (2, 4)
    assert caught.value.hold.done == 4


def test_cancel_does_not_keep_the_text() -> None:
    class _Gpu:
        def wait(self) -> None:
            raise AssertionError("cancel waits for the dispatch, then drops the stage")

        def read(self, buf: object, count: int, dtype: np.dtype) -> np.ndarray:
            del buf, count, dtype
            raise AssertionError("cancel does not read a checkpoint")

    class _Ask:
        def interrupt(self) -> str:
            return "cancel"

    with pytest.raises(ClipPaused) as caught:
        _boundary_device(_Gpu(), _Ask(), "text", text=(object(), (2, 4)))
    assert caught.value.reason == "cancel"
    assert caught.value.hold.text is None


class _Flag:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False


class _Reply:
    def __init__(self) -> None:
        self.payload: dict | None = None

    def json(self, status: int, payload: dict) -> None:
        del status
        self.payload = payload


class _Pausing:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.gpu = self
        self.holds: list[str | None] = []

    def end_clip(self) -> None:
        return None

    def generate(self, fields: object, progress: object, hold: ClipHold | None = None) -> bytes:
        del fields
        self.holds.append(None if hold is None else hold.stage)
        if hold is not None and hold.stage == "text":
            return wav_bytes(np.zeros((2, 8), dtype=np.float32))
        progress("text")
        self.state["preempt"].set()
        raise ClipPaused("preempt", ClipHold(stage="text", text=np.zeros((1, 4), dtype=np.float32), done=progress.done))


def _body(prompt: str = "solo piano") -> dict[str, object]:
    return {"model": "row", "prompt": prompt, "seconds": 10, "response_format": "wav"}


def test_a_pause_closes_the_device(tmp_path) -> None:
    state: dict = {"music_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)
    runtime.closed = False

    def close() -> None:
        runtime.closed = True

    runtime.close = close
    handle_music(_Reply(), runtime, _body(), None, None, state)
    assert runtime.closed
    assert state["music_hold"]["hold"].stage == "text"


def test_preempt_pauses_and_the_same_job_resumes(tmp_path) -> None:
    state: dict = {"music_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)
    first = _Reply()
    handle_music(first, runtime, _body(), None, None, state)
    job = state["music"][first.payload["id"]]
    assert job["status"] == "in_progress"
    assert state["music_hold"]["hold"].stage == "text"
    assert state["music_hold"]["job"] is job
    state["preempt"].clear()
    second = _Reply()
    handle_music(second, runtime, _body(), None, None, state)
    assert runtime.holds == [None, "text"]
    assert second.payload["id"] == job["id"]
    assert job["status"] == "completed"
    assert "music_hold" not in state
    assert (music_dir(tmp_path) / f"{job['id']}.wav").is_file()


def test_halt_stops_the_clip_like_cancel(tmp_path) -> None:
    state: dict = {"music_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)
    runtime._halted = True

    def generate(fields: object, progress: object, hold: ClipHold | None = None) -> bytes:
        del fields, hold
        reason = progress.interrupt()
        raise ClipPaused(str(reason), ClipHold(stage="start"))

    runtime.generate = generate
    reply = _Reply()
    handle_music(reply, runtime, _body(), None, None, state)
    job = state["music"][reply.payload["id"]]
    assert job["status"] == "in_progress"
    assert "music_hold" not in state


def test_cancel_discards_the_music_hold(tmp_path) -> None:
    state: dict = {"music_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)

    def generate(fields: object, progress: object, hold: ClipHold | None = None) -> bytes:
        del fields, progress, hold
        raise ClipPaused("cancel", ClipHold(stage="text", text=np.zeros((1, 4), dtype=np.float32)))

    runtime.generate = generate
    state["cancel"].set()
    reply = _Reply()
    handle_music(reply, runtime, _body(), None, None, state)
    job = state["music"][reply.payload["id"]]
    assert job["status"] == "in_progress"
    assert "music_hold" not in state
    assert not music_dir(tmp_path).exists() or not any(music_dir(tmp_path).glob("*.wav"))


def test_a_different_prompt_drops_the_paused_clip(tmp_path) -> None:
    state: dict = {"music_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)
    handle_music(_Reply(), runtime, _body(), None, None, state)
    paused = state["music_hold"]["job"]["id"]

    def finish(fields: object, progress: object, hold: ClipHold | None = None) -> bytes:
        del fields, progress
        assert hold is None
        return wav_bytes(np.zeros((2, 8), dtype=np.float32))

    runtime.generate = finish
    state["preempt"].clear()
    reply = _Reply()
    handle_music(reply, runtime, _body("other"), None, None, state)
    assert reply.payload["id"] != paused
    assert state["music"][paused]["status"] == "in_progress"
    assert state["music"][reply.payload["id"]]["status"] == "completed"
    assert "music_hold" not in state


def _brute_dependency(pos: int) -> tuple[int, int]:
    """Every convolution tap, walked backward from one output sample."""

    def conv(
        lo: int,
        hi: int,
        kernel: int,
        pad: int,
        stride: int,
        dilation: int,
    ) -> tuple[int, int]:
        return lo * stride - pad, hi * stride + (kernel - 1) * dilation - pad

    def transpose(lo: int, hi: int, kernel: int, pad: int, stride: int) -> tuple[int, int]:
        found: list[int] = []
        for sample in range(lo, hi + 1):
            for tap in range(kernel):
                numer = sample + pad - tap
                if numer >= 0 and numer % stride == 0:
                    found.append(numer // stride)
        return min(found), max(found)

    def residual(lo: int, hi: int, dilation: int) -> tuple[int, int]:
        lo, hi = conv(lo, hi, 1, 0, 1, 1)
        pad = ((7 - 1) * dilation) // 2
        return conv(lo, hi, 7, pad, 1, dilation)

    lo, hi = conv(pos, pos, 7, 3, 1, 1)
    for stride in (2, 4, 4, 6, 10):
        for dilation in (9, 3, 1):
            lo, hi = residual(lo, hi, dilation)
        lo, hi = transpose(lo, hi, stride * 2, stride // 2, stride)
    return conv(lo, hi, 7, 3, 1, 1)
