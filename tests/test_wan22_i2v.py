"""Geometry, pack, and the graph record. They do not dispatch Metal."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from wan22_i2v.chat import VideoRequestError, _seconds_label, handle_video, video_fields
from wan22_i2v.formats import block_winners, pack_linear, relative_error, restore_linear
from wan22_i2v.hold import ClipHold, ClipPaused, reload_quietly
from wan22_i2v.latent import condition_volume, unpatchify
from wan22_i2v.png import decode_png, encode_png
from wan22_i2v.progress import ClipProgress, clip_label, sample_total
from wan22_i2v.record import quantize, write_graph_source
from wan22_i2v.schedule import (
    FIVE_SEC_FRAMES,
    P320_HEIGHT,
    P320_WIDTH,
    SHORT_FRAMES,
    expert_name,
    geometry,
    mask_latent,
    promote_count,
    timesteps,
)
from wan22_i2v.source import (
    FILES,
    dit_dir,
    expert_paths,
    mask_from_prefix,
    umt5_dir,
    vae_dir,
)
from wan22_i2v.text import encoder_path, tokenizer_path
from wan22_i2v.vae import vae_path
from wan22_i2v.video import avi_meta, write_avi, write_mp4


def test_setup_writes_each_repo_under_models() -> None:
    assert vae_dir().parts[-2:] == ("Wan-AI", "Wan2.2-I2V-A14B")
    assert umt5_dir().parts[-2:] == ("google", "umt5-xxl")
    assert dit_dir().parts[-2:] == ("lightx2v", "Wan2.2-Distill-Models")
    assert vae_path().parent == vae_dir()
    assert encoder_path().parent == tokenizer_path().parent == umt5_dir()
    other = umt5_dir(repo="example/umt5")
    assert other.parts[-2:] == ("example", "umt5")
    assert encoder_path(repo="example/umt5").parent == other
    assert [item[2] for item in FILES] == [
        "Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth",
        "google/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth",
        "lightx2v/Wan2.2-Distill-Models/high.safetensors",
        "lightx2v/Wan2.2-Distill-Models/low.safetensors",
        "google/umt5-xxl/tokenizer.json",
    ]
    high, low = expert_paths()
    assert high.parent == low.parent == dit_dir()
    assert high.name == "high.safetensors"
    assert low.name == "low.safetensors"


def test_resume_keeps_only_full_ranges() -> None:
    done = mask_from_prefix(100, 40, 16)
    assert list(done) == [1, 1, 0, 0, 0, 0, 0]
    done = mask_from_prefix(100, 100, 16)
    assert list(done) == [1, 1, 1, 1, 1, 1, 1]


def test_promote_count_is_eight_blocks() -> None:
    assert promote_count(40) == 8
    assert promote_count(1) == 0


def test_tied_block_keeps_the_lower_index() -> None:
    scores = {0: 0.1, 1: 0.5, 2: 0.5, 3: 0.2}
    assert block_winners(scores, 2) == [1, 2]
    assert block_winners(scores, 1) == [1]


def test_shortest_320p_and_five_seconds() -> None:
    assert (SHORT_FRAMES, P320_WIDTH, P320_HEIGHT) == (5, 576, 320)
    assert geometry(SHORT_FRAMES, P320_HEIGHT, P320_WIDTH) == (2, 40, 72, 1440)
    assert FIVE_SEC_FRAMES == 81
    assert geometry(FIVE_SEC_FRAMES, P320_HEIGHT, P320_WIDTH)[0] == 21


def test_euler_grid_and_expert_switch() -> None:
    times = timesteps()
    assert times.shape == (5,)
    assert times[0] == np.float32(1000)
    assert times[1] == np.float32(937.5)
    assert abs(float(times[2]) - 833.3333333) < 1e-3
    assert times[3] == np.float32(625)
    assert times[4] == np.float32(0)
    assert [expert_name(float(value)) for value in times[:-1]] == ["high", "high", "low", "low"]


def test_umt5_buckets_are_bidirectional() -> None:
    from wan22_i2v.text import relative_buckets

    buckets = relative_buckets(3, 3)
    assert buckets[1, 1] == 0
    assert buckets[1, 2] == 17
    assert buckets[1, 0] == 1


def test_mask_marks_the_first_latent_frame() -> None:
    mask = mask_latent(5)
    assert mask.shape == (4, 2)
    assert np.all(mask[:, 0] == 1)
    assert np.all(mask[:, 1] == 0)


def test_unpatchify_is_spatial_patch_then_channel() -> None:
    tokens = np.zeros((1, 64), dtype=np.float32)
    tokens[0, 0] = 3.0
    tokens[0, 1] = 7.0
    tokens[0, 16] = 4.0
    tokens[0, 32] = 5.0
    tokens[0, 48] = 6.0
    out = unpatchify(tokens, (1, 1, 1))
    assert out.shape == (16, 1, 2, 2)
    assert out[0, 0, 0, 0] == 3.0
    assert out[1, 0, 0, 0] == 7.0
    assert out[0, 0, 0, 1] == 4.0
    assert out[0, 0, 1, 0] == 5.0
    assert out[0, 0, 1, 1] == 6.0


def test_condition_is_noise_then_mask_then_latent() -> None:
    noise = np.zeros((16, 2, 1, 1), dtype=np.float32)
    latent = np.ones((16, 2, 1, 1), dtype=np.float32)
    volume = condition_volume(noise, latent, 5)
    assert volume.shape == (36, 2, 1, 1)
    assert np.all(volume[:16] == 0)
    assert np.all(volume[16:20, 0] == 1)
    assert np.all(volume[16:20, 1] == 0)
    assert np.all(volume[20:] == 1)


def test_fp4_roundtrip_of_a_codepoint() -> None:
    weight = np.ones((4, 64), dtype=np.float32)
    pack = pack_linear(weight, None, "fp4")
    restored = restore_linear(pack)
    assert relative_error(restored, weight) == 0.0


def test_png_and_avi_roundtrip() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    assert np.array_equal(decode_png(encode_png(image)), image)
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    frames[0, 0, 0] = (10, 20, 30)
    blob = write_avi(frames, fps=16)
    assert avi_meta(blob) == (1, 2, 2)
    start = blob.find(b"00db")
    top = blob[start + 8 + 8 : start + 8 + 8 + 3]
    assert top == bytes((30, 20, 10))


def test_graph_quantize_promotes_nothing(tmp_path) -> None:
    source = write_graph_source(tmp_path / "source")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    manifest = quantize(
        source / "high.safetensors",
        source / "low.safetensors",
        tmp_path / "record",
        config,
    )
    assert manifest["high_promoted"] == []
    assert manifest["low_promoted"] == []
    assert (tmp_path / "record" / "embed.npy").is_file()


def _png() -> str:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return base64.b64encode(encode_png(image)).decode("ascii")


def test_video_request_defaults_to_320p_and_81_frames() -> None:
    encoded = _png()
    fields = video_fields({"prompt": " move ", "image": encoded})
    assert fields.prompt == "move"
    assert fields.frames == FIVE_SEC_FRAMES
    assert (fields.width, fields.height) == (P320_WIDTH, P320_HEIGHT)
    assert fields.image.shape == (P320_HEIGHT, P320_WIDTH, 3)
    snapped = video_fields({"prompt": "move", "input_reference": encoded, "seconds": "5"})
    assert snapped.frames == FIVE_SEC_FRAMES
    short = video_fields(
        {
            "prompt": "move",
            "image": f"data:image/png;base64,{encoded}",
            "frames": 9,
            "seconds": "5",
            "size": "576x320",
        }
    )
    assert short.frames == 9
    assert short.quality == "high"
    level = {"prompt": "move", "image": encoded, "frames": 5, "size": "16x16"}
    assert video_fields({**level, "quality": "hd"}).quality == "high"
    assert video_fields({**level, "quality": "standard"}).quality == "high"
    assert video_fields({**level, "quality": "max"}).quality == "max"
    try:
        video_fields({**level, "quality": "ultra"})
    except VideoRequestError as exc:
        assert exc.param == "quality"
    else:
        raise AssertionError("quality was accepted")
    assert _seconds_label(FIVE_SEC_FRAMES) == "5.0625"
    assert _seconds_label(9) == "0.5625"


class _Reply:
    def __init__(self) -> None:
        self.status = 0
        self.payload: dict = {}

    def json(self, status: int, payload: dict) -> None:
        self.status = status
        self.payload = json.loads(json.dumps(payload))


class _Clip:
    def generate(self, prompt, image, frames, width, height, seed, progress=None):
        del prompt, image, seed
        if progress is not None:
            progress.stage("text")
            progress.advance()
        clip = np.zeros((frames, height, width, 3), dtype=np.uint8)
        return write_avi(clip, fps=16), {"steps": [1]}


class _Broken:
    def generate(self, prompt, image, frames, width, height, seed, progress=None):
        del prompt, image, frames, width, height, seed, progress
        message = "vae missing"
        raise RuntimeError(message)


def test_max_quality_asks_for_crf_14(tmp_path, monkeypatch) -> None:
    source = tmp_path / "clip.avi"
    source.write_bytes(b"RIFF")
    seen: dict[str, list[str]] = {}

    def fake_run(command, **_kwargs):
        seen["command"] = list(command)
        tmp_path.joinpath("clip.tmp.mp4").write_bytes(b"mp4")

        class _Result:
            returncode = 0
            stderr = b""

        return _Result()

    monkeypatch.setattr("favlo.videos.subprocess.run", fake_run)
    write_mp4(source, "max")
    command = seen["command"]
    assert command[command.index("-crf") + 1] == "14"


def test_two_content_qualities_share_one_avi(tmp_path) -> None:
    from favlo.videos import load_video, source_path, store_source

    frames = np.random.default_rng(0).integers(0, 256, size=(5, 16, 16, 3), dtype=np.uint8)
    store_source("video_q", write_avi(frames, fps=16), home=tmp_path)
    write_mp4(source_path(tmp_path, "video_q"), "high")
    low = load_video("video_q", home=tmp_path, quality="low")
    high = load_video("video_q", home=tmp_path)
    maximum = load_video("video_q", home=tmp_path, quality="max")
    assert low is not None and high is not None and maximum is not None
    assert low[4:8] == high[4:8] == maximum[4:8] == b"ftyp"
    assert len(maximum) > len(low)
    assert load_video("video_q", home=tmp_path, quality="max") == maximum


def test_video_job_returns_before_the_mp4_is_stored(tmp_path) -> None:
    from favlo.videos import load_video, source_path

    handler = _Reply()
    state: dict = {"video_home": tmp_path}
    body = {"prompt": "move", "image": _png(), "frames": 5, "size": "16x16", "model": "row"}
    handle_video(handler, _Clip(), body, None, None, state)
    assert handler.status == 200
    assert handler.payload["status"] == "in_progress"
    assert handler.payload["seconds"] == "0.3125"
    assert handler.payload["size"] == "16x16"
    assert handler.payload["quality"] == "high"
    job = state["videos"][handler.payload["id"]]
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["stage"] is None
    stored_mp4 = load_video(job["id"], home=tmp_path)
    assert stored_mp4 is not None and stored_mp4[4:8] == b"ftyp"
    assert source_path(tmp_path, job["id"]).read_bytes()[:4] == b"RIFF"
    failed = _Reply()
    failed_state: dict = {}
    handle_video(failed, _Broken(), body, None, None, failed_state)
    assert failed.payload["status"] == "in_progress"
    stored = failed_state["videos"][failed.payload["id"]]
    assert stored["status"] == "failed"
    assert stored["progress"] == 0
    assert stored["error"]["message"] == "vae missing"


def test_progress_counts_finished_stages() -> None:
    job = {"status": "in_progress", "progress": 0, "stage": None}
    row: dict[str, object] = {}

    class _Row:
        def __init__(self) -> None:
            self.done = 0
            self.detail = ""

    class _Console:
        shows_bars = True

        def begin(self, phase: str, **kwargs: object) -> _Row:
            row["phase"] = phase
            row["unit"] = kwargs["unit"]
            row["total"] = kwargs["total"]
            row["detail"] = kwargs["detail"]
            opened = _Row()
            opened.detail = str(kwargs["detail"])
            row["handle"] = opened
            return opened

        def end(self, progress: _Row, summary: str) -> None:
            row["summary"] = summary
            row["closed"] = progress.done

        def log(self, line: str) -> None:
            row["line"] = line

    label = clip_label(576, 320, 81, 16)
    assert label == "576x320 81 frames 5.0625s"
    assert (
        sample_total(
            steps=4,
            dit_layers=40,
            loads=2,
            weights=20,
            text_layers=24,
            vae_blocks=21 * 36,
            pipelines=19,
            gpus=3,
        )
        == 24 + 40 + 160 + 756 + 57
    )
    progress = ClipProgress(job=job, console=_Console(), clip=label)
    progress.open(9)
    assert row["line"] == label
    assert row["phase"] == "sample"
    assert row["unit"] == "layers"
    assert row["total"] == 9
    assert row["detail"] == ""
    progress.stage("text")
    assert row["handle"].detail == "text"
    progress.advance()
    assert job["stage"] == "text"
    assert job["progress"] == 11
    assert isinstance(job["eta"], int)
    for _ in range(7):
        progress.advance()
    assert job["progress"] == 88
    assert job["status"] == "in_progress"
    progress.note("step 0 expert high 1.00s")
    assert row["line"] == "step 0 expert high 1.00s"
    progress.complete()
    job["status"] = "completed"
    progress.close(progress.summary())
    assert job["progress"] == 100
    assert job["stage"] is None
    assert row["summary"] == "sample 9/9 layers"
    assert job["eta"] == 0
    assert row["closed"] == 9


def test_hold_keeps_a_finished_step_and_reloads_its_expert_quietly() -> None:
    hold = ClipHold(stage="step", step=2, expert="low", done=4)
    assert hold.reached("encode")
    assert hold.reached("step")
    assert not hold.reached("decode")
    assert reload_quietly(hold, "low", 2)
    assert not reload_quietly(hold, "low", 3)
    assert not reload_quietly(hold, "high", 2)
    assert not reload_quietly(ClipHold(stage="encode"), "low", 0)


class _Flag:
    def __init__(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False


class _Pausing:
    """Stops after the text stage once, then packs the clip on the resume."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.stages: list[str | None] = []

    def generate(self, prompt, image, frames, width, height, seed, progress=None):
        del prompt, image, seed
        hold = getattr(progress, "hold", None)
        self.stages.append(None if hold is None else hold.stage)
        if hold is not None and hold.reached("text"):
            clip = np.zeros((frames, height, width, 3), dtype=np.uint8)
            return write_avi(clip, fps=16), {"steps": [1]}
        progress.stage("text")
        progress.advance()
        self.state["preempt"].set()
        reason = progress.interrupt()
        raise ClipPaused(reason, ClipHold(stage="text", done=progress.done))


def test_preempt_pauses_and_the_same_job_resumes(tmp_path) -> None:
    from favlo.videos import load_video

    state: dict = {"video_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)
    body = {"prompt": "move", "image": _png(), "frames": 5, "size": "16x16", "model": "row"}
    first = _Reply()
    handle_video(first, runtime, body, None, None, state)
    job = state["videos"][first.payload["id"]]
    assert job["status"] == "in_progress"
    assert state["video_hold"]["hold"].stage == "text"
    assert state["video_hold"]["job"] is job
    state["preempt"].clear()
    second = _Reply()
    handle_video(second, runtime, body, None, None, state)
    assert runtime.stages == [None, "text"]
    assert second.payload["id"] == job["id"]
    assert job["status"] == "completed"
    assert "video_hold" not in state
    stored = load_video(job["id"], home=tmp_path)
    assert stored is not None and stored[4:8] == b"ftyp"


def test_expert_load_checkpoints_the_unfinished_step(monkeypatch) -> None:
    import wan22_i2v.runtime as runtime_mod
    from wan22_i2v.runtime import Runtime

    monkeypatch.setattr(
        runtime_mod,
        "_volume",
        lambda *_args, **_kwargs: np.zeros((1,), dtype=np.float32),
    )

    class _Host:
        def __init__(self) -> None:
            self.saved: ClipHold | None = None

        def _pause(self, progress, hold) -> None:
            hold.done = progress.done
            self.saved = hold
            raise ClipPaused("preempt", hold)

    host = _Host()
    progress = ClipProgress(console=None, clip="clip")
    progress.interrupt = lambda: "preempt"
    progress.done = 4
    table = np.zeros((1,), dtype=np.float32)
    args = (
        host,
        progress,
        1,
        "high_noise",
        table,
        object(),
        object(),
        object(),
        (1,),
        (1,),
        [0.5],
        object(),
    )
    try:
        Runtime._load_stop(*args, quiet=False)(False)
    except ClipPaused:
        pass
    assert host.saved is not None
    assert host.saved.stage == "step"
    assert host.saved.step == 1
    assert host.saved.expert is None
    assert host.saved.done == 4
    try:
        Runtime._load_stop(*args, quiet=True)(True)
    except ClipPaused:
        pass
    assert host.saved.expert == "high_noise"
    assert host.saved.step == 1


def test_cancel_discards_the_hold(tmp_path) -> None:
    state: dict = {"video_home": tmp_path, "preempt": _Flag(), "cancel": _Flag()}
    runtime = _Pausing(state)
    runtime.state["preempt"].set = lambda: runtime.state["cancel"].set()
    body = {"prompt": "move", "image": _png(), "frames": 5, "size": "16x16", "model": "row"}
    reply = _Reply()
    handle_video(reply, runtime, body, None, None, state)
    job = state["videos"][reply.payload["id"]]
    assert job["status"] == "in_progress"
    assert "video_hold" not in state


def test_close_drops_the_device_and_the_expert() -> None:
    from wan22_i2v.dit import Resident
    from wan22_i2v.runtime import Runtime

    class _Gpu:
        def __init__(self) -> None:
            self.closed = False
            self.unkept: list[object] = []
            self.reclaimed = False

        def unkeep(self, buf: object) -> None:
            self.unkept.append(buf)

        def reclaim(self) -> None:
            self.reclaimed = True

        def close(self) -> None:
            self.closed = True

    gpu = _Gpu()
    runtime = Runtime.__new__(Runtime)
    runtime.gpu = gpu
    runtime._forward = None
    buf = object()
    host = np.zeros(2, dtype=np.float32)
    runtime._resident = Resident(
        packs={"pack": host},
        weights={"weight": {}},
        halves={},
        host={"bias": host},
        owned=[buf],
    )
    runtime.close()
    assert gpu.closed
    assert gpu.reclaimed
    assert buf in gpu.unkept
    assert runtime.gpu is None
    assert runtime._resident is None
    runtime.close()


def test_open_runtime_forwards_the_listener_owner(monkeypatch) -> None:
    from wan22_i2v import runtime as runtime_mod

    seen: dict[str, object] = {}

    def fake(
        path: Path,
        console: object = None,
        *,
        owner: str | None = None,
        max_memory: object = None,
    ) -> str:
        del path, console
        seen["owner"] = owner
        seen["cap"] = max_memory
        return "runtime"

    monkeypatch.setattr(runtime_mod, "Runtime", fake)
    opened = runtime_mod.open_runtime(Path("record"), owner="wan22-i2v:4", max_memory="16GiB")
    assert opened == "runtime"
    assert seen == {"owner": "wan22-i2v:4", "cap": "16GiB"}


def test_missing_record_raises_before_the_device(tmp_path: Path) -> None:
    from wan22_i2v.runtime import Runtime

    with pytest.raises(FileNotFoundError):
        Runtime(tmp_path / "missing", owner="wan22-i2v:4")


def test_forward_scratch_grows_with_the_clip() -> None:
    from wan22_i2v.dit import forward_bytes

    config = {"dim": 64, "ffn": 128, "text_len": 8, "text_dim": 32, "freq_dim": 16}
    short = forward_bytes(config, 16, 36 * 2 * 4 * 4)
    longer = forward_bytes(config, 160, 36 * 20 * 4 * 4)
    assert longer > short


def test_a_short_ledger_keeps_the_requested_clip(monkeypatch, tmp_path: Path) -> None:
    from wan22_i2v import runtime as runtime_mod

    runtime = runtime_mod.Runtime.__new__(runtime_mod.Runtime)
    runtime.path = tmp_path
    runtime._memory_cap = 1
    runtime.config = {"dim": 128, "ffn": 256, "text_len": 16, "text_dim": 32, "freq_dim": 16}
    monkeypatch.setattr(runtime, "_room", lambda: 1)
    lat_t, lat_h, lat_w, tokens = geometry(5, 320, 576)
    with pytest.raises(RuntimeError, match="5 frames at 576x320"):
        runtime._reserve(5, 576, 320, lat_t, lat_h, lat_w, tokens)


def test_decode_keeps_one_latent_frame_when_the_volume_does_not_fit() -> None:
    from wan22_i2v.runtime import resident_latent_frames

    frames, height, width, lat_t = 81, 320, 576, 21
    stage = 3 * 4 * height * width * 4
    full = 3 * frames * height * width * 4
    assert resident_latent_frames(full, frames, height, width, lat_t) == lat_t
    assert resident_latent_frames(stage, frames, height, width, lat_t) == 1
    with pytest.raises(RuntimeError, match="81 frames at 576x320"):
        resident_latent_frames(0, frames, height, width, lat_t)
