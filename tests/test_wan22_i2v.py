"""Geometry, pack, and the graph record. They do not dispatch Metal."""

from __future__ import annotations

import base64
import json

import numpy as np
from wan22_i2v.chat import _seconds_label, handle_video, video_fields
from wan22_i2v.formats import block_winners, pack_linear, relative_error, restore_linear
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
from wan22_i2v.video import avi_meta, write_avi


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
    assert _seconds_label(FIVE_SEC_FRAMES) == "5.0625"
    assert _seconds_label(9) == "0.5625"


class _Reply:
    def __init__(self) -> None:
        self.status = 0
        self.payload: dict = {}

    def _json(self, status: int, payload: dict) -> None:
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


def test_video_job_returns_before_the_mp4_is_stored() -> None:
    handler = _Reply()
    state: dict = {}
    body = {"prompt": "move", "image": _png(), "frames": 5, "size": "16x16", "model": "row"}
    handle_video(handler, _Clip(), body, None, None, state)
    assert handler.status == 200
    assert handler.payload["status"] == "in_progress"
    assert handler.payload["seconds"] == "0.3125"
    assert handler.payload["size"] == "16x16"
    job = state["videos"][handler.payload["id"]]
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["stage"] is None
    assert state["video_mp4"][job["id"]][4:8] == b"ftyp"
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
