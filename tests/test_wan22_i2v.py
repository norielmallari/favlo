"""Geometry, pack, and the graph record. They do not dispatch Metal."""

from __future__ import annotations

import base64
import json

import numpy as np
from wan22_i2v.chat import _seconds_label, handle_video, video_fields
from wan22_i2v.formats import block_winners, pack_linear, relative_error, restore_linear
from wan22_i2v.latent import condition_volume, unpatchify
from wan22_i2v.png import decode_png, encode_png
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
from wan22_i2v.source import FILES, FOLDER, expert_paths, mask_from_prefix, source_dir
from wan22_i2v.video import avi_meta, write_avi


def test_setup_writes_under_the_official_folder() -> None:
    assert FOLDER == "Wan2.2-I2V-A14B"
    assert source_dir().name == FOLDER
    assert [item[2] for item in FILES] == [
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "high.safetensors",
        "low.safetensors",
        "tokenizer.json",
    ]
    high, low = expert_paths()
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
    def generate(self, prompt, image, frames, width, height, seed):
        del prompt, image, seed
        clip = np.zeros((frames, height, width, 3), dtype=np.uint8)
        return write_avi(clip, fps=16), {"steps": [1]}


class _Broken:
    def generate(self, prompt, image, frames, width, height, seed):
        del prompt, image, frames, width, height, seed
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
    assert state["video_mp4"][job["id"]][4:8] == b"ftyp"
    failed = _Reply()
    failed_state: dict = {}
    handle_video(failed, _Broken(), body, None, None, failed_state)
    assert failed.payload["status"] == "in_progress"
    stored = failed_state["videos"][failed.payload["id"]]
    assert stored["status"] == "failed"
    assert stored["error"]["message"] == "vae missing"
