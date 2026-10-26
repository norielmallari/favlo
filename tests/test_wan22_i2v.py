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
from wan22_i2v.progress import ClipProgress, clip_label, stage_shares
from wan22_i2v.record import load_expert, quantize, quantize_int8, write_graph_source
from wan22_i2v.schedule import (
    FIVE_SEC_FRAMES,
    P320_HEIGHT,
    P320_WIDTH,
    SAMPLE_NEG_PROMPT,
    SHORT_FRAMES,
    expert_name,
    geometry,
    guide_scale,
    list_resolutions,
    mask_latent,
    promote_count,
    resolved_negative,
    timesteps,
    velocity_walks,
    video_modes,
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


def test_a_custom_umt5_replaces_the_default_text_files() -> None:
    from wan22_i2v.source import file_list, repo_id

    rows = file_list("example-org/example-model", sizes=lambda _repo, name: len(name))
    stored = [row[2] for row in rows]
    assert stored[0] == "Wan-AI/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
    assert "google/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth" not in stored
    assert "google/umt5-xxl/tokenizer.json" not in stored
    assert stored[1] == "example-org/example-model/models_t5_umt5-xxl-enc-bf16.pth"
    assert stored[-1] == "example-org/example-model/tokenizer.json"
    link = "https://huggingface.co/example-org/example-model/tree/main"
    assert repo_id(link) == "example-org/example-model"
    assert file_list(link, sizes=lambda _repo, name: len(name))[1][0] == "example-org/example-model"


def test_name_is_the_packed_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wan22_i2v.commands import main

    def models_dir(root: Path | None = None) -> Path:
        del root
        return tmp_path / "models"

    monkeypatch.setattr("wan22_i2v.source.models_dir", models_dir)
    monkeypatch.setattr("favlo.registry.models_dir", models_dir)
    seen: dict[str, Path] = {}

    def quantize(
        high: Path,
        low: Path,
        dest: Path,
        config: dict,
        *,
        high_dir: Path | None = None,
        low_dir: Path | None = None,
        percent: int = 20,
    ) -> dict:
        del percent
        del high, low, config
        seen["dest"] = dest
        seen["high"] = high_dir
        seen["low"] = low_dir
        dest.mkdir(parents=True)
        return {"high_promoted": [1], "low_promoted": [2]}

    def quantize_umt5(src: Path | None = None, dest: Path | None = None) -> float:
        seen["src"] = src
        seen["text"] = dest
        assert dest is not None
        dest.mkdir(parents=True)
        return 0.0

    monkeypatch.setattr("wan22_i2v.record.quantize", quantize)
    monkeypatch.setattr("wan22_i2v.umt5_pack.quantize_umt5", quantize_umt5)
    high = tmp_path / "high.safetensors"
    low = tmp_path / "low.safetensors"
    high.write_bytes(b"h")
    low.write_bytes(b"l")
    named = main(["quantize", "--high", str(high), "--low", str(low), "--name", "example-model"])
    assert named == 0
    assert seen["dest"] == tmp_path / "models" / "example-model"
    registry = json.loads((tmp_path / "models" / "registry.json").read_text())
    row = registry["models"]["example-model"]
    assert row["alias"] is None
    assert row["directory"] == str(tmp_path / "models" / "example-model")
    assert row["parts"]["high"] == str(tmp_path / "models" / "example-model" / "high")
    assert row["parts"]["low"] == str(tmp_path / "models" / "example-model" / "low")
    high_dir = tmp_path / "models" / "example-high"
    low_dir = tmp_path / "models" / "example-low"
    custom = main(
        [
            "quantize",
            "--high",
            str(high),
            "--low",
            str(low),
            "--name",
            "example-parts",
            "--high-dir",
            str(high_dir),
            "--low-dir",
            str(low_dir),
        ]
    )
    assert custom == 0
    assert seen["high"] == high_dir
    assert seen["low"] == low_dir
    refused = main(
        [
            "quantize",
            "--high",
            str(high),
            "--low",
            str(low),
            "--name",
            "example-model",
            "--out",
            "packed",
        ]
    )
    assert refused == 1
    packed = main(
        ["quantize-text", "--name", "example-model", "--umt5", "example-org/example-model"]
    )
    assert packed == 0
    assert seen["text"] == tmp_path / "models" / "google" / "umt5-xxl-int8"
    assert seen["src"] == models_dir() / "example-org" / "example-model" / (
        "models_t5_umt5-xxl-enc-bf16.pth"
    )
    stored = json.loads((tmp_path / "models" / "registry.json").read_text())
    umt5 = tmp_path / "models" / "google" / "umt5-xxl-int8"
    assert stored["models"]["example-model"]["parts"]["umt5"] == str(umt5)
    custom_umt5 = tmp_path / "models" / "example-umt5"
    moved = main(
        ["quantize-text", "--name", "example-model", "--out", str(custom_umt5)]
    )
    assert moved == 0
    stored = json.loads((tmp_path / "models" / "registry.json").read_text())
    row = stored["models"]["example-model"]
    assert row["parts"]["umt5"] == str(custom_umt5)
    assert row["directory"] == str(tmp_path / "models" / "example-model")


def test_resume_keeps_only_full_ranges() -> None:
    done = mask_from_prefix(100, 40, 16)
    assert list(done) == [1, 1, 0, 0, 0, 0, 0]
    done = mask_from_prefix(100, 100, 16)
    assert list(done) == [1, 1, 1, 1, 1, 1, 1]


def test_promote_count_is_eight_blocks() -> None:
    assert promote_count(40) == 8
    assert promote_count(40, 50) == 20
    assert promote_count(24) == 4
    assert promote_count(1) == 0


def test_tied_block_keeps_the_lower_index() -> None:
    scores = {0: 0.1, 1: 0.5, 2: 0.5, 3: 0.2}
    assert block_winners(scores, 2) == [1, 2]
    assert block_winners(scores, 1) == [1]


def test_video_mode_is_image_to_video() -> None:
    assert video_modes() == ("i2v",)


def test_listed_resolutions_are_the_four_buckets() -> None:
    listed = list_resolutions()
    assert [row["size"] for row in listed] == [
        "416x240",
        "240x416",
        "576x320",
        "320x576",
        "832x480",
        "480x832",
        "1280x720",
        "720x1280",
    ]
    assert [row["name"] for row in listed] == [
        "240p",
        "240p",
        "320p",
        "320p",
        "480p",
        "480p",
        "720p",
        "720p",
    ]


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
    assert velocity_walks(times, 1, 1) == 4
    assert velocity_walks(times, 3.5, 3.5) == 8
    assert velocity_walks(times, 1.0, 3.5) == 6
    assert guide_scale(900, 1, 3.5) == 3.5
    assert guide_scale(833.3, 1, 3.5) == 1
    assert resolved_negative("  ") == SAMPLE_NEG_PROMPT
    assert resolved_negative(" blur ") == "blur"
    assert timesteps(8).shape == (9,)


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


def test_a_clip_encodes_with_the_int8_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wan22_i2v.commands import main
    from wan22_i2v.runtime import _text_encoder

    def models_dir(root: Path | None = None) -> Path:
        del root
        return tmp_path / "models"

    monkeypatch.setattr("wan22_i2v.source.models_dir", models_dir)
    monkeypatch.setattr("favlo.registry.models_dir", models_dir)
    monkeypatch.setattr("wan22_i2v.runtime.tokenizer_path", lambda: tmp_path / "tokenizer.json")

    def quantize_int8(
        high: Path,
        low: Path,
        dest: Path,
        config: dict,
        *,
        high_dir: Path | None = None,
        low_dir: Path | None = None,
    ) -> dict:
        del high, low, config, high_dir, low_dir
        dest.mkdir(parents=True)
        return {"high_worst": 0.0, "low_worst": 0.0}

    monkeypatch.setattr("wan22_i2v.record.quantize_int8", quantize_int8)
    high = tmp_path / "high.safetensors"
    low = tmp_path / "low.safetensors"
    high.write_bytes(b"h")
    low.write_bytes(b"l")
    assert main(["quantize", "--int8", "--high", str(high), "--low", str(low)]) == 0
    stored = json.loads((tmp_path / "models" / "registry.json").read_text())
    pack = tmp_path / "models" / "google" / "umt5-xxl-int8"
    assert stored["models"]["Wan2.2-I2V-A14B-int8"]["parts"]["umt5"] == str(pack)
    assert _text_encoder(4096, pack) is False
    pack.mkdir(parents=True)
    (pack / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert _text_encoder(4096, pack) is True
    assert _text_encoder(4096, tmp_path / "missing") is False


def test_int8_roundtrip_stays_within_half_a_step() -> None:
    weight = np.array([[2.0, -1.0, 0.25, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    bias = np.array([0.5, -0.25], dtype=np.float32)
    pack = pack_linear(weight, bias, "int8")
    restored = restore_linear(pack)
    assert pack.kind == "int8"
    assert pack.packed.dtype == np.int8
    assert int(pack.packed.min()) >= -127
    assert int(pack.packed.max()) <= 127
    assert restored.shape == weight.shape
    step = 2.0 / 254.0
    assert np.max(np.abs(restored[0] - weight[0])) <= step + 1e-5
    assert restored[0, 0] == pytest.approx(2.0)
    assert np.all(restored[1] == 0.0)
    assert relative_error(restored, weight) < 0.02
    assert pack.bias.dtype == np.float16


def test_graph_int8_pack_keeps_the_head_in_half(tmp_path: Path) -> None:
    source = write_graph_source(tmp_path / "source")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    manifest = quantize_int8(
        source / "high.safetensors",
        source / "low.safetensors",
        tmp_path / "record",
        config,
    )
    assert manifest["weight"] == "int8"
    loaded = load_expert(tmp_path / "record" / "high", int(config["layers"]))
    assert loaded.packs["blocks.0.ffn.0.weight"].kind == "int8"
    assert loaded.halves["head.head.weight"].dtype == np.float16


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
    assert manifest["fp8_percent"] == 20
    high = tmp_path / "record" / "high"
    head = high / "head.head.weight.npy"
    assert head.is_file()
    assert not (high / "head.head.weight.npz").is_file()
    assert (high / "head.head.bias.npy").is_file()
    assert (tmp_path / "record" / "embed.npy").is_file()
    loaded = load_expert(high, int(config["layers"]))
    assert loaded.halves["head.head.weight"].dtype == np.float16
    assert loaded.halves["head.head.weight"].shape == (64, 128)
    weight = np.array(loaded.halves["head.head.weight"], dtype=np.float32)
    bias = np.array(loaded.halves["head.head.bias"], dtype=np.float32)
    head.unlink()
    (high / "head.head.bias.npy").unlink()
    packed = pack_linear(weight, bias, "fp4")
    np.savez(
        high / "head.head.weight.npz",
        kind=np.array(packed.kind),
        packed=packed.packed,
        scales=packed.scales,
        bias=packed.bias,
        perm=packed.perm,
        logical_n=np.int32(packed.logical_n),
        logical_k=np.int32(packed.logical_k),
        padded_n=np.int32(packed.padded_n),
        stored_k=np.int32(packed.stored_k),
    )
    restored = load_expert(high, int(config["layers"]))
    assert restored.halves["head.head.weight"].shape == (64, 128)
    assert restored.halves["head.head.bias"].shape == (64,)


def _png() -> str:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return base64.b64encode(encode_png(image)).decode("ascii")


def test_duration_floors_to_four_n_plus_one() -> None:
    from wan22_i2v.schedule import frames_from_duration

    assert frames_from_duration(5.0) == 77
    assert video_fields({"prompt": "move", "image": _png(), "duration": 5.0}).frames == 77


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
    assert short.negative_prompt == ""
    assert short.guidance == (1.0, 1.0)
    assert short.steps == 4
    guided = video_fields(
        {
            "prompt": "move",
            "image": encoded,
            "frames": 5,
            "size": "16x16",
            "negative_prompt": " blur ",
            "guidance_scale": [2, 4.5],
            "steps": 8,
        }
    )
    assert guided.negative_prompt == "blur"
    assert guided.guidance == (2.0, 4.5)
    assert guided.steps == 8
    both = video_fields(
        {"prompt": "move", "image": encoded, "frames": 5, "size": "16x16", "guidance_scale": 3.5}
    )
    assert both.guidance == (3.5, 3.5)
    try:
        video_fields(
            {"prompt": "move", "image": encoded, "frames": 5, "size": "16x16", "steps": 41}
        )
    except VideoRequestError as exc:
        assert exc.param == "steps"
    else:
        raise AssertionError("steps was accepted")
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
    def generate(self, prompt, image, frames, width, height, seed, progress=None, **_rest):
        del prompt, image, seed, _rest
        if progress is not None:
            progress.stage("text")
            progress.advance()
        clip = np.zeros((frames, height, width, 3), dtype=np.uint8)
        return write_avi(clip, fps=16), {"steps": [1]}


class _Broken:
    def generate(self, prompt, image, frames, width, height, seed, progress=None, **_rest):
        del prompt, image, frames, width, height, seed, progress, _rest
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


def test_video_job_passes_guidance_into_generate(tmp_path) -> None:
    seen: dict[str, object] = {}

    class _Seen:
        def generate(self, prompt, image, frames, width, height, seed, progress=None, **kwargs):
            del image, seed
            seen["prompt"] = prompt
            seen["kwargs"] = kwargs
            if progress is not None:
                progress.stage("text")
                progress.advance()
            clip = np.zeros((frames, height, width, 3), dtype=np.uint8)
            return write_avi(clip, fps=16), {"steps": [1]}

    handler = _Reply()
    body = {
        "prompt": "move",
        "image": _png(),
        "frames": 5,
        "size": "16x16",
        "model": "row",
        "negative_prompt": " blur ",
        "guidance_scale": [1, 3.5],
        "steps": 6,
    }
    handle_video(handler, _Seen(), body, None, None, {"video_home": tmp_path})
    assert seen["prompt"] == "move"
    assert seen["kwargs"] == {
        "negative_prompt": "blur",
        "guidance": (1.0, 3.5),
        "steps": 6,
    }
    assert handler.payload["guidance_scale"] == [1.0, 3.5]
    assert handler.payload["steps"] == 6
    assert handler.payload["negative_prompt"] == "blur"


def test_stage_shares_sum_to_100() -> None:
    shares = stage_shares(text=True, vae=True, loads=2, steps=4)
    assert shares["text"] == 8
    assert shares["encode"] == 8
    assert shares["load"] == [4, 4]
    assert shares["decode"] == 12
    assert shares["pack"] == 2
    assert sum(shares["step"]) == 62
    wide = stage_shares(text=True, vae=True, loads=2, steps=80)
    total = (
        wide["text"]
        + wide["encode"]
        + sum(wide["load"])
        + sum(wide["step"])
        + wide["decode"]
        + wide["pack"]
    )
    assert total == 100
    assert all(part >= 1 for part in wide["step"])


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
    progress = ClipProgress(job=job, console=_Console(), clip=label)
    progress.open(
        text_layers=2,
        pipelines=1,
        loads=1,
        steps=1,
        encode_blocks=1,
        decode_blocks=1,
        dit_layers=2,
        files=2,
    )
    progress.stage("compile")
    assert row["line"] == label
    assert row["phase"] == "sample"
    assert row["unit"] == "layers"
    assert row["total"] == 3
    assert row["detail"] == "compile"
    progress.advance()
    progress.stage("text")
    assert row["handle"].detail == "text"
    progress.advance()
    progress.advance()
    assert job["stage"] == "text"
    assert job["progress"] == 8
    assert isinstance(job["eta"], int)
    progress.stage("load")
    assert row["phase"] == "load"
    assert row["unit"] == "files"
    assert row["total"] == 2
    assert row["summary"] == "text 3/3 layers"
    assert row["closed"] == 3
    progress.advance()
    progress.advance()
    assert job["progress"] == 12
    progress.stage("high")
    assert row["unit"] == "layers"
    assert row["detail"] == "high"
    progress.advance()
    progress.advance()
    assert job["status"] == "in_progress"
    progress.note("step 0 expert high 1.00s")
    assert row["line"] == "step 0 expert high 1.00s"
    progress.complete()
    job["status"] = "completed"
    progress.close(progress.summary())
    assert job["progress"] == 100
    assert job["stage"] is None
    assert job["eta"] == 0
    assert row["closed"] == 2


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

    def generate(self, prompt, image, frames, width, height, seed, progress=None, **_rest):
        del prompt, image, seed, _rest
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
        None,
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


def test_a_committed_layer_checkpoints_before_the_step_updates(monkeypatch) -> None:
    import wan22_i2v.runtime as runtime_mod
    from wan22_i2v.runtime import Runtime

    monkeypatch.setattr(runtime_mod, "_volume", lambda *_args, **_kwargs: np.zeros((1,)))

    class _Host:
        def __init__(self) -> None:
            self.saved: ClipHold | None = None

        def _pause(self, progress, hold) -> None:
            hold.done = progress.done
            self.saved = hold
            raise ClipPaused("preempt", hold)

    host = _Host()
    progress = ClipProgress(console=None, clip="clip")
    progress.interrupt = lambda: None
    hold = ClipHold(stage="step", step=2, expert="high")
    Runtime._after_commit(host, progress, hold)
    assert host.saved is None
    progress.interrupt = lambda: "preempt"
    progress.done = 9
    try:
        Runtime._after_commit(host, progress, hold)
    except ClipPaused:
        pass
    assert host.saved is not None
    assert host.saved.stage == "step"
    assert host.saved.step == 2
    assert host.saved.expert == "high"
    assert host.saved.done == 9


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


def test_close_confirms_the_bytes_it_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    from wan22_i2v.runtime import Runtime

    seen = {"live": 4096}

    class _Ledger:
        def owner_bytes(self) -> dict[str, int]:
            return {"wan22-i2v:9": seen["live"]}

        def sweep(self, owner: str) -> None:
            seen["swept"] = owner
            seen["live"] = 0

        def confirm_released(self, before: int, nbytes: int) -> bool:
            seen["before"] = before
            seen["nbytes"] = nbytes
            return True

    ledger = _Ledger()
    monkeypatch.setattr("wan22_i2v.runtime.memory", ledger)
    monkeypatch.setattr("favlo.memory.memory", ledger)
    monkeypatch.setattr("favlo.memory.reclaimable_bytes", lambda: 80)
    runtime = Runtime.__new__(Runtime)
    runtime.gpu = None
    runtime._forward = None
    runtime._resident = None
    runtime._owner = "wan22-i2v:9"
    runtime.close()
    assert seen["swept"] == "wan22-i2v:9"
    assert seen["before"] == 80
    assert seen["nbytes"] == 4096


def test_open_runtime_forwards_the_listener_owner(monkeypatch) -> None:
    from wan22_i2v import runtime as runtime_mod

    seen: dict[str, object] = {}

    def fake(
        path: Path,
        console: object = None,
        *,
        owner: str | None = None,
        max_memory: object = None,
        parts: object = None,
    ) -> str:
        del path, console, parts
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
    from wan22_i2v.formats import TILE_M, TILE_N, pad_to, stored_k

    config = {"dim": 64, "ffn": 128, "text_len": 8, "text_dim": 32, "freq_dim": 16}
    short = forward_bytes(config, 16, 36 * 2 * 4 * 4)
    longer = forward_bytes(config, 160, 36 * 20 * 4 * 4)
    assert longer > short
    dim = 5120
    ffn = 13824
    rows = pad_to(64, TILE_M)
    text_rows = pad_to(512, TILE_M)
    span = stored_k(max(dim, ffn, 4096, 256))
    slab = max(3 * pad_to(dim, TILE_N), pad_to(ffn, TILE_N))
    volume = 36 * 4
    official = {
        "dim": dim,
        "ffn": ffn,
        "text_len": 512,
        "text_dim": 4096,
        "freq_dim": 256,
        "out_dim": 16,
    }
    expected = (
        rows * span * 2
        + 2 * rows * dim * 4
        + rows * slab * 4
        + 64 * dim * 6 * 4
        + rows * 64 * 4
        + text_rows * dim * 4
        + text_rows * 4096 * 4
        + 64 * 256 * 4
        + volume * 4
        + 5 * dim * 4
        + dim * 6 * 4
        + (volume // 36) * 16 * 4
    )
    assert forward_bytes(official, 64, volume) == expected
    separate = 5 * rows * dim * 4 + rows * ffn * 4 - rows * slab * 4
    assert separate > 0


def test_room_subtracts_live_and_pending(monkeypatch) -> None:
    from wan22_i2v import runtime as runtime_mod

    runtime = runtime_mod.Runtime.__new__(runtime_mod.Runtime)
    runtime._memory_cap = 80

    class Ledger:
        @staticmethod
        def budget() -> int:
            return 100

        live = 30
        pending = 10

    monkeypatch.setattr(runtime_mod, "memory", Ledger())
    assert runtime._room() == 40


def test_a_short_ledger_keeps_the_requested_clip(monkeypatch, tmp_path: Path) -> None:
    from wan22_i2v import runtime as runtime_mod

    runtime = runtime_mod.Runtime.__new__(runtime_mod.Runtime)
    runtime.path = tmp_path
    runtime.high = tmp_path / "high"
    runtime.low = tmp_path / "low"
    runtime._memory_cap = 1
    runtime.config = {"dim": 128, "ffn": 256, "text_len": 16, "text_dim": 32, "freq_dim": 16}
    monkeypatch.setattr(runtime, "_room", lambda: 1)
    lat_t, lat_h, lat_w, tokens = geometry(5, 320, 576)
    with pytest.raises(RuntimeError, match="5 frames at 576x320"):
        runtime._reserve(5, 576, 320, lat_t, lat_h, lat_w, tokens)


def test_decode_keeps_one_latent_frame_when_the_volume_does_not_fit() -> None:
    from wan22_i2v.runtime import decode_peak_bytes, resident_latent_frames

    frames, height, width, lat_t = 81, 320, 576, 21
    lat_h, lat_w = height // 8, width // 8
    stage = 3 * 4 * height * width * 4
    full = 3 * frames * height * width * 4
    assert resident_latent_frames(full, frames, height, width, lat_t) == lat_t
    assert resident_latent_frames(stage, frames, height, width, lat_t) == 1
    with pytest.raises(RuntimeError, match="81 frames at 576x320"):
        resident_latent_frames(0, frames, height, width, lat_t)
    tight = decode_peak_bytes(frames, height, width, lat_t, lat_h, lat_w, stage)
    wide = decode_peak_bytes(frames, height, width, lat_t, lat_h, lat_w, full)
    assert tight < wide


def test_slim_pack_drops_host_quant_arrays() -> None:
    from wan22_i2v.formats import LinearPack, slim_pack

    pack = LinearPack(
        kind="fp4",
        packed=np.ones(8, dtype=np.uint8),
        scales=np.ones(2, dtype=np.uint8),
        bias=np.ones(4, dtype=np.float32),
        perm=np.arange(4, dtype=np.int32),
        logical_n=32,
        logical_k=256,
        padded_n=32,
        stored_k=256,
    )
    slim = slim_pack(pack)
    assert slim.logical_n == pack.logical_n
    assert slim.packed.size == 0
    assert slim.scales.size == 0


def test_vae_release_device_drops_params() -> None:
    from wan22_i2v.vae import WanVAE

    vae = WanVAE.__new__(WanVAE)
    vae.weights = {}
    vae.cache = {}
    vae._params = {"k": (object(), (1,))}
    vae._mean = object()
    vae._std = object()
    vae._own = False
    dropped: list[object] = []

    class Gpu:
        def drop(self, buf: object) -> None:
            dropped.append(buf)

        def reclaim(self) -> None:
            return None

    vae.gpu = Gpu()
    vae.release_device()
    assert vae._params == {}
    assert vae._mean is None
    assert vae._std is None
    assert len(dropped) == 3
