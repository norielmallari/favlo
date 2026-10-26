"""Pack, schedule, and request-field checks. They do not dispatch Metal."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from favlo.reply import MemoryReply
from z_image_turbo.chat import ImageRequestError, _fields, handle_image
from z_image_turbo.formats import dequant_fp4, pack_fp4, pack_int8, relative_error, restore_int8, stored_k
from z_image_turbo.hold import ImageHold, ImageJob, ImagePaused
from z_image_turbo.png import ZLIB_LEVEL, encode_png, pack_rgb, unpack_rgb, write_png
from z_image_turbo.progress import (
    CAPTION_PIECES,
    DECODER_PIECES,
    STEP_PIECES,
    ImageProgress,
    sample_total,
)
from z_image_turbo.record import Plane, load, quantize, weight_bytes
from z_image_turbo.runtime import Runtime, _patchify, _unpatchify, open_runtime
from z_image_turbo.schedule import sigmas, timestep
from z_image_turbo.scratch import MAX_EDGE, Scratch, scratch_bytes


def test_stored_k_pads_a_conv_row() -> None:
    assert stored_k(144) == 256
    assert stored_k(256) == 256
    assert stored_k(1152) == 1280


def test_pack_int8_pads_and_restores_the_peak() -> None:
    weight = np.zeros((3, 40), dtype=np.float32)
    weight[0, 0] = 127.0
    weight[1, 3] = -63.5
    codes, scales, stored, padded = pack_int8(weight)
    assert stored == 256
    assert padded == 128
    assert codes.shape == (128, 256)
    assert scales.shape == (128,)
    assert int(codes[0, 0]) == 127
    assert scales[0] == np.float32(1.0)
    assert scales[2] == np.float32(0.0)
    restored = restore_int8(codes, scales, 3, 40)
    assert relative_error(restored, weight) < 0.01
    assert restored[0, 0] == np.float32(127.0)


def test_pack_preserves_an_fp4_codepoint() -> None:
    weight = np.ones((8, 256), dtype=np.float32)
    packed, scales, sk = pack_fp4(weight)
    restored = dequant_fp4(packed, scales, sk)
    assert sk == 256
    assert np.allclose(restored, 1.0)


def test_pack_zero_block_stays_zero() -> None:
    weight = np.zeros((4, 256), dtype=np.float32)
    packed, scales, sk = pack_fp4(weight)
    assert np.all(scales == 0)
    assert np.all(dequant_fp4(packed, scales, sk) == 0.0)


def test_patch_keeps_the_channel_fastest() -> None:
    latent = np.zeros((16, 4, 2), dtype=np.float32)
    latent[3, 0, 1] = 7.0
    tokens = _patchify(latent)
    assert tokens.shape == (2, 64)
    assert tokens[0, 19] == 7.0
    restored = _unpatchify(tokens, 4, 2)
    assert restored[3, 0, 1] == 7.0
    assert restored.shape == (16, 4, 2)


def test_embedded_time_is_the_complement() -> None:
    assert timestep(1.0) == 0.0
    assert timestep(0.0) == 1000.0


def test_max_edge_is_the_training_square() -> None:
    assert MAX_EDGE == 1280
    assert MAX_EDGE % 16 == 0


def _covering_scratch() -> Scratch:
    scratch = Scratch.__new__(Scratch)
    scratch.width = 16
    scratch.height = 16
    scratch.image_rows = 64
    scratch.seq_rows = 1600
    scratch.latent_count = 64
    return scratch


def test_open_runtime_forwards_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(path, console=None, *, owner=None, max_memory=None):
        captured["path"] = path
        captured["owner"] = owner
        captured["max_memory"] = max_memory
        return "runtime"

    monkeypatch.setattr("z_image_turbo.runtime.Runtime", fake)
    assert open_runtime(Path("record"), owner="z-image-turbo:7", max_memory="16GiB") == "runtime"
    assert captured["owner"] == "z-image-turbo:7"
    assert captured["max_memory"] == "16GiB"


def test_a_missing_record_keeps_the_passed_owner(tmp_path: Path) -> None:
    owner = "z-image-turbo:7"
    held: dict[str, str] = {}

    class _Probe(Runtime):
        def __init__(self, path: Path, console=None, *, owner=None, max_memory=None) -> None:
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
        def __init__(self, path: Path, console=None, *, owner=None, max_memory=None) -> None:
            try:
                super().__init__(path, console, owner=owner, max_memory=max_memory)
            except FileNotFoundError:
                held["owner"] = self._owner
                raise

    with pytest.raises(FileNotFoundError):
        _Probe(tmp_path)
    minted = held["owner"]
    assert minted.startswith("z-image-turbo:")
    assert minted != "z-image-turbo:7"


def test_scratch_shrinks_with_the_image() -> None:
    assert scratch_bytes(512, 512) < scratch_bytes(MAX_EDGE, MAX_EDGE)


def test_a_short_ledger_refuses_the_requested_image() -> None:
    runtime = Runtime.__new__(Runtime)
    runtime.scratch = None
    runtime._memory_cap = None
    runtime._room = lambda: scratch_bytes(16, 32) - 1
    with pytest.raises(RuntimeError, match="16x32"):
        runtime._reserve_scratch(16, 32)
    assert runtime.scratch is None


def test_a_tight_ledger_reserves_the_requested_size(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, tuple[int, int]] = {}

    def fake_scratch(gpu, width=MAX_EDGE, height=MAX_EDGE):
        del gpu
        created["size"] = (width, height)
        scratch = Scratch.__new__(Scratch)
        scratch.width = width
        scratch.height = height
        scratch.image_rows = 1
        scratch.seq_rows = 1
        scratch.latent_count = 1
        return scratch

    monkeypatch.setattr("z_image_turbo.runtime.Scratch", fake_scratch)
    runtime = Runtime.__new__(Runtime)
    runtime.gpu = object()
    runtime.scratch = None
    needed = scratch_bytes(512, 512)
    runtime._room = lambda: needed
    runtime._reserve_scratch(512, 512)
    assert created["size"] == (512, 512)
    assert needed < scratch_bytes(MAX_EDGE, MAX_EDGE)


def test_shifted_knots_have_eight_intervals() -> None:
    knots = sigmas()
    assert knots.shape == (9,)
    assert knots[0] == np.float32(1.0)
    assert knots[-1] == np.float32(0.0)
    assert np.all(np.diff(knots) < 0)


def test_png_signature() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    blob = encode_png(image)
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    assert np.array_equal(unpack_rgb(pack_rgb(image)), image)


def test_lower_quality_uses_a_higher_zlib_level(tmp_path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = np.arange(64, dtype=np.uint8)[:, None]
    image[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
    assert ZLIB_LEVEL == {"low": 9, "medium": 7, "high": 5, "xhigh": 3, "max": 1}
    low = encode_png(image, "low")
    top = encode_png(image, "max")
    assert low != top
    assert len(low) < len(top)
    source = tmp_path / "image_a.rgb"
    source.write_bytes(pack_rgb(image))
    assert write_png(source, "low").read_bytes() == low


def test_image_request_default_size() -> None:
    fields = _fields({"prompt": "a cat"})
    assert fields.prompt == "a cat"
    assert (fields.width, fields.height, fields.seed, fields.count) == (1280, 720, 0, 1)
    assert fields.quality == "high"
    assert fields.background == "opaque"
    assert fields.output_format == "png"
    assert fields.size == "1280x720"
    assert fields.safety_identifier is None


def test_image_request_accepts_a_standard_body() -> None:
    fields = _fields(
        {
            "prompt": "a cat",
            "n": 2,
            "size": "auto",
            "quality": "auto",
            "background": "opaque",
            "moderation": "low",
            "style": "natural",
            "output_format": "png",
            "response_format": "b64_json",
            "stream": False,
            "partial_images": 0,
            "seed": 4,
            "output_compression": 80,
            "model": "Z-Image-Turbo-fp4",
            "safety_identifier": "",
            "user": "reader",
        }
    )
    assert fields.count == 2
    assert fields.seed == 4
    assert fields.size == "1280x720"
    assert fields.quality == "high"
    assert fields.safety_identifier == "reader"


def test_image_request_rejects_counts_past_ten() -> None:
    with pytest.raises(ImageRequestError, match="n must be from 1 to 10") as raised:
        _fields({"prompt": "a cat", "n": 11})
    assert raised.value.param == "n"


def test_image_request_rejects_sizes_this_model_cannot_draw() -> None:
    with pytest.raises(ImageRequestError, match="1280") as raised:
        _fields({"prompt": "a cat", "size": "1536x1024"})
    assert raised.value.param == "size"
    with pytest.raises(ImageRequestError, match="1:3"):
        _fields({"prompt": "a cat", "size": "16x1280"})


def test_image_request_rejects_unsupported_image_options() -> None:
    rejected = {
        "background": "transparent",
        "response_format": "url",
        "output_format": "jpeg",
        "stream": True,
        "partial_images": 1,
    }
    for key, value in rejected.items():
        with pytest.raises(ImageRequestError) as raised:
            _fields({"prompt": "a cat", key: value})
        assert raised.value.param == key


def test_image_response_returns_one_png_per_count(tmp_path) -> None:
    class _Runtime:
        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **_extra):
            del prompt, width, height, steps
            timing = {"total": 1.0, "caption": 0.1, "vae": 0.2, "steps": [0.1]}
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[..., 0] = seed
            return image, "digest", timing

    reply = MemoryReply()
    state = {"image_home": tmp_path}
    handle_image(reply, _Runtime(), {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert reply.status == 200
    data = reply.body["data"]
    assert isinstance(data, list)
    assert len(data) == 2
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    first[..., 0] = 7
    second = np.zeros((8, 8, 3), dtype=np.uint8)
    second[..., 0] = 8
    assert base64.b64decode(data[0]["b64_json"]) == encode_png(first, "high")
    assert base64.b64decode(data[1]["b64_json"]) == encode_png(second, "high")
    rgb = tmp_path / "images" / f"{data[0]['id']}.rgb"
    assert np.array_equal(unpack_rgb(rgb.read_bytes()), first)
    assert (tmp_path / "images" / f"{data[0]['id']}.png").read_bytes() == encode_png(first, "high")
    timings = reply.body["generation_seconds"]
    assert isinstance(timings, list)
    assert len(timings) == 2
    assert reply.body["output_format"] == "png"
    assert reply.body["background"] == "opaque"
    assert reply.body["size"] == "1280x720"
    assert "usage" not in reply.body
    assert "status" not in reply.body
    assert "revised_prompt" not in data[0]
    handle_image(reply, _Runtime(), {"prompt": "a cat"}, None, None, state)
    assert isinstance(reply.body["generation_seconds"], dict)


def _picture(seed: int) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = seed
    return image


def test_the_image_request_releases_the_runtime(tmp_path) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.closed = 0

        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **_extra):
            del prompt, width, height, steps
            timing = {"total": 1.0, "caption": 0.1, "vae": 0.2, "steps": [0.1]}
            return _picture(seed), "digest", timing

        def close(self) -> None:
            self.closed += 1

    runtime = _Runtime()
    reply = MemoryReply()
    state = {"image_home": tmp_path}
    handle_image(reply, runtime, {"prompt": "a cat", "seed": 1}, None, None, state)
    assert reply.status == 200
    assert runtime.closed == 1
    handle_image(reply, runtime, {"prompt": "a cat", "size": "3x3"}, None, None, state)
    assert runtime.closed == 2


def test_preempt_keeps_a_stored_image_and_the_same_id(tmp_path) -> None:
    preempt = threading.Event()

    class _Runtime:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **_extra):
            del prompt, width, height, steps
            self.seeds.append(seed)
            if len(self.seeds) > 1:
                preempt.set()
            timing = {"total": 1.0, "caption": 0.1, "vae": 0.2, "steps": [0.1]}
            return _picture(seed), "digest", timing

    runtime = _Runtime()
    reply = MemoryReply()
    state = {"image_home": tmp_path, "preempt": preempt, "cancel": threading.Event()}
    handle_image(reply, runtime, {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert reply.body["status"] == "in_progress"
    assert reply.body["stage"] == "decode"
    assert len(reply.body["data"]) == 1
    assert state["image_hold"]["hold"].id == reply.body["id"]
    assert runtime.seeds == [7, 8]
    preempt.clear()
    handle_image(reply, runtime, {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert "status" not in reply.body
    assert len(reply.body["data"]) == 2
    assert runtime.seeds == [7, 8]
    assert state.get("image_hold") is None


def test_a_paused_step_resumes_with_its_latent(tmp_path) -> None:
    latent = np.arange(4, dtype=np.float32)

    class _Runtime:
        def __init__(self) -> None:
            self.resumed: list[dict] = []

        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **extra):
            del prompt, width, height, steps, seed
            if not extra.get("done_steps"):
                raise ImagePaused(
                    "preempt",
                    ImageHold(stage="step", step=4, latent=latent, times=[0.2, 0.2, 0.2, 0.2]),
                )
            self.resumed.append(extra)
            timing = {"total": 2.0, "caption": 0.1, "vae": 0.2, "steps": [0.2] * 8}
            return _picture(3), "digest", timing

    runtime = _Runtime()
    reply = MemoryReply()
    state = {"image_home": tmp_path, "preempt": threading.Event(), "cancel": threading.Event()}
    handle_image(reply, runtime, {"prompt": "a cat", "seed": 1}, None, None, state)
    assert reply.body["status"] == "in_progress"
    assert reply.body["stage"] == "step"
    assert reply.body["progress"] > 0
    assert reply.body["eta"] > 0
    assert reply.body["data"] == []
    job_id = reply.body["id"]
    handle_image(reply, runtime, {"prompt": "a cat", "seed": 1}, None, None, state)
    assert "status" not in reply.body
    assert runtime.resumed[0]["done_steps"] == 4
    assert np.array_equal(runtime.resumed[0]["latent"], latent)
    assert len(reply.body["data"]) == 1
    assert state.get("image_hold") is None
    assert job_id


def test_a_waiting_chat_still_draws_the_next_step(tmp_path) -> None:
    preempt = threading.Event()
    preempt.set()
    called: list[int] = []

    class _Runtime:
        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **extra):
            del prompt, width, height, steps, seed
            called.append(extra.get("done_steps") or 0)
            raise ImagePaused(
                "preempt",
                ImageHold(stage="step", step=1, latent=np.arange(4, dtype=np.float32), times=[0.2]),
            )

        def close(self) -> None:
            return None

    runtime = _Runtime()
    reply = MemoryReply()
    state = {"image_home": tmp_path, "preempt": preempt, "cancel": threading.Event()}
    handle_image(reply, runtime, {"prompt": "a cat", "seed": 1}, None, None, state)
    assert called == [0]
    assert reply.body["status"] == "in_progress"
    assert reply.body["stage"] == "step"
    assert state["image_hold"]["hold"].hold.step == 1


def test_cancel_before_the_draw_skips_the_runtime(tmp_path) -> None:
    class _Runtime:
        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **_extra):
            del prompt, width, height, steps, seed
            raise AssertionError("draw")

        def close(self) -> None:
            return None

    reply = MemoryReply()
    state = {"image_home": tmp_path, "preempt": threading.Event(), "cancel": threading.Event()}
    state["cancel"].set()
    handle_image(reply, _Runtime(), {"prompt": "a cat", "seed": 1}, None, None, state)
    assert reply.body["status"] == "in_progress"
    assert state.get("image_hold") is None


def test_cancel_drops_the_checkpoint_and_skips_the_remaining_image(tmp_path) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **_extra):
            del prompt, width, height, steps
            self.seeds.append(seed)
            if seed == 8:
                raise ImagePaused("cancel", ImageHold(stage="step", step=1, latent=_picture(1)))
            timing = {"total": 1.0, "caption": 0.1, "vae": 0.2, "steps": [0.1]}
            return _picture(seed), "digest", timing

    runtime = _Runtime()
    reply = MemoryReply()
    state = {"image_home": tmp_path, "preempt": threading.Event(), "cancel": threading.Event()}
    handle_image(reply, runtime, {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert reply.body["status"] == "in_progress"
    assert len(reply.body["data"]) == 1
    assert state.get("image_hold") is None
    assert (tmp_path / "images" / f"{reply.body['data'][0]['id']}.png").is_file()
    handle_image(reply, runtime, {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert runtime.seeds == [7, 8, 7, 8]


def test_a_different_prompt_drops_the_paused_image(tmp_path) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int, **_extra):
            del prompt, width, height, steps, seed
            self.calls += 1
            timing = {"total": 1.0, "caption": 0.1, "vae": 0.2, "steps": [0.1]}
            return _picture(1), "digest", timing

    state = {
        "image_home": tmp_path,
        "image_hold": ImageJob(
            id="image_old",
            created=1,
            prompt="a dog",
            seed=0,
            width=1024,
            height=1024,
            count=1,
            quality="high",
            hold=ImageHold(stage="step", step=3, latent=_picture(1)),
        ),
    }
    reply = MemoryReply()
    runtime = _Runtime()
    handle_image(reply, runtime, {"prompt": "a cat"}, None, None, state)
    assert runtime.calls == 1
    assert "status" not in reply.body
    assert reply.body["created"] != 1


def test_close_drops_the_device_buffers() -> None:
    runtime = Runtime.__new__(Runtime)
    runtime._closed = False
    plane = Plane.__new__(Plane)
    plane.data = object()
    plane.scale = object()
    runtime.weights = {"w": plane}
    scratch = Scratch.__new__(Scratch)
    scratch.text_h = object()
    runtime.scratch = scratch
    runtime.ones = object()

    class _Gpu:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    gpu = _Gpu()
    runtime.gpu = gpu
    runtime.close()
    runtime.close()
    assert gpu.closed
    assert plane.data is None
    assert plane.scale is None
    assert runtime.weights == {}
    assert runtime.scratch is None
    assert runtime.ones is None
    assert scratch.text_h is None


def test_sample_total_counts_finished_layers() -> None:
    assert CAPTION_PIECES == 36
    assert STEP_PIECES == 35
    assert DECODER_PIECES == 25
    assert sample_total(steps=8, done_steps=0) == 36 + 8 * 35 + 25
    assert sample_total(steps=8, done_steps=4) == 36 + 4 * 35 + 25
    assert sample_total(steps=8, done_steps=8) == 25
    with pytest.raises(ValueError):
        sample_total(steps=0, done_steps=0)


def test_sample_row_names_the_stage_and_counts_layers() -> None:
    store: dict[str, object] = {}
    progress = ImageProgress(_bars(store), "16x16")
    progress.open(4)
    progress.stage("caption")
    progress.advance()
    progress.advance()
    handle = store["handle"]
    assert store["phase"] == "sample"
    assert store["unit"] == "layers"
    assert store["total"] == 4
    assert store["lines"] == ["16x16"]
    assert handle.detail == "caption"
    assert handle.done == 2
    progress.note("step 1/8 0.10s")
    progress.close(progress.summary())
    assert store["lines"] == ["16x16", "step 1/8 0.10s"]
    assert store["summary"] == "sample 2/4 layers"
    assert store["closed"] == 2


def test_generate_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def _run(self, prompt, width, height, steps, seed, **kwargs):
        del self, prompt, width, height, steps, seed
        progress = kwargs["progress"]
        progress.stage("step 1/8")
        for _ in range(progress.total):
            progress.advance()
        return _picture(1), "digest", {"total": 3.5, "caption": 0.2, "vae": 0.4, "steps": [0.1]}

    monkeypatch.setattr(Runtime, "_run", _run)
    runtime = Runtime.__new__(Runtime)
    runtime.scratch = _covering_scratch()
    runtime.console = _bars(store)
    pixels, digest, timing = runtime.generate("a cat", 16, 16, 8, 0)
    assert digest == "digest"
    assert timing["total"] == 3.5
    assert pixels.shape == (8, 8, 3)
    total = sample_total(steps=8, done_steps=0)
    assert store["total"] == total
    assert store["lines"] == ["16x16", "generate 3.50s caption 0.20s vae 0.40s"]
    assert store["handle"].detail == "step 1/8"
    assert store["summary"] == f"sample {total}/{total} layers"
    assert store["closed"] == total


def test_a_paused_generate_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def _run(self, prompt, width, height, steps, seed, **kwargs):
        del self, prompt, width, height, steps, seed, kwargs
        raise ImagePaused("preempt", ImageHold(stage="step", step=4))

    monkeypatch.setattr(Runtime, "_run", _run)
    runtime = Runtime.__new__(Runtime)
    runtime.scratch = _covering_scratch()
    runtime.console = _bars(store)
    latent = np.zeros((16, 2, 2), dtype=np.float32)
    with pytest.raises(ImagePaused):
        runtime.generate("a cat", 16, 16, 8, 0, done_steps=4, latent=latent)
    assert store["total"] == sample_total(steps=8, done_steps=4)
    assert store["summary"] == "sample paused"
    assert store["lines"] == ["16x16"]


def test_a_cancelled_generate_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def _run(self, prompt, width, height, steps, seed, **kwargs):
        del self, prompt, width, height, steps, seed, kwargs
        raise ImagePaused("cancel", ImageHold(stage="caption"))

    monkeypatch.setattr(Runtime, "_run", _run)
    runtime = Runtime.__new__(Runtime)
    runtime.scratch = _covering_scratch()
    runtime.console = _bars(store)
    with pytest.raises(ImagePaused):
        runtime.generate("a cat", 16, 16, 8, 0)
    assert store["summary"] == "sample cancelled"


def test_a_failed_generate_closes_the_sample_row(monkeypatch) -> None:
    store: dict[str, object] = {}

    def _run(self, prompt, width, height, steps, seed, **kwargs):
        del self, prompt, width, height, steps, seed, kwargs
        raise RuntimeError("latent is not finite")

    monkeypatch.setattr(Runtime, "_run", _run)
    runtime = Runtime.__new__(Runtime)
    runtime.scratch = _covering_scratch()
    runtime.console = _bars(store)
    with pytest.raises(RuntimeError, match="latent is not finite"):
        runtime.generate("a cat", 16, 16, 8, 0)
    assert store["summary"] == "sample failed"


def test_generate_without_a_console_prints_the_lines(monkeypatch, capsys) -> None:
    def _run(self, prompt, width, height, steps, seed, **kwargs):
        del self, prompt, width, height, steps, seed
        kwargs["progress"].note("step 1/8 0.10s")
        return _picture(1), "digest", {"total": 1.25, "caption": 0.2, "vae": 0.3, "steps": [0.1]}

    monkeypatch.setattr(Runtime, "_run", _run)
    runtime = Runtime.__new__(Runtime)
    runtime.scratch = _covering_scratch()
    runtime.console = None
    runtime.generate("a cat", 16, 16, 8, 0)
    text = capsys.readouterr().out
    assert text == "16x16\nstep 1/8 0.10s\ngenerate 1.25s caption 0.20s vae 0.30s\n"


def test_load_opens_a_byte_row(monkeypatch, tmp_path) -> None:
    store: dict[str, object] = {}

    def fake_load(path, gpu, on_bytes=None):
        del path, gpu
        assert on_bytes is not None
        on_bytes(4)
        on_bytes(6)
        return {"w": object()}

    monkeypatch.setattr("z_image_turbo.runtime.weight_bytes", lambda path: 10)
    monkeypatch.setattr("z_image_turbo.runtime.load", fake_load)
    runtime = Runtime.__new__(Runtime)
    runtime.console = _bars(store)
    runtime.gpu = object()
    assert set(runtime._load(tmp_path)) == {"w"}
    assert store["phase"] == "load"
    assert store["unit"] == "bytes"
    assert store["total"] == 10
    assert store["detail"] == tmp_path.name
    assert store["summary"] == "load 1 tensors"
    assert store["closed"] == 10


def test_a_failed_load_closes_the_byte_row(monkeypatch, tmp_path) -> None:
    store: dict[str, object] = {}

    def fake_load(path, gpu, on_bytes=None):
        del path, gpu, on_bytes
        raise RuntimeError("missing")

    monkeypatch.setattr("z_image_turbo.runtime.weight_bytes", lambda path: 10)
    monkeypatch.setattr("z_image_turbo.runtime.load", fake_load)
    runtime = Runtime.__new__(Runtime)
    runtime.console = _bars(store)
    runtime.gpu = object()
    with pytest.raises(RuntimeError, match="missing"):
        runtime._load(tmp_path)
    assert store["summary"] == "load failed"


def test_a_plain_console_keeps_the_load_print(monkeypatch, tmp_path) -> None:
    def fake_load(path, gpu, on_bytes=None):
        del path, gpu
        assert on_bytes is None
        return {}

    monkeypatch.setattr("z_image_turbo.runtime.weight_bytes", lambda path: 10)
    monkeypatch.setattr("z_image_turbo.runtime.load", fake_load)
    runtime = Runtime.__new__(Runtime)
    runtime.console = _Plain()
    runtime.gpu = object()
    assert runtime._load(tmp_path) == {}


def test_quantize_accepts_one_safetensors_file(tmp_path: Path) -> None:
    weights = tmp_path / "custom.safetensors"
    matrix = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    _write_safetensors(
        weights,
        {
            "layers.0.weight": matrix,
            "cap_pad_token": np.ones(4, dtype=np.float32),
        },
    )
    tokenizer = tmp_path / "tokenizer-tree"
    (tokenizer / "tokenizer").mkdir(parents=True)
    dest = tmp_path / "packed"
    quantize(weights, dest, tokenizer_source=tokenizer)
    body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert body["complete"] is True
    assert body["source"] == str(tokenizer.resolve())
    assert body["tensors"]["layers.0.weight"]["plane"] == "fp4"
    assert body["tensors"]["cap_pad_token"]["plane"] == "f16"
    assert body["format"] == "z-image-turbo-fp4"


def test_quantize_int8_packs_from_the_float_weights(tmp_path: Path) -> None:
    weights = tmp_path / "custom.safetensors"
    matrix = np.arange(8 * 32, dtype=np.float32).reshape(8, 32)
    _write_safetensors(weights, {"layers.0.weight": matrix})
    tokenizer = tmp_path / "tokenizer-tree"
    (tokenizer / "tokenizer").mkdir(parents=True)
    dest = tmp_path / "packed"
    quantize(weights, dest, tokenizer_source=tokenizer, kind="int8")
    body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    entry = body["tensors"]["layers.0.weight"]
    assert body["format"] == "z-image-turbo-int8"
    assert body["tile_n"] == 128
    assert entry["plane"] == "int8"
    assert entry["padded_n"] == 128
    assert entry["stored_k"] == 256
    codes = np.frombuffer((dest / entry["file"]).read_bytes(), dtype=np.int8)
    assert codes.size == 128 * 256


def test_quantize_accepts_a_directory_of_safetensors(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    folder.mkdir()
    _write_safetensors(
        folder / "a.safetensors",
        {"layers.0.weight": np.ones((64, 32), dtype=np.float32)},
    )
    _write_safetensors(
        folder / "b.safetensors",
        {"layers.1.weight": np.ones((64, 32), dtype=np.float32)},
    )
    dest = tmp_path / "packed"
    quantize(folder, dest, tokenizer_source=tmp_path / "tok")
    body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert set(body["tensors"]) == {"layers.0.weight", "layers.1.weight"}


def test_a_custom_safetensors_file_is_its_own_registry_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from z_image_turbo.commands import main

    def models_dir(root: Path | None = None) -> Path:
        del root
        return tmp_path / "models"

    monkeypatch.setattr("favlo.registry.models_dir", models_dir)
    monkeypatch.setattr(
        "z_image_turbo.source.source_dir",
        lambda: tmp_path / "models" / "Z-Image-Turbo",
    )
    weights = tmp_path / "incoming" / "example-model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"st")
    seen: dict[str, Path] = {}

    def quantize(
        origin: Path,
        dest: Path,
        tokenizer_source: Path | None = None,
        kind: str = "fp4",
    ) -> Path:
        del tokenizer_source
        seen["origin"] = origin
        seen["dest"] = dest
        seen["kind"] = kind
        dest.mkdir(parents=True)
        (dest / "manifest.json").write_text("{}\n")
        return dest

    monkeypatch.setattr("z_image_turbo.record.quantize", quantize)
    assert main(["quantize"]) == 0
    assert seen["dest"] == tmp_path / "models" / "Z-Image-Turbo-fp4"
    assert main(["quantize", "--weights", str(weights)]) == 0
    assert seen["dest"] == tmp_path / "models" / "example-model-fp4"
    named = main(["quantize", "--weights", str(weights), "--name", "example-model"])
    assert named == 0
    assert seen["dest"] == tmp_path / "models" / "example-model"
    registry = json.loads((tmp_path / "models" / "registry.json").read_text())
    assert registry["models"]["example-model"]["alias"] is None
    assert registry["models"]["example-model"]["source"] == str(weights)
    refused = main(
        ["quantize", "--weights", str(weights), "--name", "example-model", "--out", "packed"]
    )
    assert refused == 1
    spaced = tmp_path / "bad name.safetensors"
    spaced.write_bytes(b"st")
    assert main(["quantize", "--weights", str(spaced)]) == 1
    assert main(["quantize", "--int8"]) == 0
    assert seen["dest"] == tmp_path / "models" / "Z-Image-Turbo-int8"
    assert seen["kind"] == "int8"
    assert main(["quantize", "--weights", str(weights), "--int8"]) == 0
    assert seen["dest"] == tmp_path / "models" / "example-model-int8"


def test_load_counts_bytes_and_skips_the_print(tmp_path, capsys) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "a.f16").write_bytes(b"\x00\x01")
    (weights / "b.fp4").write_bytes(b"\xab")
    (weights / "b.scale").write_bytes(b"\xcd\xce")
    body = {
        "complete": True,
        "tensors": {
            "a": {"plane": "f16", "shape": [1], "file": "weights/a.f16"},
            "b": {
                "plane": "fp4",
                "shape": [1, 2],
                "file": "weights/b.fp4",
                "scale": "weights/b.scale",
                "logical_k": 2,
                "stored_k": 32,
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    assert weight_bytes(tmp_path) == 5
    seen: list[int] = []
    planes = load(tmp_path, _Upload(), on_bytes=seen.append)
    assert seen == [2, 3]
    assert set(planes) == {"a", "b"}
    assert capsys.readouterr().out == ""
    load(tmp_path, _Upload())
    assert capsys.readouterr().out == "load 0/2\nload 2 tensors\n"


def _write_safetensors(path: Path, arrays: dict[str, np.ndarray]) -> None:
    import struct

    header: dict[str, object] = {}
    offset = 0
    blobs: list[bytes] = []
    for name, array in arrays.items():
        raw = np.ascontiguousarray(array).tobytes()
        dtype = "F32" if array.dtype == np.float32 else "F16"
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {}
    payload = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)


class _Row:
    def __init__(self) -> None:
        self.done = 0
        self.detail = ""


class _Plain:
    shows_bars = False


class _Upload:
    def keep(self, value):
        return value

    def upload(self, value):
        return value


def _bars(store: dict[str, object]):
    class _Console:
        shows_bars = True

        def begin(self, phase: str, **kwargs: object) -> _Row:
            store["phase"] = phase
            store["unit"] = kwargs["unit"]
            store["total"] = kwargs["total"]
            store["detail"] = kwargs["detail"]
            opened = _Row()
            store["handle"] = opened
            return opened

        def end(self, progress: _Row, summary: str) -> None:
            store["summary"] = summary
            store["closed"] = progress.done

        def log(self, line: str) -> None:
            lines = store.setdefault("lines", [])
            assert isinstance(lines, list)
            lines.append(line)

    return _Console()
