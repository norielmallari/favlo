"""Pack, schedule, and request-field checks. They do not dispatch Metal."""

from __future__ import annotations

import base64
import threading

import numpy as np
import pytest
from z_image_turbo.chat import ImageRequestError, _fields, handle_image
from z_image_turbo.formats import dequant_fp4, pack_fp4, stored_k
from z_image_turbo.hold import ImageHold, ImageJob, ImagePaused
from z_image_turbo.png import ZLIB_LEVEL, encode_png, pack_rgb, unpack_rgb, write_png
from z_image_turbo.record import Plane
from z_image_turbo.runtime import Runtime, _patchify, _unpatchify
from z_image_turbo.schedule import sigmas, timestep
from z_image_turbo.scratch import MAX_EDGE, Scratch

from favlo.reply import MemoryReply


def test_stored_k_pads_a_conv_row() -> None:
    assert stored_k(144) == 256
    assert stored_k(256) == 256
    assert stored_k(1152) == 1280


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
    assert (fields.width, fields.height, fields.seed, fields.count) == (1024, 1024, 0, 1)
    assert fields.quality == "high"
    assert fields.background == "opaque"
    assert fields.output_format == "png"
    assert fields.size == "1024x1024"
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
    assert fields.size == "1024x1024"
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
    assert reply.body["size"] == "1024x1024"
    assert "usage" not in reply.body
    assert "status" not in reply.body
    assert "revised_prompt" not in data[0]
    handle_image(reply, _Runtime(), {"prompt": "a cat"}, None, None, state)
    assert isinstance(reply.body["generation_seconds"], dict)


def _picture(seed: int) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = seed
    return image


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
    assert state["image_hold"].id == reply.body["id"]
    assert runtime.seeds == [7, 8]
    preempt.clear()
    handle_image(reply, runtime, {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert "status" not in reply.body
    assert len(reply.body["data"]) == 2
    assert runtime.seeds == [7, 8]
    assert "image_hold" not in state


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
    assert "image_hold" not in state
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
