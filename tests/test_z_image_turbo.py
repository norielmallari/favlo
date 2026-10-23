"""Pack, schedule, and request-field checks. They do not dispatch Metal."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from z_image_turbo.chat import ImageRequestError, _fields, handle_image
from z_image_turbo.formats import dequant_fp4, pack_fp4, stored_k
from z_image_turbo.png import encode_png
from z_image_turbo.record import Plane
from z_image_turbo.runtime import Runtime, _patchify, _unpatchify
from z_image_turbo.schedule import sigmas, timestep
from z_image_turbo.scratch import MAX_EDGE, Scratch


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
    class _Handler:
        def __init__(self) -> None:
            self.status = 0
            self.body: dict[str, object] = {}

        def _json(self, status: int, body: dict[str, object]) -> None:
            self.status = status
            self.body = body

    class _Runtime:
        def generate(self, prompt: str, width: int, height: int, steps: int, seed: int) -> tuple:
            del prompt, width, height, steps
            timing = {"total": 1.0, "caption": 0.1, "vae": 0.2, "steps": [0.1]}
            return f"png-{seed}".encode(), "digest", timing

    handler = _Handler()
    state = {"image_home": tmp_path}
    handle_image(handler, _Runtime(), {"prompt": "a cat", "n": 2, "seed": 7}, None, None, state)
    assert handler.status == 200
    data = handler.body["data"]
    assert isinstance(data, list)
    assert len(data) == 2
    assert base64.b64decode(data[0]["b64_json"]) == b"png-7"
    assert base64.b64decode(data[1]["b64_json"]) == b"png-8"
    assert (tmp_path / "images" / f"{data[0]['id']}.png").read_bytes() == b"png-7"
    assert (tmp_path / "images" / f"{data[1]['id']}.png").read_bytes() == b"png-8"
    timings = handler.body["generation_seconds"]
    assert isinstance(timings, list)
    assert len(timings) == 2
    assert handler.body["output_format"] == "png"
    assert handler.body["background"] == "opaque"
    assert handler.body["size"] == "1024x1024"
    assert "usage" not in handler.body
    assert "revised_prompt" not in data[0]
    handle_image(handler, _Runtime(), {"prompt": "a cat"}, None, None, state)
    assert isinstance(handler.body["generation_seconds"], dict)


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
