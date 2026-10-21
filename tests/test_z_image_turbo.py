"""Pack, schedule, and request-field checks. They do not dispatch Metal."""

from __future__ import annotations

import numpy as np
import pytest
from z_image_turbo.chat import _fields
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


def test_image_request_rejects_n() -> None:
    with pytest.raises(ValueError, match="n must be 1"):
        _fields({"prompt": "a cat", "n": 2})


def test_image_request_default_size() -> None:
    prompt, width, height, seed = _fields({"prompt": "a cat"})
    assert prompt == "a cat"
    assert (width, height, seed) == (1024, 1024, 0)


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
