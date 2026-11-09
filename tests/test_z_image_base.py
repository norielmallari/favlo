"""Sampler and request-field checks. They do not dispatch Metal."""

from __future__ import annotations

import numpy as np
import pytest
from z_image_base.chat import ImageRequestError, Pipeline, _fields
from z_image_base.guide import applies
from z_image_base.runtime import _plan
from z_image_base.schedule import sigmas


def test_default_knots_follow_the_base_scheduler() -> None:
    knots = sigmas()
    assert knots.shape == (29,)
    assert knots[0] == pytest.approx(1.0)
    assert knots[-1] == pytest.approx(0.0)
    assert np.all(np.diff(knots) < 0)


def test_truncation_one_keeps_every_step() -> None:
    knots = sigmas(50, 6.0)
    assert all(applies(float(sigma), 4.0, 1.0) for sigma in knots)


def test_truncation_zero_guides_only_the_noise_knot() -> None:
    assert applies(1.0, 4.0, 0.0)
    assert not applies(0.5, 4.0, 0.0)


def test_guidance_zero_skips_the_negative_velocity() -> None:
    assert not applies(1.0, 0.0, 1.0)


def test_a_sampler_change_is_a_different_image() -> None:
    pipe = Pipeline()
    guided = _fields(
        {"prompt": "a red cube", "guidance": 3, "steps": 28, "negative_prompt": "blur"}
    )
    plain = _fields({"prompt": "a red cube"})
    assert pipe.identity(guided) != pipe.identity(plain)
    assert guided.guidance == 3.0
    assert guided.steps == 28
    assert guided.negative == "blur"
    assert guided.shift == 6.0
    assert guided.cfg_truncation == 1.0
    assert guided.cfg_normalization == 0.0
    assert plain.steps == 28
    assert plain.guidance == 4.0


def test_registry_sampler_overrides_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(root: object = None) -> dict[str, dict[str, dict[str, int | float]]]:
        del root
        return {
            "models": {
                "Z-Image-int8": {
                    "steps": 12,
                    "guidance": 2.5,
                    "shift": 3.0,
                    "cfg_truncation": 0.4,
                    "cfg_normalization": 1.0,
                }
            }
        }

    monkeypatch.setattr("favlo.registry.load_registry", load)
    fields = _fields({"prompt": "a red cube", "model": "Z-Image-int8"})
    assert fields.steps == 12
    assert fields.guidance == 2.5
    assert fields.shift == 3.0
    assert fields.cfg_truncation == 0.4
    assert fields.cfg_normalization == 1.0
    explicit = _fields({"prompt": "a red cube", "model": "Z-Image-int8", "guidance": 4})
    assert explicit.guidance == 4.0


def test_a_bad_registry_step_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(root: object = None) -> dict[str, dict[str, dict[str, int]]]:
        del root
        return {"models": {"Z-Image-int8": {"steps": 0}}}

    monkeypatch.setattr("favlo.registry.load_registry", load)
    with pytest.raises(ImageRequestError, match="registry steps"):
        _fields({"prompt": "a red cube", "model": "Z-Image-int8"})


def test_true_normalization_means_one() -> None:
    fields = _fields({"prompt": "a red cube", "cfg_normalization": True})
    assert fields.cfg_normalization == 1.0


def test_full_truncation_counts_a_second_velocity_on_every_step() -> None:
    knots = sigmas(4, 6.0)
    captions, guided = _plan(knots, 0, 4, 4.0, 1.0)
    assert captions == 2
    assert guided == 4
    quiet, none = _plan(knots, 0, 4, 0.0, 1.0)
    assert quiet == 1
    assert none == 0
