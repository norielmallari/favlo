"""Records stored beside generated media."""

from __future__ import annotations

import os
import time

import pytest

from favlo.images import sweep_images
from favlo.media_meta import drop_media, load_media, save_media


def test_a_record_round_trips(tmp_path) -> None:
    saved = save_media(
        tmp_path,
        "image",
        "image_one",
        {"prompt": "a cat", "seed": 4, "size": "720x1280"},
        "example-model",
    )
    assert saved["model"] == "example-model"
    assert saved["seed"] == 4
    assert saved["prompt"] == "a cat"
    assert load_media("image_one", home=tmp_path) == saved
    assert load_media("missing", home=tmp_path) is None


def test_a_dotted_id_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_media(tmp_path, "music", "music_old.wav", {"prompt": "solo", "seed": 1}, "")


def test_sweep_drops_the_record(tmp_path) -> None:
    save_media(tmp_path, "image", "image_old", {"prompt": "a cat", "seed": 3}, "")
    save_media(tmp_path, "music", "music_old", {"prompt": "solo", "seed": 3, "lyrics": ""}, "")
    folder = tmp_path / "images"
    png = folder / "image_old.png"
    png.write_bytes(b"png")
    old = time.time() - 90_000
    os.utime(png, (old, old))
    sweep_images(home=tmp_path, now=time.time())
    assert not png.exists()
    assert load_media("image_old", home=tmp_path) is None
    assert load_media("music_old", home=tmp_path)["seed"] == 3
    drop_media(tmp_path, "music", "music_old")
    assert load_media("music_old", home=tmp_path) is None
