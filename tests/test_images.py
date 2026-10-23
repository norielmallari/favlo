"""Stored PNG expiry. No device and no checkpoint."""

import json
import os

from favlo.images import image_path, load_image, store_image, sweep_images
from favlo.logs import settings_path


def test_a_fresh_image_stays_and_an_old_one_is_removed(tmp_path) -> None:
    settings_path(tmp_path).write_text(
        json.dumps({"image_ttl_seconds": 60}),
        encoding="utf-8",
    )
    images = {"image_old": {"id": "image_old"}, "image_new": {"id": "image_new"}}
    store_image("image_old", b"old-png", home=tmp_path, images=images)
    store_image("image_new", b"new-png", home=tmp_path, images=images)
    old = image_path(tmp_path, "image_old")
    fresh = image_path(tmp_path, "image_new").stat().st_mtime
    expired = fresh - 120
    os.utime(old, (expired, expired))
    sweep_images(images, home=tmp_path, now=fresh)
    assert not old.is_file()
    assert "image_old" not in images
    assert load_image("image_new", home=tmp_path) == b"new-png"
    assert "image_new" in images


def test_a_path_in_the_id_is_refused(tmp_path) -> None:
    try:
        store_image("../clip", b"nope", home=tmp_path)
    except ValueError as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("path id was stored")
    assert load_image("../clip", home=tmp_path) is None
