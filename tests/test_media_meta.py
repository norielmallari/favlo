"""Standardized metadata stored for generated media."""

from __future__ import annotations

import os
import time

import pytest

from favlo.images import sweep_images
from favlo.media_meta import drop_media, load_media, media_relative, save_media
from favlo.media_metadata import METADATA_DIR, merge_analyzed, prompt_text


def test_a_record_round_trips(tmp_path) -> None:
    saved = save_media(
        tmp_path,
        "image",
        "image_one",
        {"prompt": "a cat", "seed": 4, "size": "720x1280"},
        "example-model",
    )
    assert saved["source"] == "generate"
    assert saved["kind"] == "image"
    assert saved["generation"]["model"] == "example-model"
    assert saved["generation"]["seed"] == 4
    assert saved["generation"]["prompt"] == "a cat"
    assert saved["description"] == "a cat"
    side = (
        tmp_path
        / "images"
        / METADATA_DIR
        / f"{media_relative('image', 'image_one')}.json"
    )
    assert side.is_file()
    assert not (tmp_path / "images" / "image_one.png.json").exists()
    assert load_media("image_one", home=tmp_path) == saved
    assert load_media("missing", home=tmp_path) is None
    prompt = prompt_text(saved)
    assert "example-model" in prompt
    assert "a cat" in prompt


def test_merge_keeps_generation() -> None:
    existing = {
        "version": 1,
        "filename": "image_x.png",
        "kind": "image",
        "title": "a horse",
        "description": "a horse",
        "tags": ["generated", "image"],
        "place": "",
        "people": [],
        "time": "",
        "objects": [],
        "style": "",
        "colors": [],
        "text_in_image": "",
        "duration_seconds": None,
        "nsfw": False,
        "source": "generate",
        "generation": {"model": "example-model", "prompt": "a horse", "seed": 1},
    }
    analyzed = {
        "filename": "image_x.png",
        "kind": "image",
        "title": "Brown horse",
        "description": "A brown horse in a field.",
        "tags": ["horse"],
        "place": "field",
        "people": [],
        "time": "",
        "objects": ["horse"],
        "style": "photo",
        "colors": ["brown"],
        "text_in_image": "",
        "nsfw": False,
        "source": "index",
    }
    merged = merge_analyzed(existing, analyzed)
    assert merged["source"] == "generate"
    assert merged["generation"]["prompt"] == "a horse"
    assert merged["title"] == "Brown horse"
    assert merged["objects"] == ["horse"]


def test_a_legacy_image_record_still_loads(tmp_path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "image_old.png.json").write_text(
        '{"name": "image_old", "kind": "image", "prompt": "a cat", "seed": 2}\n',
        encoding="utf-8",
    )
    loaded = load_media("image_old", home=tmp_path)
    assert loaded is not None
    assert loaded["generation"]["seed"] == 2
    assert loaded["generation"]["prompt"] == "a cat"
    saved = save_media(tmp_path, "image", "image_old", {"prompt": "a dog", "seed": 5}, "")
    assert saved["generation"]["seed"] == 5
    assert (
        tmp_path / "images" / METADATA_DIR / "image_old.png.json"
    ).is_file()
    assert not (folder / "image_old.png.json").exists()


def test_a_dotted_id_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_media(tmp_path, "music", "music_old.wav", {"prompt": "solo", "seed": 1}, "")


def test_sweep_drops_the_record(tmp_path) -> None:
    save_media(tmp_path, "image", "image_old", {"prompt": "a cat", "seed": 3}, "")
    save_media(tmp_path, "music", "music_old", {"prompt": "solo", "seed": 3, "lyrics": ""}, "")
    folder = tmp_path / "images"
    png = folder / "image_old.png"
    meta = folder / METADATA_DIR / "image_old.png.json"
    png.write_bytes(b"png")
    old = time.time() - 90_000
    os.utime(png, (old, old))
    sweep_images(home=tmp_path, now=time.time())
    assert not png.exists()
    assert not meta.exists()
    assert load_media("image_old", home=tmp_path) is None
    assert load_media("music_old", home=tmp_path)["generation"]["seed"] == 3
    drop_media(tmp_path, "music", "music_old")
    assert load_media("music_old", home=tmp_path) is None


def test_prompt_text_includes_generation() -> None:
    text = prompt_text(
        {
            "filename": "shot.png",
            "kind": "image",
            "title": "Shot",
            "description": "A still.",
            "generation": {"model": "example-model", "prompt": "a horse"},
        }
    )
    assert "Generated with example-model." in text
    assert "Generation prompt: a horse" in text
