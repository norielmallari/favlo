"""Gallery media metadata sits beside each media file."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.gallery_metadata import (  # noqa: E402
    METADATA_DIR,
    build_index_sidecar,
    from_generation,
    kind_for,
    merge_analyzed,
    move_sidecar,
    prompt_text,
    read_sidecar,
    remove_sidecar,
    sidecar_path,
    write_sidecar,
)
from web_api.gallery_vision import parse_json_object  # noqa: E402


def test_sidecar_lives_beside_the_media_file(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    root.mkdir()
    side = sidecar_path(root, "album/media.mp3")
    assert side == root / "album" / "media.mp3.json"
    assert sidecar_path(root, "shot.png") == root / "shot.png.json"
    assert side.name == "media.mp3.json"
    assert side.name != "media.json"


def test_kinds() -> None:
    assert kind_for("a.MP3") == "audio"
    assert kind_for("b.mp4") == "video"
    assert kind_for("c.JPEG") == "image"
    assert kind_for("notes.txt") is None


def test_walk_media_skips_metadata_dir(tmp_path: Path) -> None:
    from web_api.folder_port import LocalFolder

    root = tmp_path / "lib"
    nested = root / "album"
    nested.mkdir(parents=True)
    (root / "still.png").write_bytes(b"png")
    (root / "track.mp3").write_bytes(b"id3")
    (nested / "clip.mp4").write_bytes(b"ftyp")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    write_sidecar(root, "track.mp3", build_index_sidecar(root / "track.mp3", relative="track.mp3", mtime_iso="t"))
    write_sidecar(
        root,
        "album/clip.mp4",
        build_index_sidecar(nested / "clip.mp4", relative="album/clip.mp4", mtime_iso="t"),
    )
    folder = LocalFolder(str(root))
    found = folder.walk_media()
    paths = {row["path"] for row in found["media"]}
    assert paths == {"still.png", "track.mp3", "album/clip.mp4"}
    assert (root / "track.mp3.json").is_file()
    assert (nested / "clip.mp4.json").is_file()
    assert not (root / METADATA_DIR).exists()
    by_path = {row["path"]: row for row in found["media"]}
    assert by_path["track.mp3"]["has_sidecar"] is True
    assert by_path["still.png"]["has_sidecar"] is False
    assert by_path["album/clip.mp4"]["has_sidecar"] is True
    assert by_path["album/clip.mp4"]["kind"] == "video"


def test_create_folder_does_not_seed_a_metadata_dir(tmp_path: Path) -> None:
    from web_api.folder_port import LocalFolder

    root = tmp_path / "lib"
    root.mkdir()
    folder = LocalFolder(str(root))
    created = folder.create_folder("Albums")
    assert created["path"] == "Albums"
    assert (root / "Albums").is_dir()
    assert not (root / "Albums" / METADATA_DIR).exists()


def test_index_sidecar_and_prompt(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    (root / "trips").mkdir(parents=True)
    media = root / "trips" / "beach_sunset.mp3"
    media.write_bytes(b"id3")
    payload = build_index_sidecar(
        media,
        relative="trips/beach_sunset.mp3",
        mtime_iso="2026-01-02T03:04:05Z",
        duration_seconds=12.5,
    )
    assert payload["filename"] == "beach_sunset.mp3"
    assert payload["kind"] == "audio"
    assert payload["source"] == "index"
    assert "audio" in payload["tags"]
    written = write_sidecar(root, "trips/beach_sunset.mp3", payload)
    assert written == root / "trips" / "beach_sunset.mp3.json"
    assert read_sidecar(root, "trips/beach_sunset.mp3")["title"]
    prompt = prompt_text(payload)
    assert "beach_sunset.mp3" in prompt
    assert "audio" in prompt


def test_move_and_remove_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    (root / "album").mkdir(parents=True)
    media = root / "album" / "shot.png"
    media.write_bytes(b"png")
    write_sidecar(root, "album/shot.png", build_index_sidecar(media, relative="album/shot.png", mtime_iso="t"))
    assert (root / "album" / "shot.png.json").is_file()
    move_sidecar(root, "album/shot.png", "shot.png")
    assert read_sidecar(root, "shot.png") is not None
    assert read_sidecar(root, "album/shot.png") is None
    assert not (root / "album" / "shot.png.json").exists()
    remove_sidecar(root, "shot.png")
    assert read_sidecar(root, "shot.png") is None
    assert not (root / "shot.png.json").exists()


def test_folder_rename_keeps_per_folder_metadata(tmp_path: Path) -> None:
    from web_api.folder_port import LocalFolder

    root = tmp_path / "lib"
    album = root / "album"
    album.mkdir(parents=True)
    media = album / "shot.png"
    media.write_bytes(b"png")
    write_sidecar(root, "album/shot.png", build_index_sidecar(media, relative="album/shot.png", mtime_iso="t"))
    folder = LocalFolder(str(root))
    folder.rename_folder("album", "trips")
    assert (root / "trips" / "shot.png.json").is_file()
    assert read_sidecar(root, "trips/shot.png") is not None
    assert not (root / "album").exists()


def test_vision_json_parse() -> None:
    assert parse_json_object('{"title": "Horse"}')["title"] == "Horse"
    assert parse_json_object('```json\n{"title": "Horse"}\n```')["title"] == "Horse"


def test_from_generation_and_merge() -> None:
    generated = from_generation(
        filename="shot.png",
        kind="image",
        generation={"model": "example-model", "prompt": "a horse", "seed": 9},
    )
    assert generated["source"] == "generate"
    assert generated["generation"]["seed"] == 9
    assert "a horse" in prompt_text(generated)
    merged = merge_analyzed(
        generated,
        {
            "filename": "shot.png",
            "kind": "image",
            "title": "Horse",
            "description": "A horse.",
            "tags": ["horse"],
            "objects": ["horse"],
            "style": "photo",
            "source": "index",
        },
    )
    assert merged["generation"]["prompt"] == "a horse"
    assert merged["source"] == "generate"
    assert merged["style"] == "photo"

