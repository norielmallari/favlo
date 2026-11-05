"""Plain file names under a workspace root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.store import Rejected  # noqa: E402
from web_api.workspace_files import (  # noqa: E402
    release_output,
    remember_source,
    reserve_output,
    rewrite_still,
    source_link,
)


def test_reserve_claims_a_new_png(tmp_path: Path) -> None:
    arguments = {"filename": "star.png", "prompt": "a star"}
    assert reserve_output(tmp_path, "generate_image", arguments) == "star.png"
    assert "filename" not in arguments
    with pytest.raises(Rejected, match="already being made"):
        reserve_output(tmp_path, "generate_image", {"filename": "star.png"})
    release_output(tmp_path, "star.png")
    (tmp_path / "star.png").write_bytes(b"png")
    with pytest.raises(Rejected, match="already in this folder"):
        reserve_output(tmp_path, "generate_image", {"filename": "star.png"})


def test_reserve_refuses_a_bad_name(tmp_path: Path) -> None:
    with pytest.raises(Rejected, match="folder root"):
        reserve_output(tmp_path, "generate_image", {"filename": "../star.png"})
    with pytest.raises(Rejected, match="does not match"):
        reserve_output(tmp_path, "generate_image", {"filename": "star.mp3"})
    with pytest.raises(Rejected, match="mp3 or wav"):
        reserve_output(
            tmp_path,
            "generate_music",
            {"filename": "song.flac", "response_format": "flac"},
        )


def test_a_source_link_is_stored_beside_the_file(tmp_path: Path) -> None:
    remember_source(tmp_path, "star.png", "favlo://images/image_abc.png")
    assert source_link(tmp_path, "star.png") == "favlo://images/image_abc.png"
    assert source_link(tmp_path, "missing.png") == ""


def test_rewrite_still_reads_a_root_image(tmp_path: Path) -> None:
    (tmp_path / "source.png").write_bytes(b"png")
    rewritten = rewrite_still(tmp_path, {"image": "source.png", "prompt": "next"})
    assert rewritten["image"] != "source.png"
    assert rewritten["prompt"] == "next"
    with pytest.raises(Rejected, match="not in this folder"):
        rewrite_still(tmp_path, {"image": "missing.png"})
