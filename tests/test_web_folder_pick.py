"""Native folder picker returns a directory on this Mac."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web-api"))

from web_api.folder_pick import choose_directory  # noqa: E402
from web_api.store import Rejected  # noqa: E402


def test_choose_directory_strips_the_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("web_api.folder_pick.sys.platform", "darwin")
    monkeypatch.setattr(
        "web_api.folder_pick.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="/tmp/gallery/\n"),
    )
    assert choose_directory() == "/tmp/gallery"


def test_choose_directory_treats_cancel_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("web_api.folder_pick.sys.platform", "darwin")
    monkeypatch.setattr(
        "web_api.folder_pick.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="\n"),
    )
    assert choose_directory() is None


def test_choose_directory_rejects_a_failed_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("web_api.folder_pick.sys.platform", "darwin")
    monkeypatch.setattr(
        "web_api.folder_pick.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(Rejected, match="did not open"):
        choose_directory()
