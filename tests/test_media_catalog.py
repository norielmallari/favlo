"""The media catalog is registry data. It does not open a device."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from favlo.media_catalog import CatalogError, build_catalog, publish


def test_publish_turns_hyphens_into_underscores() -> None:
    assert publish("z-image-turbo", "generate_image") == "z_image_turbo__generate_image"


def test_a_missing_entry_fails_the_catalog(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "missing-model": {
                        "pipeline": "missing-pipeline",
                        "runtime": "metal41",
                        "alias": None,
                        "variants": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="did not import"):
        build_catalog(tmp_path)


def test_registered_media_tools_publish_without_a_device() -> None:
    names = [name for name, _record, _spec in build_catalog().published()]
    assert "z_image_turbo__generate_image" in names
    assert "wan22_i2v__generate_video" in names
    assert "generate_image" not in names
    assert all("__" in name for name in names)
